import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.naive_classifier.model import PhraseClassifier
from models.naive_classifier.eval import do_eval
from models.naive_classifier.config_default import C as cfg
from models.naive_classifier.dataset import PhraseClassifyDataset


def train():
    from torch.utils.tensorboard import SummaryWriter
    # load configs
    parser = argparse.ArgumentParser(description="Phrase Classification Training")
    parser.add_argument('-c', '--config_file', default=None, help="path to config file")
    parser.add_argument('-o', '--opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line. E.g. TRAIN.INIT_LR 0.01",)
    args = parser.parse_args()

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    print(cfg.dump())

    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)
    with open(os.path.join(cfg.OUTPUT_PATH, 'train.cfg'), 'w') as f:
        f.write(cfg.dump())

    # set random seed
    torch.manual_seed(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    random.seed(cfg.RAND_SEED)

    # make data_loader, model, criterion, optimizer
    dataset = PhraseClassifyDataset(split=cfg.TRAIN_SPLIT, is_train=True, cached_resnet_feats=None)
    train_data_loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True)

    eval_data_loader = None
    if cfg.TRAIN.EVAL_EVERY_EPOCH > 0:
        eval_dataset = PhraseClassifyDataset(split=cfg.EVAL_SPLIT, is_train=False, cached_resnet_feats=None)
        eval_data_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    model: PhraseClassifier = PhraseClassifier(class_num=len(dataset.phrases), pretrained_backbone=True,
                                               fc_dims=cfg.MODEL.FC_DIMS, use_feats=cfg.MODEL.BACKBONE_FEATS)
    if not cfg.TRAIN.TUNE_BACKBONE:
        model.img_encoder.requires_grad = False
        model.img_encoder.eval()

    if len(cfg.MODEL.LOAD_WEIGHTS) > 0:
        model.load_state_dict(torch.load(cfg.MODEL.LOAD_WEIGHTS))

    model.train()
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    # re-weight loss based on phrase frequency, more weights on positive samples
    if cfg.TRAIN.LOSS_REWEIGHT:
        class_weights = get_class_weights(dataset.phrase_freq)
        class_weights = torch.from_numpy(class_weights).to(device)
        criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.INIT_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=(cfg.TRAIN.ADAM.ALPHA, cfg.TRAIN.ADAM.BETA),
                                 eps=cfg.TRAIN.ADAM.EPSILON)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_EPOCH,
                                                   gamma=cfg.TRAIN.LR_DECAY_GAMMA)

    # make tensorboard writer and dirs
    checkpoint_dir = os.path.join(cfg.OUTPUT_PATH, 'checkpoints')
    tb_dir = os.path.join(cfg.OUTPUT_PATH, 'tensorboard')
    tb_writer = SummaryWriter(tb_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # training loop
    step = 1
    epoch = 1
    loss = None
    pred_labels = None
    while epoch <= cfg.TRAIN.MAX_EPOCH:
        lr = optimizer.param_groups[0]['lr']

        for _, imgs, labels in train_data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred_labels = model(imgs)
            loss = criterion(pred_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step <= 20:
                print('[%s] epoch-%d step-%d: loss %.4f; lr %.4f'
                      % (time.strftime('%m/%d %H:%M:%S'), epoch, step, loss, lr))
            # if epoch == 1 and step == 2:  # debug
            #     do_eval(model, eval_data_loader, device, visualize_path=visualize_path, add_to_summary_name='debug')
            step += 1

        lr_scheduler.step(epoch=epoch)
        print('[%s] epoch-%d step-%d: loss %.4f; lr %.4f'
              % (time.strftime('%m/%d %H:%M:%S'), epoch, step, loss, lr))

        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_scalar('train/lr', lr, epoch)
        tb_writer.add_scalar('step', step, epoch)
        tb_writer.add_histogram('pred_labels', pred_labels, epoch)

        vis = None
        if epoch % cfg.TRAIN.CHECKPOINT_EVERY_EPOCH == 0:
            vis = os.path.join(cfg.OUTPUT_PATH, 'eval_visualize_epoch%03d' % epoch)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'epoch%03d.pth' % epoch))

        if epoch % cfg.TRAIN.EVAL_EVERY_EPOCH == 0 and cfg.TRAIN.EVAL_EVERY_EPOCH > 0:
            p2i_result, i2p_result = do_eval(model, eval_data_loader, device, visualize_path=vis)
            for m, v in p2i_result.items():
                tb_writer.add_scalar('eval_p2i/%s' % m, v, epoch)
            for m, v in i2p_result.items():
                tb_writer.add_scalar('eval_i2p/%s' % m, v, epoch)
            model.train()
            if not cfg.TRAIN.TUNE_BACKBONE:
                model.img_encoder.eval()

        epoch += 1

    tb_writer.close()
    visualize_path = os.path.join(cfg.OUTPUT_PATH, 'eval_visualize')
    p2i_result, i2p_result = do_eval(model, eval_data_loader, device, visualize_path=visualize_path,
                                     add_to_summary_name='%s:epoch-%d' % (cfg.OUTPUT_PATH, epoch))
    return p2i_result, i2p_result


def get_class_weights(class_freq):
    freq = np.array(class_freq, dtype=np.float32)
    weights = np.ones_like(freq, dtype=np.float32) * 3222.0 / freq - 1.0
    return weights


if __name__ == "__main__":
    train()
