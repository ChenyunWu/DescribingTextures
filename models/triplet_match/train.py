import argparse
import os
import random
import time
import numpy as np
# from tqdm import tqdm
from distutils.dir_util import copy_tree

import torch
from torch.utils.data import DataLoader

from data_api.dataset_api import ImgOnlyDataset, PhraseOnlyDataset, WordEncoder
from data_api.eval_retrieve import log_to_summary
from models.layers.pretrained_word_embed import get_word_embed
from models.layers.img_encoder import build_transforms
from models.triplet_match.model import TripletMatch
from models.triplet_match.eval import do_eval
from models.triplet_match.config_default import C as cfg
from models.triplet_match.config_default import prepare
from models.triplet_match.dataset import TripletTrainData


def train():
    from torch.utils.tensorboard import SummaryWriter
    # load configs
    parser = argparse.ArgumentParser(description="Triplet Matching Training")
    parser.add_argument('-c', '--config_file', default=None, help="path to config file")
    parser.add_argument('-o', '--opts', default=None, nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line. E.g. TRAIN.INIT_LR 0.01",)
    args = parser.parse_args()

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    prepare(cfg)
    cfg.freeze()
    print(cfg.dump())

    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)
    with open(os.path.join(cfg.OUTPUT_PATH, 'train.yml'), 'w') as f:
        f.write(cfg.dump())

    # set random seed
    torch.manual_seed(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    random.seed(cfg.RAND_SEED)

    # make data_loader, model, criterion, optimizer
    dataset = TripletTrainData(split=cfg.TRAIN_SPLIT, neg_img=cfg.LOSS.IMG_SENT_WEIGHTS[0] > 0,
                               neg_lang=cfg.LOSS.IMG_SENT_WEIGHTS[1] > 0, lang_input=cfg.LANG_INPUT)
    data_loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)

    img_datset = ImgOnlyDataset(split=cfg.EVAL_SPLIT, transform=build_transforms(is_train=False),
                                texture_dataset=dataset)
    eval_img_dataloader = DataLoader(img_datset, batch_size=1, shuffle=False)

    phrase_dataset = PhraseOnlyDataset(texture_dataset=dataset)
    eval_phrase_dataloader = DataLoader(phrase_dataset, batch_size=32, shuffle=False)

    word_encoder = WordEncoder()
    model: TripletMatch = TripletMatch(vec_dim=cfg.MODEL.VEC_DIM, neg_margin=cfg.LOSS.MARGIN,
                                       distance=cfg.MODEL.DISTANCE, img_feats=cfg.MODEL.IMG_FEATS,
                                       lang_encoder_method=cfg.MODEL.LANG_ENCODER, word_encoder=word_encoder)

    if cfg.INIT_WORD_EMBED != 'rand' and cfg.MODEL.LANG_ENCODER in ['mean', 'lstm']:
        word_emb = get_word_embed(word_encoder.word_list, cfg.INIT_WORD_EMBED)
        model.lang_embed.embeds.weight.data.copy_(torch.from_numpy(word_emb))
    if len(cfg.LOAD_WEIGHTS) > 0:
        model.load_state_dict(torch.load(cfg.LOAD_WEIGHTS))

    model.train()
    device = torch.device(cfg.DEVICE)
    model.to(device)

    if not cfg.TRAIN.TUNE_RESNET:
        model.resnet_encoder.requires_grad = False
        model.resnet_encoder.eval()
    if not cfg.TRAIN.TUNE_LANG_ENCODER:
        model.lang_embed.requires_grad = False
        model.lang_embed.eval()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.INIT_LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=(cfg.TRAIN.ADAM.ALPHA, cfg.TRAIN.ADAM.BETA),
                                 eps=cfg.TRAIN.ADAM.EPSILON)

    # make tensorboard writer and dirs
    checkpoint_dir = os.path.join(cfg.OUTPUT_PATH, 'checkpoints')
    vis_path = os.path.join(cfg.OUTPUT_PATH, 'eval_visualize_%s_LAST' % cfg.EVAL_SPLIT)
    best_vis_path = os.path.join(cfg.OUTPUT_PATH, 'eval_visualize_%s_BEST' % cfg.EVAL_SPLIT)
    tb_dir = os.path.join(cfg.OUTPUT_PATH, 'tensorboard')
    tb_writer = SummaryWriter(tb_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)

    # training loop
    step = 1
    epoch = 1
    epoch_float = 0.0
    epoch_per_step = cfg.TRAIN.BATCH_SIZE * 1.0 / len(dataset)
    best_eval_metric = 0
    best_metrics = None
    best_eval_count = 0
    early_stop = False
    while epoch <= cfg.TRAIN.MAX_EPOCH and not early_stop:
        # for pos_imgs, pos_langs, neg_imgs, neg_langs in tqdm(data_loader, desc='TRAIN epoch %d' % epoch):
        for pos_imgs, pos_langs, neg_imgs, neg_langs in data_loader:
            pos_imgs = pos_imgs.to(device)
            if neg_imgs is not None and neg_imgs[0] is not None:
                neg_imgs = neg_imgs.to(device)

            verbose = step <= 5 or step % 50 == 0

            neg_img_loss, neg_lang_loss = model(pos_imgs, pos_langs, neg_imgs, neg_langs, verbose=verbose)

            loss = cfg.LOSS.IMG_SENT_WEIGHTS[0] * neg_img_loss + cfg.LOSS.IMG_SENT_WEIGHTS[1] * neg_lang_loss
            loss /= sum(cfg.LOSS.IMG_SENT_WEIGHTS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            tb_writer.add_scalar('train/loss', loss, step)
            tb_writer.add_scalar('train/neg_img_loss', neg_img_loss, step)
            tb_writer.add_scalar('train/neg_lang_loss', neg_lang_loss, step)
            tb_writer.add_scalar('train/lr', lr, step)

            if verbose:
                print('[%s] epoch-%d step-%d: loss %.4f (neg_img: %.4f, neg_lang: %.4f); lr %.1E'
                      % (time.strftime('%m/%d %H:%M:%S'), epoch, step, loss, neg_img_loss, neg_lang_loss, lr))

            # if epoch == 1 and step == 2:  # debug eval
            #     visualize_path = os.path.join(cfg.OUTPUT_PATH, 'eval_visualize_debug')
            #     do_eval(model, eval_img_dataloader, eval_phrase_dataloader, device, split=cfg.EVAL_SPLIT,
            #             visualize_path=visualize_path, add_to_summary_name=None)

            if epoch_float % cfg.TRAIN.EVAL_EVERY_EPOCH < epoch_per_step and epoch_float > 0:
                p2i_result, i2p_result = do_eval(model, eval_img_dataloader, eval_phrase_dataloader, device,
                                                 split=cfg.EVAL_SPLIT, visualize_path=vis_path)

                for m, v in p2i_result.items():
                    tb_writer.add_scalar('eval_p2i/%s' % m, v, step)
                for m, v in i2p_result.items():
                    tb_writer.add_scalar('eval_i2p/%s' % m, v, step)

                eval_metric = p2i_result['mean_average_precision'] + i2p_result['mean_average_precision']
                if eval_metric > best_eval_metric:
                    print('EVAL: new best!')
                    best_eval_metric = eval_metric
                    best_metrics = (p2i_result, i2p_result)
                    best_eval_count = 0
                    copy_tree(vis_path, best_vis_path, update=1)
                    with open(os.path.join(checkpoint_dir, 'epoch_step.txt'), 'w') as f:
                        f.write('BEST: epoch {}, step {}\n'.format(epoch, step))
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'BEST_checkpoint.pth'))

                else:
                    best_eval_count += 1
                    print('EVAL: since last best: %d' % best_eval_count)
                    if epoch_float % cfg.TRAIN.CHECKPOINT_EVERY_EPOCH < epoch_per_step and epoch_float > 0:
                        with open(os.path.join(checkpoint_dir, 'epoch_step.txt'), 'a') as f:
                            f.write('LAST: epoch {}, step {}\n'.format(epoch, step))
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'LAST_checkpoint.pth'))

                if best_eval_count % cfg.TRAIN.LR_DECAY_EVAL_COUNT == 0 and best_eval_count > 0:
                    print('EVAL: lr decay triggered')
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= cfg.TRAIN.LR_DECAY_GAMMA

                if best_eval_count % cfg.TRAIN.EARLY_STOP_EVAL_COUNT == 0 and best_eval_count > 0:
                    print('EVAL: early stop triggered')
                    early_stop = True
                    break

                model.train()
                if not cfg.TRAIN.TUNE_RESNET:
                    model.resnet_encoder.eval()
                if not cfg.TRAIN.TUNE_LANG_ENCODER:
                    model.lang_embed.eval()

            step += 1
            epoch_float += epoch_per_step

        epoch += 1

    tb_writer.close()
    if best_metrics is not None:
        exp_name = '%s:%s' % (cfg.OUTPUT_PATH, cfg.EVAL_SPLIT)
        log_to_summary(exp_name, best_metrics[0], best_metrics[1])
    return best_metrics


if __name__ == "__main__":
    train()
