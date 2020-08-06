import os
import copy
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.naive_classifier.model import PhraseClassifier as AttClassifier
from models.layers.util import print_tensor_stats
from applications.fine_grained_classification.cub_dataset import CUBDataset


def train():
    from torch.utils.tensorboard import SummaryWriter
    # load configs
    output_path = 'applications/fine_grained_classification/att_pred_1'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # make data_loader, model, criterion, optimizer
    dataset = CUBDataset(split='train', val_ratio=0.1)
    train_data_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

    eval_dataset = copy.deepcopy(dataset)
    eval_dataset.split = 'val'
    eval_data_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

    model: AttClassifier = AttClassifier(class_num=len(dataset.att_names), pretrained_backbone=True,
                                         fc_dims=[512, ], use_feats=[2, 4])

    model.train()
    device = torch.device('cuda')
    model.to(device)

    # re-weight loss based on phrase frequency, more weights on positive samples
    class_weights = get_class_weights(dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=class_weights)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=0.0001, weight_decay=0.0001, betas=(0.8, 0.999), eps=1.0e-08)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # make tensorboard writer and dirs
    checkpoint_dir = os.path.join(output_path, 'checkpoints')
    tb_dir = os.path.join(output_path, 'tensorboard')
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
    # best_eval_loss = 100
    best_eval_acc = 0
    best_eval_count = 0
    while epoch <= 100:
        lr = optimizer.param_groups[0]['lr']

        for _, imgs, _, labels in train_data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device, dtype=torch.float)
            pred_labels = model(imgs)
            loss = criterion(pred_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step <= 20:
                print('[%s] epoch-%d step-%d: loss %.4f; lr %.4f'
                      % (time.strftime('%m/%d %H:%M:%S'), epoch, step, loss, lr))
            if epoch == 1 and step == 2:  # debug
                eval_loss, acc = do_eval(model, eval_data_loader, device, criterion)
                print(eval_loss, acc)
            step += 1

        lr_scheduler.step(epoch=epoch)
        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_scalar('train/lr', lr, epoch)
        tb_writer.add_scalar('step', step, epoch)
        tb_writer.add_histogram('pred_labels', pred_labels, epoch)

        eval_loss, acc = do_eval(model, eval_data_loader, device, criterion)
        print('[%s] epoch-%d step-%d: loss %.4f; lr %.4f; eval loss %.4f, acc %.4f '
              % (time.strftime('%m/%d %H:%M:%S'), epoch, step, loss, lr, eval_loss, acc))
        tb_writer.add_scalar('eval/loss', loss, epoch)
        tb_writer.add_scalar('eval/acc', acc, epoch)
        model.train()

        # if eval_loss < best_eval_loss:
        if acc > best_eval_acc:
            print('EVAL: new best!')
            # best_eval_loss = eval_loss
            best_eval_acc = acc
            best_eval_count = 0
            with open(os.path.join(checkpoint_dir, 'epoch.txt'), 'w') as f:
                f.write('BEST: epoch {}\n'.format(epoch))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'BEST_checkpoint.pth'))

        else:
            best_eval_count += 1
            print('EVAL: since last best: %d' % best_eval_count)
            if epoch % 15 == 0:
                with open(os.path.join(checkpoint_dir, 'epoch.txt'), 'a') as f:
                    f.write('LAST: epoch {}\n'.format(epoch))
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'LAST_checkpoint.pth'))

        if best_eval_count % 10 == 0 and best_eval_count > 0:
            print('EVAL: lr decay triggered')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        if best_eval_count % 20 == 0 and best_eval_count > 0:
            print('EVAL: early stop triggered')
            break

        epoch += 1

    tb_writer.close()
    return


def do_eval(model, data_loader, device, criterion):
    model.eval()
    dataset_pred_scores = []
    for _, imgs, _, labels in data_loader:
        with torch.no_grad():
            pred_scores = model(imgs.to(device))
        dataset_pred_scores.append(pred_scores)
    pred_scores = torch.cat(dataset_pred_scores)  # img_num x class_num
    ds = data_loader.dataset
    gt_labels = np.asarray([ds.gt_att_labels[i] for i in ds.img_splits[ds.split]])
    gt_labels_cuda = torch.tensor(gt_labels, dtype=torch.float, device=device)
    loss = criterion(pred_scores, gt_labels_cuda).to('cpu')
    pred = (pred_scores > 0).to('cpu').numpy()
    acc = np.sum(pred == gt_labels) / np.size(gt_labels)
    return loss, acc


def get_class_weights(cub_dataset: CUBDataset):
    att_freq = np.mean(cub_dataset.class_att_labels, axis=0)
    weights = 100.0 / att_freq - 1.0
    return torch.tensor(weights, dtype=torch.float)


def predict_attributes():
    dataset = CUBDataset(split=None, val_ratio=0)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model: AttClassifier = AttClassifier(class_num=len(dataset.att_names), pretrained_backbone=True,
                                         fc_dims=[512, ], use_feats=[2, 4])
    device = 'cuda:0'
    model.to(device)
    model.load_state_dict(torch.load(
        'applications/fine_grained_classification/att_pred_1/checkpoints/BEST_checkpoint.pth'))
    model.eval()

    dataset_pred_scores = []
    for _, imgs, _, labels in tqdm(data_loader, desc='pred scores on images'):
        with torch.no_grad():
            pred_scores = model(imgs.to(device))
        dataset_pred_scores.append(pred_scores)
    pred_scores = torch.cat(dataset_pred_scores).cpu().numpy()  # img_num x class_num
    np.save('applications/fine_grained_classification/att_scores.npy', pred_scores)
    print('att_scores saved.')
    correct = dataset.gt_att_labels == (pred_scores > 0)
    correct_train = correct[dataset.img_splits['train'], :]
    acc_train = np.sum(correct_train) / np.size(correct_train)
    correct_test = correct[dataset.img_splits['test'], :]
    acc_test = np.sum(correct_test) / np.size(correct_test)
    print('acc on train_val: %.4f; on test: %.4f' % (acc_train, acc_test))  # 0.9658, 0.9036
    return


if __name__ == "__main__":
    # train()
    predict_attributes()
