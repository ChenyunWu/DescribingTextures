import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.naive_classifier.model import PhraseClassifier
from models.naive_classifier.config_default import C as cfg
from models.naive_classifier.dataset import PhraseClassifyDataset
from data_api.eval_retrieve import retrieve_eval


def do_eval(model, data_loader, device, eval_p2i=True, eval_i2p=True, visualize_path=None, add_to_summary_name=None):
    model.eval()
    dataset_pred_scores = []

    for _, imgs, labels in data_loader:
        with torch.no_grad():
            pred_scores = model(imgs.to(device)).to('cpu').numpy()
        dataset_pred_scores.append(pred_scores)
    pred_scores = np.vstack(dataset_pred_scores)  # img_num x class_num

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)
    save_path = os.path.join(visualize_path, 'pred_scores.npy')
    np.save(save_path, pred_scores)

    p2i_result = None
    if eval_p2i:
        p2i_result = retrieve_eval(mode='phrase2img', match_scores=pred_scores, dataset=data_loader.dataset,
                                   split=data_loader.dataset.split, visualize_path=visualize_path,
                                   add_to_summary_name=add_to_summary_name)
    i2p_result = None
    if eval_i2p:
        i2p_result = retrieve_eval(mode='img2phrase', match_scores=pred_scores, dataset=data_loader.dataset,
                                   split=data_loader.dataset.split, visualize_path=visualize_path,
                                   add_to_summary_name=add_to_summary_name)
    return p2i_result, i2p_result


def main_eval():
    # load configs
    parser = argparse.ArgumentParser(description="Classification evaluation")
    parser.add_argument('-p', '--trained_path', help="path to trained model (where there is cfg file)",
                        default='output/naive_classify/v1_36_ft3,4_fc512_tuneTrue')
                        # default='output/naive_classify/v1_35_ft2,4_fc512_tuneTrue')
    parser.add_argument('-m', '--model_file', help='file name of the cached model ',
                        default='epoch075.pth')
    parser.add_argument('-o', '--opts', default=None, nargs=argparse.REMAINDER,
                        help="e.g. EVAL_SPLIT test")
    args = parser.parse_args()

    cfg.merge_from_file(os.path.join(args.trained_path, 'train.yml'))
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # set random seed
    torch.manual_seed(cfg.RAND_SEED)
    np.random.seed(cfg.RAND_SEED)
    random.seed(cfg.RAND_SEED)

    eval_dataset = PhraseClassifyDataset(split=cfg.EVAL_SPLIT, is_train=False, cached_resnet_feats=None)
    eval_data_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)

    model: PhraseClassifier = PhraseClassifier(class_num=len(eval_dataset.phrases), pretrained_backbone=True,
                                               fc_dims=cfg.MODEL.FC_DIMS, use_feats=cfg.MODEL.BACKBONE_FEATS)

    model_path = os.path.join(args.trained_path, 'checkpoints', args.model_file)
    model.load_state_dict(torch.load(model_path))

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    do_eval(model, eval_data_loader, device,
            visualize_path=os.path.join(args.trained_path, 'eval_visualize_%s' % cfg.EVAL_SPLIT),
            add_to_summary_name=model_path + '_' + cfg.EVAL_SPLIT)


if __name__ == "__main__":
    main_eval()
