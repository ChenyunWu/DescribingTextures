import argparse
import os
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_api.eval_retrieve import retrieve_eval
from data_api.dataset_api import ImgOnlyDataset, PhraseOnlyDataset, TextureDescriptionData
from models.layers.img_encoder import build_transforms
from models.layers.util import print_tensor_stats
from models.triplet_match.config_default import C as cfg
from models.triplet_match.model import TripletMatch


def predict(model, img_dataloader, phrase_dataloader, device):
    model.eval()
    img_num = len(img_dataloader.dataset)
    phrase_num = len(phrase_dataloader.dataset)
    neg_distances = torch.zeros((img_num, phrase_num))
    with torch.no_grad():
        phrase_vecs = list()
        # for phrases in tqdm(phrase_dataloader, desc='PREDICTION: getting phrase_vecs in batches'):
        for phrases in phrase_dataloader:
            batch_phrase_vecs = model.lang_encoder(phrases).to('cpu')
            phrase_vecs.append(batch_phrase_vecs)
        phrase_vecs = torch.cat(phrase_vecs)  # phrase_num x vec_dim

        img_i = 0
        # for _, imgs in tqdm(img_dataloader, desc='PREDICTION: getting img_vecs and distances'):
        for _, imgs in img_dataloader:
            img_vec = model.img_encoder(imgs.to(device)).to('cpu')[0]
            for ph_i in range(phrase_num):
                ph_vec = phrase_vecs[ph_i]
                neg_distances[img_i, ph_i] = - model.dist_fn(img_vec, ph_vec)
            img_i += 1

    print_tensor_stats(neg_distances, 'pred_scores')
    return neg_distances.numpy()


def do_eval(model, img_dataloader, phrase_dataloader, device, split,
            eval_p2i=True, eval_i2p=True, visualize_path=None, add_to_summary_name=None):
    model.eval()
    pred_scores = predict(model, img_dataloader, phrase_dataloader, device)

    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)
    save_path = os.path.join(visualize_path, 'pred_scores.npy')
    np.save(save_path, pred_scores)

    p2i_result = None
    if eval_p2i:
        p2i_result = retrieve_eval(mode='phrase2img', match_scores=pred_scores, dataset=img_dataloader.dataset.dataset,
                                   split=split, visualize_path=visualize_path, add_to_summary_name=add_to_summary_name)
    i2p_result = None
    if eval_i2p:
        i2p_result = retrieve_eval(mode='img2phrase', match_scores=pred_scores, dataset=img_dataloader.dataset.dataset,
                                   split=split, visualize_path=visualize_path, add_to_summary_name=add_to_summary_name)
    return p2i_result, i2p_result


def main_eval():
    # load configs
    parser = argparse.ArgumentParser(description="Triplet (phrase) retrieval evaluation")
    parser.add_argument('-p', '--trained_path', help="path to trained model (where there is cfg file)",
                        default='output/triplet_match/c34_bert_l2_s_lr0.00001')
    parser.add_argument('-m', '--model_file', help='file name of the cached model ',
                        default='BEST_checkpoint.pth')
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

    dataset = TextureDescriptionData(phid_format=None)
    img_dataset = ImgOnlyDataset(split=cfg.EVAL_SPLIT, transform=build_transforms(is_train=False),
                                texture_dataset=dataset)
    img_dataloader = DataLoader(img_dataset, batch_size=1, shuffle=False)

    phrase_dataset = PhraseOnlyDataset(texture_dataset=dataset)
    phrase_dataloader = DataLoader(phrase_dataset, batch_size=32, shuffle=False)

    model: TripletMatch = TripletMatch(vec_dim=cfg.MODEL.VEC_DIM, neg_margin=cfg.LOSS.MARGIN,
                                       distance=cfg.MODEL.DISTANCE, img_feats=cfg.MODEL.IMG_FEATS,
                                       lang_encoder_method=cfg.MODEL.LANG_ENCODER)

    model_path = os.path.join(args.trained_path, 'checkpoints', args.model_file)
    model.load_state_dict(torch.load(model_path))

    device = torch.device(cfg.DEVICE)
    model.to(device)

    do_eval(model, img_dataloader, phrase_dataloader, device, split=cfg.EVAL_SPLIT,
            visualize_path=os.path.join(args.trained_path, 'eval_visualize_%s' % cfg.EVAL_SPLIT),
            add_to_summary_name=model_path + ':' + cfg.EVAL_SPLIT)


if __name__ == "__main__":
    main_eval()

