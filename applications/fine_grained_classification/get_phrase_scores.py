import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from models.naive_classifier.predictors import load_model as cls_load_model
from models.triplet_match.predictors import load_model as tri_load_model
from data_api.dataset_api import PhraseOnlyDataset
from applications.fine_grained_classification.cub_dataset import CUBDataset


def get_phrase_scores_cls(split=None):
    cub_dataset = CUBDataset(split=split)
    data_loader = DataLoader(cub_dataset, batch_size=128, shuffle=False)
    model, device = cls_load_model()
    model.eval()
    dataset_pred_scores = []
    for _, imgs, _, _ in tqdm(data_loader, desc='predicting phrase scores over images'):
        with torch.no_grad():
            pred_scores = model(imgs.to(device)).to('cpu')
        dataset_pred_scores.append(pred_scores)
    pred_scores = torch.cat(dataset_pred_scores)
    return pred_scores


def get_phrase_scores_tri(split=None):
    phrase_dataset = PhraseOnlyDataset()
    phrase_dataloader = DataLoader(phrase_dataset, batch_size=128, shuffle=False)
    cub_dataset = CUBDataset(split=split)
    img_dataloader = DataLoader(cub_dataset, batch_size=1, shuffle=False)
    model, device = tri_load_model()
    neg_distances = torch.zeros((len(cub_dataset), len(phrase_dataset)))

    phrase_vecs = list()
    with torch.no_grad():
        for phrases in tqdm(phrase_dataloader, desc='getting phrase_vecs in batches'):
            batch_phrase_vecs = model.lang_encoder(phrases).to('cpu')
            phrase_vecs.append(batch_phrase_vecs)
    phrase_vecs = torch.cat(phrase_vecs)  # phrase_num x vec_dim

    img_i = 0
    with torch.no_grad():
        for _, imgs, _, _ in tqdm(img_dataloader, desc='getting img_vecs and distances'):
            img_vec = model.img_encoder(imgs.to(device)).to('cpu')[0]
            for ph_i in range(len(phrase_dataset)):
                ph_vec = phrase_vecs[ph_i]
                neg_distances[img_i, ph_i] = - model.dist_fn(img_vec, ph_vec)
            img_i += 1
    return neg_distances


if __name__ == '__main__':
    # phrase_scores_cls = get_phrase_scores_cls().numpy()
    # np.save('applications/fine_grained_classification/phrase_scores_cls.npy', phrase_scores_cls)
    # print('phrase_scores_cls saved.')
    phrase_scores_tri = get_phrase_scores_tri().numpy()
    np.save('applications/fine_grained_classification/phrase_scores_tri.npy', phrase_scores_tri)
    print('phrase_scores_tri saved.')
