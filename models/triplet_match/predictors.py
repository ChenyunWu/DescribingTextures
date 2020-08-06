import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL.Image import Image
from wordcloud import WordCloud

from data_api.dataset_api import TextureDescriptionData, PhraseOnlyDataset
from models.layers.util import print_tensor_stats
from models.layers.img_encoder import build_transforms
from models.triplet_match.config_default import C as cfg
from models.triplet_match.model import TripletMatch


def load_model(trained_path='output/triplet_match/c34_bert_l2_s_lr0.00001', model_file='BEST_checkpoint.pth'):
    cfg.merge_from_file(os.path.join(trained_path, 'train.yml'))
    model: TripletMatch = TripletMatch(vec_dim=cfg.MODEL.VEC_DIM, neg_margin=cfg.LOSS.MARGIN,
                                       distance=cfg.MODEL.DISTANCE, img_feats=cfg.MODEL.IMG_FEATS,
                                       lang_encoder_method=cfg.MODEL.LANG_ENCODER)
    model_path = os.path.join(trained_path, 'checkpoints', model_file)
    model.load_state_dict(torch.load(model_path))

    device = torch.device(cfg.DEVICE)
    model.to(device)
    return model, device


def get_phrase_vecs(model: TripletMatch, dataset):
    if type(dataset) is TextureDescriptionData:
        dataset = PhraseOnlyDataset(texture_dataset=dataset)
    assert type(dataset) is PhraseOnlyDataset

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    with torch.no_grad():
        phrase_vecs = list()
        for phrases in tqdm(dataloader, desc='Triplet: getting phrase_vecs in batches'):
        # for phrases in dataloader:
            batch_phrase_vecs = model.lang_encoder(phrases).to('cpu')
            phrase_vecs.append(batch_phrase_vecs)
        phrase_vecs = torch.cat(phrase_vecs)  # phrase_num x vec_dim
    return phrase_vecs


class RetrieveImgFromDesc:
    def __init__(self, model=None, img_transform=None, device='cuda',
                 trained_path='output/triplet_match/c34_bert_l2_s_lr0.00001', model_file='BEST_checkpoint.pth',
                 split_to_phrases=False, dataset=None):
        if model is None:
            model, device = load_model(trained_path, model_file)
            model.eval()
        self.model = model
        self.device = device

        if img_transform is None:
            img_transform = build_transforms(is_train=False)
        self.img_transform = img_transform

        self.split_to_phrases = split_to_phrases
        if dataset is None:
            dataset = TextureDescriptionData(phid_format=None)
        self.dataset = dataset

        self.ph_vec_dict = None
        if self.split_to_phrases:
            ph_vecs = get_phrase_vecs(self.model, self.dataset)
            self.ph_vec_dict = {dataset.phrases[i]: ph_vecs[i] for i in range(len(dataset.phrases))}

        self.img_vecs = dict()
        return

    def get_img_vecs(self, imgs=None, split='test'):
        self.model.eval()
        if imgs is not None:
            img_vecs = list()
            for img in tqdm(imgs, desc='triplet: getting img vecs'):
                imgs = self.img_transform(img).unsqueeze(0)
                with torch.no_grad():
                    img_vec = self.model.img_encoder(imgs.to(self.device)).to('cpu').squeeze()
                img_vecs.append(img_vec)
            img_vecs = torch.stack(img_vecs)
            return img_vecs

        if split in self.img_vecs:
            return self.img_vecs[split]

        img_vecs = list()
        for img_name in tqdm(self.dataset.img_splits[split], desc='Getting img vecs'):
            img = self.dataset.load_img(img_name)
            with torch.no_grad():
                imgs = self.img_transform(img).unsqueeze(0)
                img_vec = self.model.img_encoder(imgs.to(self.device)).to('cpu').squeeze()
                img_vecs.append(img_vec)
        img_vecs = torch.stack(img_vecs)
        self.img_vecs[split] = img_vecs
        return img_vecs

    def __call__(self, desc, img_vecs=None, imgs=None, split='test'):
        self.model.eval()
        if self.split_to_phrases:
            phrases = self.dataset.description_to_phrases(desc)
            ph_vecs = list()
            for ph in phrases:
                if ph in self.ph_vec_dict:
                    ph_vecs.append(self.ph_vec_dict[ph])
                else:
                    with torch.no_grad():
                        ph_vec = self.model.lang_encoder([ph]).to('cpu').squeeze()
                    self.ph_vec_dict[ph] = ph_vec
                    ph_vecs.append(ph_vec)
        else:
            with torch.no_grad():
                ph_vecs = self.model.lang_encoder([desc]).to('cpu')

        img_scores = list()
        with torch.no_grad():
            if img_vecs is not None:
                for img_vec in img_vecs:
                    img_score = 0
                    for ph_vec in ph_vecs:
                        img_score -= self.model.dist_fn(img_vec, ph_vec)
                    img_scores.append(img_score)
            elif imgs is not None:
                for img in imgs:
                    imgs = self.img_transform(img).unsqueeze(0)
                    img_vec = self.model.img_encoder(imgs.to(self.device)).to('cpu').squeeze()
                    img_score = 0
                    for ph_vec in ph_vecs:
                        img_score -= self.model.dist_fn(img_vec, ph_vec)
                    img_scores.append(img_score)
            else:
                img_vecs = self.get_img_vecs(split=split)
                for img_i in range(len(self.dataset.img_splits[split])):
                    img_vec = img_vecs[img_i]
                    img_score = 0
                    for ph_vec in ph_vecs:
                        img_score -= self.model.dist_fn(img_vec, ph_vec)
                    img_scores.append(img_score)
        return np.asarray(img_scores)


class GenImgCaption:
    def __init__(self, model=None, img_transform=None, dataset=None, device='cuda'):
        if model is None:
            model, device = load_model()
        self.model = model
        self.device = device

        if img_transform is None:
            img_transform = build_transforms(is_train=False)
        self.img_transform = img_transform

        if dataset is None:
            dataset = TextureDescriptionData(phid_format=None)
        self.dataset = dataset

        self.ph_vecs = get_phrase_vecs(self.model, self.dataset)
        return

    def __call__(self, img, top_k=5):
        assert type(img) is Image
        self.model.eval()
        with torch.no_grad():
            imgs = self.img_transform(img).unsqueeze(0)
            img_vec = self.model.img_encoder(imgs.to(self.device)).to('cpu').squeeze()
            ph_scores = np.zeros(self.ph_vecs.size(0))
            for ph_i in range(self.ph_vecs.size(0)):
                ph_vec = self.ph_vecs[ph_i]
                # print_tensor_stats(img_vec, 'img_vec')
                # print_tensor_stats(ph_vec, 'ph_vec')
                # d = self.model.dist_fn(img_vec, ph_vec)
                # print_tensor_stats(d, 'distance')
                ph_scores[ph_i] = -self.model.dist_fn(img_vec, ph_vec).to('cpu').numpy()
        sorted_phids = np.argsort(ph_scores * -1.0)
        caption = ', '.join([self.dataset.phid_to_phrase(phid) for phid in sorted_phids[:top_k]])
        return caption


class GenImgWordCloud:
    def __init__(self, model=None, img_transform=None, dataset=None, device='cuda'):
        if model is None:
            model, device = load_model()
        self.model = model
        self.device = device

        if img_transform is None:
            img_transform = build_transforms(is_train=False)
        self.img_transform = img_transform

        if dataset is None:
            dataset = TextureDescriptionData(phid_format=None)
        self.dataset = dataset

        self.ph_vecs = get_phrase_vecs(self.model, self.dataset)
        return

    def _get_ph_scores(self, img):
        assert type(img) is Image
        ph_score_dict = dict()
        self.model.eval()
        with torch.no_grad():
            imgs = self.img_transform(img).unsqueeze(0)
            img_vec = self.model.img_encoder(imgs.to(self.device)).to('cpu').squeeze()
            for ph_i in range(self.ph_vecs.size(0)):
                ph_vec = self.ph_vecs[ph_i]
                ph = self.dataset.phrases[ph_i]
                ph_score_dict[ph] = 1.0 / np.sqrt(self.model.dist_fn(img_vec, ph_vec).to('cpu').numpy())
        return ph_score_dict

    def __call__(self, img, out_path=None, topk=50):
        assert type(img) is Image
        ph_score_dict = self._get_ph_scores(img)
        if topk > 0:
            ph_scores = sorted(ph_score_dict.items(), key=lambda ps: ps[1], reverse=True)
            ph_score_dict = {ph: score for ph, score in ph_scores[:topk]}
        wc = WordCloud(background_color="white", colormap='tab20b', prefer_horizontal=0.9,
                       height=1200, width=1200, min_font_size=25, margin=6,
                       font_path='visualizations/DIN Alternate Bold.ttf')
        wc.generate_from_frequencies(ph_score_dict)
        if out_path is not None:
            wc.to_file(out_path)
        return wc
