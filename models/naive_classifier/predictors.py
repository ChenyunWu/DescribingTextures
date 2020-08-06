import os
import numpy as np
import torch
from tqdm import tqdm
from PIL.Image import Image

from models.naive_classifier.model import PhraseClassifier
from models.naive_classifier.config_default import C as cfg
from data_api.dataset_api import TextureDescriptionData
from models.layers.img_encoder import build_transforms


def load_model(trained_path='output/naive_classify/v1_35_ft2,4_fc512_tuneTrue', model_file='epoch075.pth', dataset=None):
    cfg.merge_from_file(os.path.join(trained_path, 'train.yml'))
    if dataset is None:
        dataset = TextureDescriptionData(phid_format=None)
    model: PhraseClassifier = PhraseClassifier(class_num=len(dataset.phrases), pretrained_backbone=True,
                                               fc_dims=cfg.MODEL.FC_DIMS, use_feats=cfg.MODEL.BACKBONE_FEATS)

    model_path = os.path.join(trained_path, 'checkpoints', model_file)
    model.load_state_dict(torch.load(model_path))

    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    return model, device


class RetrieveImgFromDesc:
    def __init__(self, model=None, img_transform=None, device='cuda', dataset=None):
        if dataset is None:
            dataset = TextureDescriptionData(phid_format=None)
        self.dataset = dataset

        if model is None:
            model, device = load_model(dataset=self.dataset)
            model.eval()
        self.model = model
        self.device = device

        if img_transform is None:
            img_transform = build_transforms(is_train=False)
        self.img_transform = img_transform

        self.img_ph_scores = dict()
        return

    def _get_img_ph_scores_split(self, split='test'):
        if split in self.img_ph_scores:
            return self.img_ph_scores[split]

        img_ph_scores = list()
        for img_name in tqdm(self.dataset.img_splits[split], desc='Getting img_ph_scores %s' % split):
            img = self.dataset.load_img(img_name)
            img = self.img_transform(img).unsqueeze(0)
            with torch.no_grad():
                ph_scores = self.model(img.to(self.device)).to('cpu').numpy().squeeze()
            img_ph_scores.append(ph_scores)
        img_ph_scores = np.stack(img_ph_scores)  # #img x #phrase
        self.img_ph_scores[split] = img_ph_scores
        return img_ph_scores

    def get_img_ph_scores(self, imgs):
        self.model.eval()
        img_ph_scores = list()
        if imgs is not None:
            for img_i, img in enumerate(imgs):
                img = self.img_transform(img).unsqueeze(0)
                with torch.no_grad():
                    ph_scores = self.model(img.to(self.device)).to('cpu').numpy().squeeze()
                img_ph_scores.append(ph_scores)
        img_ph_scores = np.asarray(img_ph_scores)
        return img_ph_scores

    def __call__(self, desc, img_ph_scores=None, imgs=None, split='test'):
        self.model.eval()
        phids = self.dataset.description_to_phids_smart(desc)
        if -1 in phids:
            phids.remove(-1)
        phids = list(phids)

        if img_ph_scores is not None:
            img_scores = np.sum(img_ph_scores[:, phids], axis=1)
        elif imgs is not None:
            img_scores = list()
            for img_i, img in enumerate(imgs):
                img = self.img_transform(img).unsqueeze(0)
                with torch.no_grad():
                    ph_scores = self.model(img.to(self.device)).to('cpu').numpy().squeeze()
                img_score = np.sum(ph_scores[phids])
                img_scores.append(img_score)
        else:
            img_ph_scores = self._get_img_ph_scores_split(split)
            img_scores = list()
            for img_i in range(len(img_ph_scores)):
                img_scores.append(np.sum(img_ph_scores[img_i][phids]))

        img_scores = np.asarray(img_scores)
        return img_scores


class GenImgCaption:
    def __init__(self, model=None, img_transform=None, dataset=None, device='cuda'):
        if dataset is None:
            dataset = TextureDescriptionData(phid_format=None)
        self.dataset = dataset

        if model is None:
            model, device = load_model(dataset=self.dataset)
        self.model = model
        self.device = device

        if img_transform is None:
            img_transform = build_transforms(is_train=False)
        self.img_transform = img_transform

    def __call__(self, img, top_k=5):
        assert type(img) is Image
        self.model.eval()
        img = self.img_transform(img).unsqueeze(0)
        with torch.no_grad():
            ph_scores = self.model(img.to(self.device)).to('cpu').numpy().squeeze()
        sorted_phids = np.argsort(ph_scores * -1.0)
        caption = ', '.join([self.dataset.phid_to_phrase(phid) for phid in sorted_phids[:top_k]])
        return caption

