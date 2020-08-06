import os
from PIL import Image
import torch
import torch.utils.data as data

from data_api.dataset_api import TextureDescriptionData, img_path
from models.layers.img_encoder import build_transforms

img_feat_path = 'output/cache/resnet101_img_feats'


class PhraseClassifyDataset(data.Dataset, TextureDescriptionData):
    def __init__(self, split='train', is_train=True, cached_resnet_feats=None):
        data.Dataset.__init__(self)
        TextureDescriptionData.__init__(self, phid_format='set')

        self.split = split
        self.is_train = is_train
        self.cached_resnet_feats = cached_resnet_feats
        self.use_cache = self.cached_resnet_feats is not None and len(self.cached_resnet_feats) > 0
        self.transform = None
        if not self.use_cache:
            self.transform = build_transforms(is_train)
        print('PhraseClassifyDataset initialized.')

    def __getitem__(self, img_idx):
        img_data = self.get_split_data(self.split, img_idx=img_idx, load_img=False)
        img_name = img_data['image_name']
        # labels
        phrase_binary_labels = torch.zeros(len(self.phrases))
        for ph_id in img_data['phrase_ids']:
            if ph_id >= 0:
                phrase_binary_labels[ph_id] = 1

        # img or img_feat
        if self.cached_resnet_feats is not None and len(self.cached_resnet_feats) > 0:
            img_feats = list()
            for feat_layer in self.cached_resnet_feats:
                feat_path = os.path.join(img_feat_path, 'layer_%d' % feat_layer, img_name)[:-len('jpg')] + 'pth'
                img_feats.append(torch.load(feat_path).squeeze())
            img_feat = torch.cat(img_feats)

        else:
            img_fpath = os.path.join(img_path, img_name)
            img = Image.open(img_fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            img_feat = img

        return img_name, img_feat, phrase_binary_labels

    def __len__(self):
        return len(self.img_splits[self.split])
