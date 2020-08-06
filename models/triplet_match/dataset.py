import random
import torch.utils.data as data

from data_api.dataset_api import TextureDescriptionData
from models.layers.img_encoder import build_transforms


class TripletTrainData(data.Dataset, TextureDescriptionData):

    def __init__(self, split='train', lang_input='phrase', neg_img=True, neg_lang=True):
        data.Dataset.__init__(self)
        TextureDescriptionData.__init__(self, phid_format='str')
        self.split = split
        self.lang_input = lang_input
        self.neg_img = neg_img
        self.neg_lang = neg_lang
        self.img_transform = build_transforms(is_train=False)

        self.pos_pairs = list()
        for img_i, img_name in enumerate(self.img_splits[self.split]):
            img_data = self.img_data_dict[img_name]
            if self.lang_input == 'phrase':
                self.pos_pairs += [(img_i, ph) for ph in img_data['phrase_ids']]
            elif self.lang_input == 'description':
                self.pos_pairs += [(img_i, desc_idx) for desc_idx in range(len(img_data['descriptions']))]
            else:
                raise NotImplementedError
        return

    def __len__(self):
        return len(self.pos_pairs)

    def __getitem__(self, pair_i):
        if self.lang_input == 'phrase':
            img_i, pos_lang = self.pos_pairs[pair_i]
            pos_img_data = self.get_split_data(self.split, img_i, load_img=True)
            pos_img = pos_img_data['image']
            pos_img = self.img_transform(pos_img)

            neg_lang = None
            if self.neg_lang:
                while True:
                    neg_lang = random.choice(self.phrases)
                    if neg_lang not in pos_img_data['phrase_ids']:
                        break

            neg_img = None
            if self.neg_img:
                while True:
                    neg_img_name = random.choice(self.img_splits[self.split])
                    neg_img_data = self.img_data_dict[neg_img_name]
                    if pos_lang not in neg_img_data['phrase_ids']:
                        break
                neg_img = self.load_img(neg_img_name)
                neg_img = self.img_transform(neg_img)

        else:  # 'descriptions'
            img_i, desc_i = self.pos_pairs[pair_i]
            pos_img_data = self.get_split_data(self.split, img_i, load_img=True)
            pos_img = pos_img_data['image']
            pos_img = self.img_transform(pos_img)
            pos_lang = pos_img_data['descriptions'][desc_i]

            neg_lang = None
            if self.neg_lang:
                while True:
                    img_name = random.choice(self.img_splits[self.split])
                    if img_name == pos_img_data['image_name']:
                        continue
                    neg_lang = random.choice(self.img_data_dict[img_name]['descriptions'])
                    if neg_lang not in pos_img_data['descriptions']:
                        break

            neg_img = None
            if self.neg_img:
                while True:
                    neg_img_name = random.choice(self.img_splits[self.split])
                    neg_img_data = self.img_data_dict[neg_img_name]
                    if pos_lang not in neg_img_data['descriptions']:
                        break
                neg_img = self.load_img(neg_img_name)
                neg_img = self.img_transform(neg_img)

        return pos_img, pos_lang, neg_img, neg_lang
