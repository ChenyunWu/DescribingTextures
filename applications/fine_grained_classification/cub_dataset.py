import os
import numpy as np
import torch
from PIL import Image
import torch.utils.data as data

from models.layers.img_encoder import build_transforms


class CUBDataset(data.Dataset):
    def __init__(self, split=None, data_path='other_datasets/CUB_200_2011', val_ratio=0.1):
        self.data_path = data_path
        self.split = split
        self.val_ratio = val_ratio
        if split not in ['train', 'val', 'test']:
            self.split = None
        self.img_transform = build_transforms(is_train=False)

        self.img_splits = {'train': list(), 'test': list()}
        with open(os.path.join(data_path, 'train_test_split.txt'), 'r') as f:
            for line in f:
                img_id, is_train = line.split(' ')
                img_id = int(img_id.strip()) - 1
                is_train = int(is_train.strip())
                if is_train:
                    self.img_splits['train'].append(img_id)
                else:
                    self.img_splits['test'].append(img_id)
        if val_ratio > 0:
            val_len = int(len(self.img_splits['train']) * val_ratio)
            np.random.seed(0)
            self.img_splits['val'] = np.random.choice(self.img_splits['train'], val_len, replace=False)
            train_ids = [i for i in self.img_splits['train'] if i not in self.img_splits['val']]
            self.img_splits['train'] = train_ids

        self.class_names = []
        with open(os.path.join(data_path, 'classes.txt'), 'r') as f:
            for li, line in enumerate(f):
                cls_id, name = line.split(' ')
                cls_id = int(cls_id.strip()) - 1
                assert cls_id == li
                name = name.strip()
                self.class_names.append(name)

        self.att_names = []
        with open(os.path.join(data_path, 'attributes/attributes.txt'), 'r') as f:
            for li, line in enumerate(f):
                att_id, name = line.split(' ')
                att_id = int(att_id.strip()) - 1
                assert att_id == li
                name = name.strip()
                self.att_names.append(name)

        self.att_types = dict()
        self.att_types['shape'] = [i for i, n in enumerate(self.att_names)
                                   if '_shape:' in n or 'length:' in n or 'size:' in n]
        for type in ['color', 'pattern']:
            self.att_types[type] = [i for i, n in enumerate(self.att_names) if '_%s:' % type in n]

        self.img_data_list = []
        with open(os.path.join(data_path, 'images.txt'), 'r') as f:
            for li, line in enumerate(f):
                img_id, name = line.split(' ')
                img_id = int(img_id.strip()) - 1
                assert img_id == li
                name = name.strip()
                self.img_data_list.append({'img_name': name})

        with open(os.path.join(data_path, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                img_id, cls = line.split(' ')
                img_id = int(img_id.strip()) - 1
                cls = int(cls.strip()) - 1
                self.img_data_list[img_id]['class_label'] = cls

        with open(os.path.join(data_path, 'bounding_boxes.txt'), 'r') as f:
            for line in f:
                numbers = line.split(' ')
                numbers = [float(n.strip()) for n in numbers]
                img_id = int(numbers[0]) - 1
                xywh = numbers[1:]
                self.img_data_list[img_id]['box'] = xywh

        for img_data in self.img_data_list:
            img_data['att_labels'] = np.zeros(len(self.att_names))

        self.gt_att_labels = np.zeros((len(self.img_data_list), len(self.att_names)))
        with open(os.path.join(data_path, 'attributes/image_attribute_labels.txt'), 'r') as f:
            # <image_id> <attribute_id> <is_present> <certainty_id> <time>
            for line in f:
                numbers = line.split(' ')
                numbers = [int(n.strip()) for n in numbers[:3]]
                img_id = numbers[0] - 1
                att_id = numbers[1] - 1
                is_present = numbers[2]
                if is_present:
                    self.img_data_list[img_id]['att_labels'][att_id] = 1
                    self.gt_att_labels[img_id, att_id] = 1

        # TODO non-localized

        self.class_att_labels = np.zeros((200, 312))
        with open(os.path.join(data_path, 'attributes/class_attribute_labels_continuous.txt'), 'r') as f:
            # 200 lines and 312 space-separated columns
            for li, line in enumerate(f):
                numbers = line.split(' ')
                numbers = np.array([float(n.strip()) for n in numbers])
                self.class_att_labels[li] = numbers

        print('CUB dataset ready.')
        return

    def load_img(self, img_idx):
        img_fpath = os.path.join(self.data_path, 'images', self.img_data_list[img_idx]['img_name'])
        img = Image.open(img_fpath).convert('RGB')
        return img

    def __getitem__(self, i):
        if self.split is not None:
            img_idx = self.img_splits[self.split][i]
        else:
            img_idx = i
        img_data = self.img_data_list[img_idx]
        img = self.load_img(img_idx)
        img = self.img_transform(img)
        # att_labels = torch.from_numpy(img_data['att_labels']).to(dtype=torch.float)
        return img_idx, img, img_data['class_label'], img_data['att_labels']

    def __len__(self):
        if self.split is not None:
            return len(self.img_splits[self.split])
        else:
            return len(self.img_data_list)


if __name__ == '__main__':
    ds = CUBDataset(split=None)
    for s, vs in ds.img_splits.items():
        print(s, len(vs))
