import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from data_api.dataset_api import TextureDescriptionData, WordEncoder


class CaptionDataset(Dataset, TextureDescriptionData):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, split, transform=None, word_encoder=None, is_train=True, caption_max_len=35):
        """
        :param split: split, one of 'train', 'val', 'test'
        :param transform: image transform pipeline
        """
        TextureDescriptionData.__init__(self, phid_format=None)
        self.transform = transform
        self.is_train = is_train
        self.caption_max_len = caption_max_len
        self.split = split
        assert self.split in ('train', 'val', 'test')

        self.word_encoder = word_encoder
        if self.word_encoder is None:
            self.word_encoder = WordEncoder()

        self.img_desc_ids = list()
        for img_i, img_name in enumerate(self.img_splits[split]):
            desc_num = len(self.img_data_dict[img_name]['descriptions'])
            self.img_desc_ids += [(img_i, desc_i) for desc_i in range(desc_num)]

    def __getitem__(self, i):
        img_i, desc_i = self.img_desc_ids[i]
        img_data = self.get_split_data(split=self.split, img_idx=img_i, load_img=True)
        img = img_data['image']
        if self.transform is not None:
            img = self.transform(img)

        desc = img_data['descriptions'][desc_i]
        caption, caplen = self.word_encoder.encode(lang_input=desc, max_len=self.caption_max_len)
        caplen = torch.as_tensor([caplen], dtype=torch.long)
        caption = torch.as_tensor(caption, dtype=torch.long)

        if self.is_train:
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = list()
            mlen = 0
            for desc in img_data['descriptions']:
                c, cl = self.word_encoder.encode(lang_input=desc, max_len=self.caption_max_len)
                all_captions.append(c)
                mlen = max(mlen, cl)

            all_captions_np = np.zeros((len(all_captions), mlen))
            for ci, c in enumerate(all_captions):
                cl = min(len(c), mlen)
                all_captions_np[ci, :cl] = c[:cl]
            all_captions = torch.as_tensor(all_captions_np, dtype=torch.long)
            return img, caption, caplen, all_captions

    def __len__(self):
        return len(self.img_desc_ids)


def caption_collate(batch):
    if len(batch[0]) == 3:  # is_train=True
        return default_collate(batch)
    elif len(batch[0]) == 4:  # is_train=False
        collated = list()
        for item_i, item_list in enumerate(zip(*batch)):
            if item_i < 3:
                collated.append(default_collate(item_list))
            else:
                collated.append(item_list)
        return collated
    else:
        raise NotImplementedError
