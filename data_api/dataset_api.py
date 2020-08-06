import json
import os
import random
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

data_path = 'data_api/data'
img_path = 'data_api/data/images'


class TextureDescriptionData:
    def __init__(self, phrase_split='train', phrase_freq_thresh=10, phid_format='set'):
        self_path = os.path.realpath(__file__)
        self_dir = os.path.dirname(self_path)
        self.data_path = os.path.join(self_dir, 'data')

        with open(os.path.join(self.data_path, 'image_splits.json'), 'r') as f:
            self.img_splits = json.load(f)

        self.phrases = list()
        self.phrase_freq = list()
        if phrase_split == 'all':
            phrase_freq_file = 'phrase_freq.txt'
        elif phrase_split == 'train':
            phrase_freq_file = 'phrase_freq_train.txt'
        else:
            raise NotImplementedError
        with open(os.path.join(self.data_path, phrase_freq_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                phrase, freq = line.split(' : ')
                if int(freq) < phrase_freq_thresh:
                    break
                self.phrases.append(phrase)
                self.phrase_freq.append(int(freq))

        self.phrase_phid_dict = {p: i for i, p in enumerate(self.phrases)}
        self.phid_format = phid_format

        self.img_data_dict = dict()
        with open(os.path.join(self.data_path, 'image_descriptions.json'), 'r') as f:
            data = json.load(f)
        for img_d in data:
            img_d['phrase_ids'] = self.descpritions_to_phids(img_d['descriptions'])
            self.img_data_dict[img_d['image_name']] = img_d

        self.img_phrase_match_matrices = dict()
        print('TextureDescriptionData ready. \n{ph_num} phrases with frequency above {freq}.\n'
              'Image count: train {train}, val {val}, test {test}'
              .format(ph_num=len(self.phrases), freq=phrase_freq_thresh, train=len(self.img_splits['train']),
                      val=len(self.img_splits['val']), test=len(self.img_splits['test'])))

    def get_split_data(self, split=None, img_idx=None, load_img=False):
        if split is None:
            img_names = self.img_splits['train'] + self.img_splits['val'] + self.img_splits['test']
        else:
            assert split in self.img_splits
            img_names = self.img_splits[split]

        if img_idx is None:
            img_name = random.choice(img_names)
        else:
            assert img_idx < len(img_names)
            img_name = img_names[img_idx]

        img_data = self.img_data_dict[img_name]
        if not load_img:
            return img_data

        img_data['image'] = self.load_img(img_name)
        return img_data

    def load_img(self, img_name):
        img_fpath = os.path.join(self.data_path, 'images', img_name)
        img = Image.open(img_fpath).convert('RGB')
        return img

    def phid_to_phrase(self, phid):
        if phid > len(self.phrases) or phid < 0:
            return '<UNK>'
        return self.phrases[phid]

    def phrase_to_phid(self, phrase):
        return self.phrase_phid_dict.get(phrase, -1)

    @staticmethod
    def description_to_phrases(desc):
        segments = re.split('[,;]', desc)
        phrases = list()
        for seg in segments:
            phrase = seg.strip()
            if len(phrase) > 0:
                phrases.append(phrase)
        return phrases

    def descpritions_to_phids(self, descriptions, phid_format=None):
        if phid_format is None:
            phid_format = self.phid_format

        if phid_format is None:
            return None

        phrases = set()
        if phid_format == 'str':
            for desc in descriptions:
                phrases.update(self.description_to_phrases(desc))
            return phrases

        phids = list()
        for desc in descriptions:
            phrases = self.description_to_phrases(desc)
            phids_desc = [self.phrase_to_phid(ph) for ph in phrases]
            phids.append(phids_desc)

        if phid_format == 'nested_list':
            return phids

        elif phid_format == 'phid_freq':
            phid_freq = dict()
            for phids_desc in phids:
                for phid in phids_desc:
                    phid_freq[phid] = phid_freq.get(phid, 0) + 1
            return phid_freq

        elif phid_format == 'set':
            phid_set = set()
            for phids_desc in phids:
                phid_set.update(phids_desc)
            return phid_set

        else:
            raise NotImplementedError

    def description_to_phids_smart(self, desc):
        phids = set()
        phrases = self.description_to_phrases(desc)
        for ph in phrases:
            if ph in self.phrases:
                phids.add(self.phrase_to_phid(ph))
            else:
                for wd in WordEncoder.tokenize(ph):
                    if wd in self.phrases:
                        phids.add(self.phrase_to_phid(wd))
        return phids

    def get_img_phrase_match_matrices(self, split):
        if split in self.img_phrase_match_matrices:
            return self.img_phrase_match_matrices[split]
        img_num = len(self.img_splits[split])
        phrase_num = len(self.phrases)

        match = np.zeros((img_num, phrase_num), dtype=int)
        for img_i, img_name in enumerate(self.img_splits[split]):
            img_data = self.img_data_dict[img_name]
            if self.phid_format == 'set':
                phid_set = img_data['phrase_ids']
            else:
                phid_set = self.descpritions_to_phids(img_data['descriptions'], phid_format='set')
            for phid in phid_set:
                if phid >= 0:
                    match[img_i, phid] = 1
        self.img_phrase_match_matrices[split] = match
        return match

    def get_gt_phrase_count(self, split):
        gt_phrase_count = np.zeros(len(self.img_splits[split]))
        for img_i, img_name in enumerate(self.img_splits[split]):
            phrases = set()
            img_data = self.img_data_dict[img_name]
            for desc in img_data['descriptions']:
                phrases.update(self.description_to_phrases(desc))
            gt_phrase_count[img_i] = len(phrases)
        return gt_phrase_count


class ImgOnlyDataset(Dataset):
    def __init__(self, split, transform=None, texture_dataset=None):
        Dataset.__init__(self)
        if texture_dataset is None:
            texture_dataset = TextureDescriptionData(phid_format=None)
        self.dataset = texture_dataset
        self.split = split
        self.transform = transform

    def __getitem__(self, idx):
        img_data = self.dataset.get_split_data(self.split, img_idx=idx, load_img=True)
        img_name = img_data['image_name']
        img = img_data['image']
        if self.transform is not None:
            img = self.transform(img)
        return img_name, img

    def __len__(self):
        return len(self.dataset.img_splits[self.split])


class PhraseOnlyDataset(Dataset):
    def __init__(self, texture_dataset=None):
        Dataset.__init__(self)
        if texture_dataset is None:
            texture_dataset = TextureDescriptionData(phid_format=None)
        self.dataset = texture_dataset

    def __getitem__(self, idx):
        return self.dataset.phrases[idx]

    def __len__(self):
        return len(self.dataset.phrases)


class WordEncoder:
    def __init__(self, word_freq_file='word_freq_train.txt', word_freq_thresh=5, special_chars=",;/&()-'"):
        self_path = os.path.realpath(__file__)
        self_dir = os.path.dirname(self_path)
        data_path = os.path.join(self_dir, 'data')

        self.word_list = ['<pad>']
        self.word_freq = [0]
        self.word_map = None

        with open(os.path.join(data_path, word_freq_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                word, freq = line.split(' : ')
                if int(freq) < word_freq_thresh:
                    break
                self.word_list.append(word)
                self.word_freq.append(int(freq))

        if special_chars is not None:
            self.word_list += [ch for ch in special_chars]
            self.word_freq += [1] * len(special_chars)
        self.word_list += ['<unk>', '<start>', '<end>']
        self.word_freq += [0, 0, 0]

        self.word_map = {w: idx for idx, w in enumerate(self.word_list)}

    @staticmethod
    def tokenize(lang_input):
        words = re.split('(\W)', lang_input)
        words = [w.strip() for w in words if len(w.strip()) > 0]
        return words

    def detokenize(self, tokens):
        caption = ' '.join(tokens)
        for ch in ',;':
            caption = caption.replace(' ' + ch + ' ', ch + ' ')
        for ch in "/()-'":
            caption = caption.replace(' ' + ch + ' ', ch)
        return caption

    def encode(self, lang_input, max_len=-1):
        tokens = ['<start>'] + self.tokenize(lang_input) + ['<end>']
        encoded = [self.word_map.get(word, self.word_map['<unk>']) for word in tokens]
        if max_len <= 0:
            return encoded, len(encoded)
        if len(encoded) >= max_len:
            return encoded[:max_len], max_len
        l = len(encoded)
        encoded += [self.word_map['<pad>']] * (max_len - len(encoded))
        return encoded, l

    def encode_pad(self, lang_inputs):
        encoded = list()
        lens = list()
        for lang in lang_inputs:
            e, l = self.encode(lang, max_len=-1)
            encoded.append(e)
            lens.append(l)
        max_l = max(lens)
        padded = np.zeros((len(encoded), max_l), dtype=np.long)
        for i in range(len(encoded)):
            padded[i, :lens[i]] = encoded[i]
        return padded, lens
