import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from models.layers.img_encoder import build_transforms


class SyntheticData(Dataset):
    def __init__(self):
        self.img_names = ['cobwebbed/cobwebbed_0088.jpg',
                          'lined/lined_0084.jpg',
                          'polka-dotted/polka-dotted_0215.jpg',
                          'swirly/swirly_0135.jpg',
                          'chequered/chequered_0103.jpg',
                          'honeycombed/honeycombed_0059.jpg',
                          'dotted/dotted_0131.jpg',
                          'striped/striped_0035.jpg',
                          'zigzagged/zigzagged_0064.jpg',
                          'banded/banded_0138.jpg']

        self.colors = {'white': (245, 245, 230),
                       'black': (20, 20, 20),
                       'brown': (120, 70, 20),
                       'green': (30, 160, 30),
                       'blue': (30, 100, 200),
                       'red': (120, 30, 30),
                       'yellow': (240, 240, 30),
                       'pink': (240, 140, 190),
                       'orange': (240, 140, 20),
                       'gray': (100, 100, 100),
                       'purple': (140, 30, 200)}
        self.color_names = list(self.colors.keys())
        # 'silver': (192, 192, 192)}
        self.is_fore_back = [True, True, True, True, False, False, True, False, False, False]
        self.patterns = ['web', 'lines', 'polka-dots', 'swirls', 'squares', 'hexagon', 'dots', 'stripes', 'zigzagged',
                         'banded']

        self.color_tuples = list()
        for i in range(len(self.color_names)):
            for j in range(len(self.color_names)):
                if i != j:
                    self.color_tuples.append((i, j))

        self.img_transform = build_transforms(is_train=False)

    def is_similar_pattern(self, p1_i, p2_i):
        similar_sets = [['polka-dots', 'dots'],
                        ['lines', 'swirls', 'stripes', 'zigzagged', 'banded']]
        if p1_i == p2_i:
            return True
        p1 = self.patterns[p1_i]
        p2 = self.patterns[p2_i]
        for s in similar_sets:
            if p1 in s and p2 in s:
                return True
        return False

    def get_img_name(self, img_i, c1_i, c2_i):
        img_name = '%s_%s_%s.jpg' \
                   % (self.img_names[img_i].split('.')[0], self.color_names[c1_i], self.color_names[c2_i])
        return img_name

    def get_img(self, img_i, c1_i, c2_i):
        img_path = 'applications/synthetic_imgs/visualizations/modified_imgs/%s' \
                   % (self.get_img_name(img_i, c1_i, c2_i))
        img = Image.open(img_path).convert('RGB')
        return img

    def get_desc(self, img_i, c1_i, c2_i):
        if self.is_fore_back[img_i]:
            f = '{C1} {P}, {C2} background'
        else:
            f = '{C1} and {C2} {P}'
        return f.format(P=self.patterns[img_i], C1=self.color_names[c1_i], C2=self.color_names[c2_i])

    def unravel_index(self, i):
        img_i, ct_i = np.unravel_index(i, (len(self.img_names), len(self.color_tuples)))
        c1_i, c2_i = self.color_tuples[ct_i]
        return img_i, c1_i, c2_i

    def ravel_index(self, img_i, c1_i, c2_i):
        ct_idx = self.color_tuples.index((c1_i, c2_i))
        return np.ravel_multi_index((img_i, ct_idx), (len(self.img_names), len(self.color_tuples)))

    def __getitem__(self, idx):
        img = self.get_img(*(self.unravel_index(idx)))
        return idx, self.img_transform(img)

    def __len__(self):
        return len(self.img_names) * len(self.color_tuples)

    def get_all_imgs(self):
        imgs = list()
        for i in range(len(self)):
            img = self.get_img(*self.unravel_index(i))
            imgs.append(img)
        return imgs
