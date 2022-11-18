"""
Wrapper class to load the BSD500 dataset.

@Article{amfm_pami2011,
 author = {Arbelaez, Pablo and Maire, Michael and Fowlkes, Charless and Malik, Jitendra},
 title = {Contour Detection and Hierarchical Image Segmentation},
 journal = {IEEE Trans. Pattern Anal. Mach. Intell.},
 volume = {33},
 number = {5},
 year = {2011},
 pages = {898--916},
 doi = {10.1109/TPAMI.2010.161},
}
"""

import os

import cv2
import numpy as np
import torch.utils.data as data

import config


class BSDS500(data.Dataset):
    """
    Access helper for the BSD500 dataset.
    """

    def __init__(self, kind='train', base_path=None, transform=None):
        assert kind in ['train', 'val', 'test']

        if base_path is None:
            base_path = os.path.join(config.dataset_root, 'BSD', 'BSDS500')

        self.transform = transform

        self.image_dir = os.path.join(base_path, 'data', 'images', kind)
        assert os.path.exists(self.image_dir)

        self.image_numbers = list(
            sorted(int(os.path.splitext(fn)[0]) for fn in os.listdir(self.image_dir) if fn.endswith('.jpg')))

    @staticmethod
    def _load_img(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img is not None

        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        img_num = self.image_numbers[item]
        return self.load_for_image_num(img_num)

    def load_for_image_num(self, img_num):
        img_path = os.path.join(self.image_dir, f'{img_num}.jpg')

        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            s = f'Failed to load image from path: {img_path}'
            raise AssertionError(s)

        sample = np.transpose(img, (2, 0, 1))

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_numbers)


if __name__ == '__main__':

    data = BSDS500()

    for i in range(len(data)):
        img = data[i]
        img = np.transpose(img, (1, 2, 0))

        cv2.imshow('img', img)
        cv2.waitKey()
