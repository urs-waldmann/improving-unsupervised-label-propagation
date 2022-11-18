import os

import cv2
import numpy as np
import torch.utils.data as data

import config


class VocSegmentation(data.Dataset):
    """
    Wrapper class to load the Pascal VOC 2012 dataset.
    """

    def __init__(self, load_seg=True, kind='train', seg_kind='class', variant='VOC2012', base_path=None,
                 transform=None):
        assert kind in ['train', 'val', 'trainval']
        assert seg_kind in ['class', 'object']
        assert variant in ['VOC2012']

        if base_path is None:
            base_path = os.path.join(config.dataset_root, variant)

        seg_mapping = {
            'class': 'SegmentationClass',
            'object': 'SegmentationObject',
        }

        self.load_segmentation = load_seg
        self.transform = transform

        self.image_dir = os.path.join(base_path, 'JPEGImages')
        assert os.path.exists(self.image_dir)
        self.seg_dir = os.path.join(base_path, seg_mapping[seg_kind])
        assert os.path.exists(self.seg_dir)

        list_path = os.path.join(base_path, 'ImageSets', 'Segmentation', kind + ".txt")
        with open(list_path, 'r', encoding='utf8') as f:
            self.image_names = f.read().splitlines()

    @staticmethod
    def _load_img(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        assert img is not None

        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        img_name = self.image_names[item]

        img = VocSegmentation._load_img(os.path.join(self.image_dir, img_name + '.jpg'))

        if self.load_segmentation:
            seg = VocSegmentation._load_img(os.path.join(self.seg_dir, img_name + '.png'))
            sample = (img, seg)
        else:
            sample = img

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


if __name__ == '__main__':

    data = VocSegmentation(load_seg=True, )

    for i in range(len(data)):
        img, seg = data[i]
        img = np.transpose(img, (1, 2, 0))
        seg = np.transpose(seg, (1, 2, 0))

        cv2.imshow('img', img)
        cv2.imshow('seg', seg)
        cv2.waitKey()
