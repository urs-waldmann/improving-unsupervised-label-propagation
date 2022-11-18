"""
Helper classes to access ILSVRC2015 data preprocessed for training with UDT or LUDT. The helpers don't provide access
to the normal ILSVRC2015 images. Instead, use the torchvision.dataset classes to access ImageNet directly.
"""

import json
import os
import random
from os.path import join
from typing import List, Optional, Any, Callable

import cv2 as cv
import numpy as np
import torch
import torch.utils.data as data

import config

imagenet_mean = np.array([109, 120, 119], dtype=np.float32).reshape([3, 1, 1])


def udt_imagenet_preproc(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing of ImageNet data for UDT.
    TODO: Frames are not converted to RGB and range 0-1 or -0.5, 0.5. Is this an error or is there a reason?

    :param img: input image
    :return: transformed image
    """
    # Bring color axis to front. Shape is c, h, w
    return np.transpose(img, (2, 0, 1)).astype(np.float32) - imagenet_mean


class LudtIlsvrcDataset(data.Dataset):
    """
    Loader class for the imagenet dataset preprocessed for LUDT.
    """

    def __init__(self, dataset_dir=None,
                 manifest_file=None,
                 num_patches=3,
                 seq_len=10,
                 transform: Optional[Callable[[np.ndarray], Any]] = udt_imagenet_preproc,
                 variant='train'):
        """
        :param dataset_dir: Directory where the data is stored.
        :param manifest_file: Path to the dataset manifest file.
        :param num_patches: Number of image patches to sample for each video.
        :param seq_len: Max distance between the patch samples.
        :param transform: transformations to apply to the patches.
        :param variant: train, val or test variant of the dataset.
        """

        if dataset_dir is None:
            dataset_dir = config.dataset_ludt_imagenet_path

        if manifest_file is None:
            manifest_file = os.path.join(dataset_dir, 'manifest.json')

        assert seq_len > 0, 'Sequence length must be positive.'
        assert num_patches > 0, 'Number of patches must be at least 1.'
        assert seq_len >= num_patches, 'Sequence must be at least as long as num_patches'
        assert os.path.isdir(dataset_dir), 'Dataset dir does not exist.'
        assert os.path.isfile(manifest_file), 'Manifest file does not exist.'
        assert variant.lower() in ['train', 'test', 'val']

        self.num_patches = num_patches
        self.seq_len = seq_len
        self.transform = transform
        self.dataset_root = dataset_dir
        self.variant = variant.lower()

        with open(manifest_file, 'r', encoding='utf8') as in_file:
            self.manifest = json.load(in_file)

        self.positions = []
        for i, video in enumerate(self.manifest):

            # only select videos from the given variant
            if not video['variant'].startswith(self.variant):
                continue

            n_frames = len(video['frames'])
            for p in range(max(0, n_frames - seq_len + 1)):
                self.positions.append((i, p))

    def __getitem__(self, item: int) -> List:
        video_pos, frame_pos = self.positions[item]
        video = self.manifest[video_pos]

        frames = video['frames'][frame_pos:frame_pos + self.seq_len]
        sample_indices = torch.multinomial(torch.ones(len(frames)), self.num_patches)
        sampled_frames = [frames[i] for i in sample_indices]
        # sampled_frames = random.sample(frames, self.num_patches)
        sampled_frames.sort()

        video_path = os.path.join(self.dataset_root, video['variant'], video['video'])

        patches = []
        for frame_name in sampled_frames:
            frame_path = os.path.join(video_path, frame_name)
            frame = cv.imread(frame_path, cv.IMREAD_COLOR)

            assert frame is not None

            if self.transform is not None:
                frame = self.transform(frame)

            patches.append(frame)

        return patches

    def __len__(self):
        return len(self.positions)


class UdtIlsvrcDataset(data.Dataset):
    """
    Loader class for the imagenet dataset preprocessed according to UDT.
    """

    _variant_map = {
        'train': 'train_set',
        'val': 'val_set'
    }

    def __init__(self, dataset_dir=None,
                 manifest_file=None,
                 image_dir=None,
                 seq_len=10,
                 variant='train',
                 transform: Optional[Callable[[np.ndarray], Any]] = udt_imagenet_preproc):
        """
        :param dataset_dir: Directory where the dat is stored.
        :param manifest_file: Path to the dataset manifest file.
        :param image_dir: Directory where the images are stored.
        :param seq_len: Number of frames to select the template and search patches from.
        :param variant: Train or val variant of the dataset.
        :param transform: Transformations to apply to the loaded patches.
        """

        if dataset_dir is None:
            dataset_dir = config.dataset_udt_imagenet_path
        if manifest_file is None:
            manifest_file = join(dataset_dir, 'dataset.json')
        if image_dir is None:
            image_dir = join(dataset_dir, 'crop_125_2.0')

        if not os.path.isfile(manifest_file):
            raise ValueError(f'Manifest file missing. Expected file at location {manifest_file}.')

        if not os.path.isdir(image_dir):
            raise ValueError(f'Image dir is missing. Expected dir at location {image_dir}')

        if variant not in UdtIlsvrcDataset._variant_map:
            raise ValueError(
                f'Invalid dataset variant {variant}. Must be one of {list(UdtIlsvrcDataset._variant_map.keys())}')

        self.variant_key = UdtIlsvrcDataset._variant_map[variant]
        self.image_root = image_dir
        self.seq_len = seq_len
        self.transform = transform

        with open(manifest_file, 'r') as f:
            self.imdb = json.load(f)

    def _load_frame(self, index: int):
        img = cv.imread(join(self.image_root, f'{index:08d}.jpg'))
        assert img is not None

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __getitem__(self, item):
        """
        output shapes:
        3 x h x w
        """
        target_id = self.imdb[self.variant_key][item]

        range_up = self.imdb['up_index'][target_id]
        indices = target_id + torch.randint(low=1, high=min(range_up, self.seq_len + 1), size=(2,))
        search_id1 = indices[0].item()
        search_id2 = indices[1].item()

        target = self._load_frame(target_id)
        search1 = self._load_frame(search_id1)
        search2 = self._load_frame(search_id2)

        return target, search1, search2

    def __len__(self):
        return len(self.imdb[self.variant_key])


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # data = ILSVRC2015(train=True, transform=None)
    data = LudtIlsvrcDataset(variant='train', transform=None)  # , num_patches=10)
    # data = UdtIlsvrcDataset(variant='train', transform=None)

    print(f'Dataset length: {len(data)}')

    fig = plt.figure(1)
    ax = fig.add_axes([0, 0, 1, 1])

    for i in range(len(data)):
        samples = data[random.randint(0, len(data))]
        # z, x = np.transpose(z, (1, 2, 0)).astype(np.uint8), np.transpose(x, (1, 2, 0)).astype(np.uint8)
        zx = np.concatenate(samples, axis=1)

        ax.imshow(cv.cvtColor(zx, cv.COLOR_BGR2RGB))
        for k in range(len(samples)):
            p = patches.Rectangle(
                (125 / 3 + k * 125, 125 / 3), 125 / 3, 125 / 3, fill=False, clip_on=False, linewidth=2, edgecolor='g')
            ax.add_patch(p)

        # plt.pause(0.5)
        plt.waitforbuttonpress()
