import glob
import os
from collections import OrderedDict
from typing import List

import cv2 as cv
import yaml

import config
from dataset.mask_dataset import MaskDataset, SimpleMaskVideo
from utils.mask_utils import read_mask


class SingleObjectVosVideo:
    """
    Video metadata class that allows access to frames and associated masks.
    """

    def __init__(self, video_name: str,
                 image_files: List[str],
                 mask_files: List[str],
                 transform, mask_transform):
        assert len(mask_files) == len(image_files)

        self.video_name = video_name
        self._image_files = image_files
        self._mask_files = mask_files
        self.transform = transform
        self.mask_transform = mask_transform

        self._length = len(self._image_files)

    def frame_at(self, index: int):
        assert 0 <= index < self._length

        path = self._image_files[index]
        frame = cv.imread(path)

        if frame is None:
            raise AssertionError(f'Video file missing! Path: {path}.')

        if self.transform is not None:
            frame = self.transform(frame)

        return frame

    def mask_at(self, index: int):
        assert 0 <= index < self._length

        path = self._mask_files[index]
        mask = read_mask(path)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return mask

    def __iter__(self):
        for i in range(self._length):
            yield self.frame_at(i), self.mask_at(i)

    def __len__(self):
        return self._length


class Davis2016:
    """
    Dataset access helper to work with the DAVIS2016 dataset.
    """
    possible_resolutions = {'1080p', '480p'}
    possible_splits = {'train', 'val', 'trainval'}

    def __init__(self, dataset_dir=None, resolution='480p', split='train', transform=None, mask_transform=None):
        """
        :param dataset_dir: Dataset root directory.
        :param resolution: Resolution of the images and frames to use. Must match the resolution available in the root
                           directory.
        :param split: Train, val or trainval dataset split.
        :param transform: transformation to apply to frames after loading.
        :param mask_transform: transformation to apply to masks after loading.
        """
        if dataset_dir is None:
            dataset_dir = os.path.join(config.dataset_root, 'DAVIS', 'DAVIS2016')
        self.dataset_dir = dataset_dir

        if resolution not in Davis2016.possible_resolutions:
            raise ValueError(f'Invalid resolution {resolution}, must be one of {Davis2016.possible_resolutions}.')
        self.resolution = resolution

        if split not in Davis2016.possible_splits:
            raise ValueError(f'Invalid split {split}, must be one of {Davis2016.possible_splits}.')
        self.split = split

        with open(os.path.join(self.dataset_dir, 'Annotations', 'db_info.yml'), 'r', encoding='utf8') as f:
            annotations = yaml.safe_load(f)

            self._sets = annotations['sets']
            self._attributes = annotations['attributes']
            self._video_meta = {v['name']: v for v in annotations['sequences']}

        image_set_file = os.path.join(self.dataset_dir, 'ImageSets', self.resolution, f'{self.split}.txt')

        with open(image_set_file, 'r', encoding='utf8') as f:
            self._sequences = OrderedDict()
            for line in f:
                im_path, mask_path = line.strip().split(' ')

                key = os.path.basename(os.path.dirname(im_path))

                frames = self._sequences.get(key, [])
                frames.append((im_path, mask_path))
                self._sequences[key] = frames

        self.transform = transform
        self.mask_transform = mask_transform

    def __iter__(self):
        for video_name in self._sequences.keys():
            yield self.get_video(video_name)

    def get_video(self, video_name) -> SingleObjectVosVideo:
        video_meta = self._video_meta[video_name]
        frame_specs = self._sequences[video_name]
        images, masks = map(list, zip(*frame_specs))

        images = [os.path.join(self.dataset_dir, ip[1:]) for ip in images]
        masks = [os.path.join(self.dataset_dir, ip[1:]) for ip in masks]

        assert len(images) == len(masks) == video_meta['num_frames']

        return SingleObjectVosVideo(
            video_name,
            images, masks,
            self.transform,
            self.mask_transform
        )

    def __len__(self):
        return len(self._sequences)


class DavisVideo(SimpleMaskVideo):
    def __init__(self, dataset_path: str, video_name: str, dataset_resolution: str):
        pattern = os.path.join(dataset_path, "JPEGImages", dataset_resolution, video_name, "*.jpg")
        frame_paths = sorted(glob.glob(pattern))
        mask_paths = [p.replace("JPEGImages", "Annotations").replace("jpg", "png") for p in frame_paths]

        super().__init__(video_name, frame_paths, mask_paths)


class MultiObjectDavisDataset(MaskDataset):
    """
    Helper class to access DAVIS2016 and DAVIS2017 datasets.
    """

    year: str
    split: str
    resolution: str
    _dataset_root: str
    _split_files: List[str]

    def __init__(self, dataset_root=None, year: str = '2017', split: str = 'train', resolution: str = '480p'):
        assert year in {"2016", "2017"}
        # Downloading DAVIS2016 separately also includes a trainval split.
        assert split in {'val', 'train', 'trainval'}
        assert resolution == '480p', 'Invalid resolution. At the moment only 480p is supported.'

        self.year = year
        self.split = split
        self.resolution = resolution
        self._dataset_root = dataset_root if dataset_root is not None else config.dataset_davis2017_root
        self._split_files = ['train.txt', 'val.txt'] if self.split == 'trainval' else [f'{self.split}.txt']

        super().__init__()

    def _read_video_names(self) -> List[str]:
        video_names = []
        for load_file in self._split_files:
            video_list_path = os.path.join(self._dataset_root, 'ImageSets', self.year, load_file)
            with open(video_list_path, 'r', encoding='utf8') as f:
                for line in f:
                    video_names.append(line.strip())

        video_names = sorted(video_names)
        return video_names

    def _create_video(self, video_name) -> DavisVideo:
        return DavisVideo(
            dataset_path=self._dataset_root,
            video_name=video_name,
            dataset_resolution=self.resolution
        )


if __name__ == '__main__':
    from utils.vis_utils import draw_semantic_masks, davis_color_map
    import numpy as np

    dataset = MultiObjectDavisDataset(year='2017')
    single_object = False

    color_map = davis_color_map()
    for video in dataset:
        for frame_info in video:
            frame = frame_info['frame']
            mask = frame_info['mask']
            if single_object:
                mask = (mask > 0).astype(np.uint8)
            frame = draw_semantic_masks(frame, mask, color_map, draw_outline=True)
            cv.imshow('Frame', frame)
            cv.waitKey(1)

    # dataset = Davis2016(resolution='480p', split='trainval')
    #
    # for video in dataset:
    #     for frame, mask in video:
    #         frame = draw_instance_mask(frame, mask, (0, 255, 0))
    #         cv.imshow('Frame', frame)
    #         cv.waitKey(1)
