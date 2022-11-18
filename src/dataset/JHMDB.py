from __future__ import annotations

import os
from enum import Enum
from typing import List, Dict, Iterator, Any

import cv2 as cv
import numpy as np
from scipy.io import loadmat

import config
from dataset.keypoint_dataset import KeypointVideoDataset, KeypointVideo
from utils import path_utils, vis_utils

# _ACTION_CLASSES = (
#     'swing_baseball', 'throw', 'walk', 'wave', 'brush_hair', 'catch', 'clap', 'climb_stairs', 'golf', 'jump',
#     'kick_ball', 'pick', 'pour', 'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow', 'shoot_gun', 'sit',
#     'stand')
_ACTION_CLASSES = (
    'swing_baseball', 'pour', 'pullup', 'stand', 'golf', 'jump', 'kick_ball', 'pick', 'sit', 'walk', 'shoot_ball',
    'catch', 'wave', 'throw', 'shoot_bow', 'climb_stairs', 'clap', 'brush_hair', 'run', 'shoot_gun', 'push',
)


class ViewPoint(Enum):
    E = 0.0
    ENE = 22.5
    NE = 45.0
    NNE = 67.5
    N = 90.0
    NNW = 112.5
    NW = 135.0
    WNW = 157.5
    W = 180.0
    WSW = 202.5
    SW = 225.0
    SSW = 247.5
    S = 270.0
    SSE = 292.5
    SE = 315.0
    ESE = 337.5


class PuppetMask(Enum):
    BACKGROUND = 0
    FOREGROUND = 1


class Keypoint(Enum):
    NECK = 1
    BELLY = 2
    FACE = 3
    RIGHT_SHOULDER = 4
    LEFT_SHOULDER = 5
    RIGHT_HIP = 6
    LEFT_HIP = 7
    RIGHT_ELBOW = 8
    LEFT_ELBOW = 9
    RIGHT_KNEE = 10
    LEFT_KNEE = 11
    RIGHT_WRIST = 12
    LEFT_WRIST = 13
    RIGHT_ANKLE = 14
    LEFT_ANKLE = 15

    @staticmethod
    def from_value(value) -> Keypoint:
        for member in Keypoint:
            if member.value == value:
                return member
        raise ValueError(f'Unknown value "{value}" for Keypoint.')


def compute_world_keypoints(pos_img, w, h, scale):
    pos_world = np.zeros_like(pos_img)
    pos_world[1, :, :] = (pos_img[1, :, :] / w - 0.5) * w / h / scale
    pos_world[2, :, :] = (pos_img[2, :, :] / h - 0.5) / scale
    return pos_world


class JhmdbVideo(KeypointVideo):
    # General video information
    video_name: str
    action_class: str
    _num_frames: int

    # Frame data
    _frame_paths: List[str]  # List of paths to frames. Length: N

    # Keypoint data
    _viewpoint: str  # one of E ENE ESE N NE NNE NNW NW S SE SSE SSW SW W WNW WSW
    _scale: np.ndarray  # Scale of the person in the corresponding frame. Length: N
    _frame_keypoints: np.ndarray  # Shape: [2,15,N]
    _world_keypoints: np.ndarray  # Shape: [2,15,N]

    # Mask data
    _puppet_masks: np.ndarray  # Shape: [W,H,N]

    def __init__(self, dataset_root: str, action: str, name: str):
        self.video_name = name
        self.action_class = action
        self._frame_dir = os.path.join(dataset_root, 'Rename_Images', action, name)
        self._keypoint_path = os.path.join(dataset_root, 'joint_positions', action, name, 'joint_positions.mat')
        self._mask_path = os.path.join(dataset_root, 'puppet_mask', action, name, 'puppet_mask.mat')

        self._frame_paths = sorted(path_utils.list_files(self._frame_dir, filter_pattern=r'\d{5,5}\.png'))
        self._num_frames = len(self._frame_paths)

        self._load_keypoints_from_mat()
        self._load_masks()

        if self._num_frames != self._scale.shape[0]:
            self._num_frames = self._scale.shape[0]

        assert self._scale.shape == (self._num_frames,)
        assert self._frame_keypoints.shape == (2, 15, self._num_frames)
        assert self._world_keypoints.shape == (2, 15, self._num_frames)
        assert self._masks.shape[-1] == self._num_frames

    def _load_keypoints_from_mat(self):
        keypoint_mat = loadmat(self._keypoint_path)

        self._viewpoint = keypoint_mat['viewpoint'][0]
        self._scale = keypoint_mat['scale'].astype(np.float32)[0, :]
        self._frame_keypoints = keypoint_mat['pos_img'].astype(np.float32) - 1  # The points use matlab coords
        self._world_keypoints = keypoint_mat['pos_world'].astype(np.float32)

    def _load_masks(self):
        mask_mat = loadmat(self._mask_path)
        self._masks = mask_mat['part_mask'] * 255

    def frame_at(self, index: int) -> np.ndarray:
        frame = cv.imread(self._frame_paths[index], cv.IMREAD_COLOR)
        assert frame is not None
        return frame

    def mask_at(self, index: int) -> np.ndarray:
        return self._masks[:, :, index]

    def keypoints_at(self, index: int) -> np.ndarray:
        """
        :return: Keypoint array of shape [2, 15] with keypoint[0,...] x coords and keypoints[1,...] y coords
        """
        return self._frame_keypoints[:, :, index]

    def get_all_keypoints(self) -> np.ndarray:
        return self._frame_keypoints.copy()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(self._num_frames):
            yield {
                'frame': self.frame_at(i),
                'mask': self.mask_at(i),
                'keypoints': self.keypoints_at(i)
            }

    def __len__(self):
        return self._num_frames


class Jhmdb(KeypointVideoDataset):
    """
    puppet_mask/[class_name]/[video_name]/puppet_mask.mat
    joint_positions/[class_name]/[video_name]/joint_positions.mat

    splits/[class_name]_test_split[ind_split].txt

    """

    _valid_splits = {1, 2, 3}
    _split_mapping = {'train': '1', 'test': '2'}
    _inverse_split_mapping = {v: k for k, v in _split_mapping.items()}

    _ds_root: str  # Root path of the dataset
    _video_names: Dict[str, List[str]]  # Mapping from action classes to video names in the corresponding class
    _videos: List[JhmdbVideo]

    def __init__(self, ds_root=None, split_num=1, split_type='train', per_class_limit=-1):
        if ds_root is None:
            ds_root = config.dataset_jhmdb_path
        self._ds_root = ds_root

        assert split_num in self._valid_splits
        assert split_type in self._split_mapping.keys()

        self._split_num = split_num
        self._split_type = self._split_mapping[split_type]

        self._video_names = {}
        self._videos = []

        for action_class in _ACTION_CLASSES:
            action_path = os.path.join(self._ds_root, 'splits', f'{action_class}_test_split{self._split_num}.txt')

            video_names = []
            with open(action_path, 'r', encoding='utf8') as f:
                for line in f:
                    vid_name, split_set = line.split(' ')

                    if split_set.strip() == self._split_type:
                        vid_name_no_ext = os.path.splitext(vid_name)[0]  # remove .avi
                        video_names.append(vid_name_no_ext)

            if per_class_limit is not None and per_class_limit > 0:
                video_names = video_names[:per_class_limit]

            self._video_names[action_class] = video_names

            for video_name in video_names:
                self._videos.append(JhmdbVideo(self._ds_root, action_class, video_name))

    @property
    def split_type(self):
        return self._inverse_split_mapping[self._split_type]

    def video_names(self) -> Iterator[str]:
        for video in self._videos:
            yield video.video_name

    def __iter__(self) -> Iterator[JhmdbVideo]:
        for video in self._videos:
            yield video

    def __getitem__(self, item):
        # assert 0 <= item < len(self._videos)
        return self._videos[item]

    def __len__(self):
        return len(self._videos)


if __name__ == '__main__':
    dataset = Jhmdb()
    for video in dataset:
        for frame_info in video:
            canvas = frame_info['frame'].copy()

            canvas = vis_utils.draw_instance_mask(canvas, frame_info['mask'], (0, 255, 0))

            keypoints = frame_info['keypoints']
            for kp in range(15):
                pos = keypoints[:, kp]
                cv.drawMarker(canvas, (int(pos[0]), int(pos[1])), (0, 255, 0))

            cv.imshow('Frame', canvas)
            cv.imshow('Mask', frame_info['mask'])
            cv.waitKey()
