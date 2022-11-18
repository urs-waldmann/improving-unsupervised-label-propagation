"""
Helper class to access the BADJA dataset.

BIGGS, Benjamin, RODDICK, Thomas, FITZGIBBON, Andrew y CIPOLLA, Roberto. Creatures great and SMAL: Recovering the shape
and motion of animals from video. Springer. 2018.p. 3–19.
"""

import json
import os
from typing import Optional

import cv2 as cv
import numpy as np
from tqdm import tqdm

import config
from dataset.keypoint_dataset import KeypointVideo, KeypointVideoDataset
from utils.keypoint_visualization import draw_skeleton, BADJA_SKELETON
from utils.path_utils import list_files


class BadjaVideo(KeypointVideo):
    def __init__(self, video_name, dataset_root, *, clip_to_first_annotation):
        self.video_name = video_name
        self.gt_path = os.path.join(dataset_root, 'joint_annotations', video_name + ".json")

        self.mask_invisible = True  # TODO: add param

        with open(self.gt_path, 'r', encoding='utf8') as f:
            gt = json.load(f)

        self.seg_dir = os.path.join(dataset_root, os.path.dirname(gt[0]['segmentation_path']))
        self.img_dir = os.path.join(dataset_root, os.path.dirname(gt[0]['image_path']))

        self.frame_paths = sorted(list_files(self.img_dir))
        self.seg_paths = sorted(list_files(self.seg_dir))

        self.keypoints = {}
        self.visibilities = {}

        mask = None

        for f in gt:
            fn = int(os.path.splitext(os.path.basename(f['segmentation_path']))[0])
            visibility = np.array(f['visibility'], dtype=bool)
            points = np.array(f['joints'], dtype=np.float32)

            if self.mask_invisible:
                if mask is None:
                    mask = visibility
                points = points[mask]

            self.visibilities[fn] = visibility
            self.keypoints[fn] = points.T[::-1, ...].copy()

        if clip_to_first_annotation:
            first_annot = sorted(self.keypoints.keys())[0]
            self.visibilities = {k - first_annot: v for k, v in self.visibilities.items()}
            self.keypoints = {k - first_annot: v for k, v in self.keypoints.items()}
            self.frame_paths = self.frame_paths[first_annot:]
            self.seg_paths = self.seg_paths[first_annot:]

    def frame_at(self, index) -> np.ndarray:
        return cv.imread(self.frame_paths[index], cv.IMREAD_COLOR)

    def keypoints_at(self, index) -> Optional[np.ndarray]:
        if index in self.keypoints:
            keypoints = self.keypoints[index]
        else:
            keypoints = None

        return keypoints

    def __iter__(self):
        for fn in range(len(self)):
            frame = self.frame_at(fn)
            keypoints = self.keypoints_at(fn)
            yield dict(
                frame=frame,
                keypoints=keypoints
            )

    def __len__(self):
        return len(self.frame_paths)


class BadjaDataset(KeypointVideoDataset):
    _VIDEO_NAMES = [
        'bear',
        'camel',
        # 'cat_jump',
        'cows',
        'dog-agility',
        'dog',
        'horsejump-high',
        'horsejump-low',
        'impala0',
        'rs_dog',
        # 'tiger'
    ]

    def __init__(self, dataset_root=None, *, clip_to_first_annotation=True):
        self.clip_to_first_annotation = clip_to_first_annotation

        self.dataset_root = dataset_root if dataset_root is not None else config.dataset_badja_path

        self.name_to_param = {}
        self.video_params = []
        for i, video_name in enumerate(sorted(BadjaDataset._VIDEO_NAMES)):
            self.video_params.append((video_name, self.dataset_root))
            self.name_to_param[video_name] = i

    def make_video(self, params) -> BadjaVideo:
        return BadjaVideo(*params, clip_to_first_annotation=self.clip_to_first_annotation)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.make_video(self.video_params[self.name_to_param[item]])
        else:
            return self.make_video(self.video_params[item])

    def __iter__(self):
        for params in self.video_params:
            yield self.make_video(params)

    def __len__(self):
        return len(self.video_params)


if __name__ == '__main__':
    dataset = BadjaDataset()

    for video in dataset:
        for frame_info in tqdm(video, desc=f'Showing video {video.video_name}: '):
            frame = frame_info['frame']
            # cv.imshow('Frame', frame)

            keypoints = frame_info['keypoints']

            keypoint_canvas = frame.copy()
            skeleton_canvas = frame.copy()
            if keypoints is not None:

                draw_skeleton(keypoint_canvas, keypoints, BADJA_SKELETON, line_thickness=2)

                for i in range(keypoints.shape[-1]):
                    pt = (int(keypoints[0, i]), int(keypoints[1, i]))
                    cv.drawMarker(keypoint_canvas, pt, (0, 255, 0))
                    cv.putText(keypoint_canvas, f'{i}', pt, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

            cv.imshow('Keypoints', keypoint_canvas)
            # cv.imshow('Skeleton', skeleton_canvas)

            k = cv.waitKey()
            if k == ord('n'):
                break
            elif k == ord('q'):
                exit()
