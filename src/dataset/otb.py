import json
import os
from typing import Iterable, List

import cv2
import numpy as np

from utils import bbox
import config
from utils.bbox import BBox

OTB2013 = {"Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace", "BlurOwl", "Bolt", "Box", "Car1",
           "Car4", "CarDark", "CarScale", "ClifBar", "Couple", "Crowds", "David", "Deer", "Diving", "DragonBaby",
           "Dudek", "Football", "Freeman4", "Girl", "Human3", "Human4", "Human6", "Human9", "Ironman", "Jump",
           "Jumping", "Liquor", "Matrix", "MotorRolling", "Panda", "RedTeam", "Shaking", "Singer2", "Skating1",
           "Skating2_1", "Skating2_2", "Skiing", "Soccer", "Surfer", "Sylvester", "Tiger2", "Trellis", "Walking",
           "Walking2", "Woman"}

OTB2015 = {"Basketball", "Biker", "Bird1", "BlurBody", "BlurCar2", "BlurFace", "BlurOwl", "Bolt", "Box", "Car1", "Car4",
           "CarDark", "CarScale", "ClifBar", "Couple", "Crowds", "David", "Deer", "Diving", "DragonBaby", "Dudek",
           "Football", "Freeman4", "Girl", "Human3", "Human4", "Human6", "Human9", "Ironman", "Jump", "Jumping",
           "Liquor", "Matrix", "MotorRolling", "Panda", "RedTeam", "Shaking", "Singer2", "Skating1", "Skating2_1",
           "Skating2_2", "Skiing", "Soccer", "Surfer", "Sylvester", "Tiger2", "Trellis", "Walking", "Walking2", "Woman",
           "Bird2", "BlurCar1", "BlurCar3", "BlurCar4", "Board", "Bolt2", "Boy", "Car2", "Car24", "Coke", "Coupon",
           "Crossing", "Dancer", "Dancer2", "David2", "David3", "Dog", "Dog1", "Doll", "FaceOcc1", "FaceOcc2", "Fish",
           "FleetFace", "Football1", "Freeman1", "Freeman3", "Girl2", "Gym", "Human2", "Human5", "Human7", "Human8",
           "Jogging_1", "Jogging_2", "KiteSurf", "Lemming", "Man", "Mhyang", "MountainBike", "Rubik", "Singer1",
           "Skater", "Skater2", "Subway", "Suv", "Tiger1", "Toy", "Trans", "Twinnings", "Vase"}

OTB_sequences = {
    'OTB2013': OTB2013,
    'OTB2015': OTB2015
}


class TrackingVideo:
    """
    Base class that encapsulates tracking video loading.
    """

    def __init__(self,
                 video_name: str,
                 image_files: List[str],
                 init_rect: BBox,
                 gt_rects: List[BBox],
                 transform
                 ):
        self.video_name: str = video_name
        self.init_rect: BBox = init_rect
        self._image_files = image_files
        self._gt_rects = gt_rects
        self._length = len(self._image_files)
        self.transform = transform

    def gt_boxes(self) -> Iterable[BBox]:
        """
        Iterates the ground-truth bounding boxes.
        """
        return self._gt_rects

    def frame_at(self, index: int):
        """
        Loads the frame at the given location and applies transformations if set.
        :param index: frame number of the frame to load.
        :return: output of the transformations or the raw image array.
        """
        assert 0 <= index < self._length

        path = self._image_files[index]
        frame = cv2.imread(path)

        if frame is None:
            raise AssertionError(f'Video file missing! Path: {path}.')

        if self.transform is not None:
            frame = self.transform(frame)

        return frame

    def __iter__(self):
        """
        Simultaneously iterates frames and ground-truth bounding boxes.
        """
        for i in range(self._length):
            yield self.frame_at(i), self._gt_rects[i]

    def __len__(self) -> int:
        """
        Number of frames in the video.
        """
        return self._length


class OtbVideo(TrackingVideo):
    """
    Extension of the simple TrackingVideo with a function to check if the video is part of a specific OTB dataset.
    """

    def __init__(self, key, video_name, image_files, init_rect, gt_rects, transform):
        super().__init__(video_name, image_files, init_rect, gt_rects, transform)
        self.key = key

    def contained_in(self, dataset_name: str):
        """
        Returns True, if the video is contained in the given dataset. Raises ValueError if the dataset name is invalid.
        """
        return self.key in OTB_sequences[dataset_name]


class OtbDataset:
    """
    Dataset accessor class for the OTB datasets.
    """

    def __init__(self, variant='OTB2015', dataset_path=None, transform=None):

        if dataset_path is None:
            dataset_path = config.dataset_otb_path
        self.dataset_path = dataset_path

        with open(os.path.join(dataset_path, f'{variant}_manifest.json'), 'r', encoding='utf8') as f:
            self.manifest = json.load(f)

        self.transform = transform

    def __iter__(self):
        for video_key, video_props in self.manifest.items():
            video_name = video_key
            vid_len = video_props['length']
            pattern = video_props['file_pattern']
            start = video_props['start']
            stop = start + vid_len
            frame_dir = video_props['framedir']
            image_files = [os.path.join(self.dataset_path, frame_dir, pattern.format(i)) for i in range(start, stop)]

            gt_rects = np.load(os.path.join(self.dataset_path, video_props['gt']))
            gt_rects[:, :2] -= 1  # conversion from matlab to python indexing
            gt_rects = bbox.boxes_from_xywh(gt_rects)

            init_rect = gt_rects[0]

            assert vid_len == len(gt_rects) and vid_len == len(image_files)

            yield OtbVideo(video_key, video_name, image_files, init_rect, gt_rects, self.transform)

    def __len__(self):
        return len(self.manifest)


if __name__ == '__main__':
    from tqdm import tqdm

    dataset = OtbDataset()

    for video in dataset:
        for frame, box in tqdm(video, desc=video.video_name, total=len(video)):

            cv2.rectangle(frame, box.ip1(), box.ip2(), (0, 255, 0))
            cv2.imshow('OTB Frame', frame)
            k = cv2.waitKey(1)

            if k == ord('q'):
                exit()
            elif k == ord('n'):
                break
