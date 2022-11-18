from abc import ABC, abstractmethod
from os.path import join
from typing import List, NamedTuple, Iterator, Union, Dict, Any, Optional

import cv2 as cv
import numpy as np
from PIL import Image


class Resolution(NamedTuple):
    width: int
    height: int


class MaskVideo(ABC):
    video_name: str

    def __init__(self, video_name: str):
        assert video_name is not None
        self.video_name = video_name

    @abstractmethod
    def get_resolution(self) -> Resolution:
        raise NotImplementedError()

    @abstractmethod
    def frame_at(self, index: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def mask_at(self, index: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self)):
            yield {
                'index': i,
                'frame': self.frame_at(i),
                'mask': self.mask_at(i),
            }


class SimpleMaskVideo(MaskVideo):
    _resolution: Optional[Resolution]
    _frame_paths: List[str]
    _mask_paths: List[str]

    def __init__(self, video_name: str, frame_paths: List[str], mask_paths: List[str]):
        super().__init__(video_name)
        assert len(frame_paths) > 0
        assert len(frame_paths) == len(mask_paths)

        self._resolution = None
        self._frame_paths = frame_paths
        self._mask_paths = mask_paths

    def get_resolution(self) -> Resolution:
        if self._resolution is None:
            with Image.open(self._frame_paths[0]) as f1:
                width, height = f1.size
                self._resolution = Resolution(width=width, height=height)
        return self._resolution

    def frame_at(self, index: int) -> np.ndarray:
        return cv.imread(self._frame_paths[index])

    def mask_at(self, index: int) -> np.ndarray:
        return np.asarray(Image.open(self._mask_paths[index]))

    def __len__(self):
        return len(self._frame_paths)


class MaskDataset(ABC):
    _video_names: List[str]

    def __init__(self):
        self._video_names = self._read_video_names()

    @abstractmethod
    def _read_video_names(self) -> List[str]:
        raise NotImplementedError()

    @abstractmethod
    def _create_video(self, video_name) -> MaskVideo:
        raise NotImplementedError()

    def __iter__(self) -> Iterator[MaskVideo]:
        for video_name in self._video_names:
            yield self._create_video(video_name)

    def __getitem__(self, item) -> Union[MaskVideo, List[MaskVideo]]:
        if isinstance(item, str):
            return self._create_video(item)
        elif isinstance(item, int):
            return self._create_video(self._video_names[item])
        elif isinstance(item, slice):
            return [self._create_video(video_name) for video_name in self._video_names[item]]

    def __len__(self):
        return len(self._video_names)


class SegmentationVideoResult:
    def __init__(self, video_dir):
        self.video_dir = video_dir

    def mask_at(self, fn):
        return np.array(Image.open(join(self.video_dir, f'{fn:05d}.png')))


class SegmentationResultReader:
    def __init__(self, result_dir):
        self.result_dir = result_dir

    def __getitem__(self, video_name):
        assert isinstance(video_name, str)

        return SegmentationVideoResult(join(self.result_dir, video_name))
