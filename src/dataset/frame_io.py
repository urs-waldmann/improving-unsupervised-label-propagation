"""
Helper classes to load a sequence of images. The image source can be a camera, directory of images or video file.
"""

import os
from abc import ABC, abstractmethod
from typing import List

import cv2


class FrameSource(ABC):
    """
    Basic frame source that can load frames one by one in a sequence.
    """

    def __init__(self, transform=None):
        self._frame_num = 0
        self.transform = transform

    def next_frame_num(self) -> int:
        num = self._frame_num
        self._frame_num += 1
        return num

    @abstractmethod
    def load_frame(self):
        raise NotImplementedError()

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.load_frame()

        if frame is None:
            raise StopIteration()

        if self.transform is not None:
            frame = self.transform(frame)

        return frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class SizedFrameSource(FrameSource):
    """
    Base class of a frame source that has a predetermined length (number of frames).
    """

    @abstractmethod
    def __len__(self):
        raise NotImplementedError


class CameraFrameSource(FrameSource):
    """
    Frame source that uses one of the connected cameras as a frame source.
    """

    def __init__(self, cam_id=0, size=None, **kwargs):
        super().__init__(**kwargs)
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)

        if size is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

        if not self.cap.isOpened():
            raise RuntimeError('Could not locate camera with id {:d}'.format(self.cam_id))

    def load_frame(self):
        is_success, frame = self.cap.read()

        if not is_success:
            return None

        return frame


class ImageFrameSource(SizedFrameSource):
    """
    Frame source that loads frames one by one from a directory.
    """
    paths: List[str]

    def __init__(self, base_dir, extensions=None, file_list=None, **kwargs):
        super().__init__(**kwargs)

        self._frame_num = 0

        if base_dir is not None:
            base_dir = str(base_dir)

        if extensions is None:
            extensions = ['.jpg', '.jpeg']

        if file_list is not None:
            # ensure all files exist
            for f in file_list:
                f = str(f)

                if base_dir is not None:
                    f = os.path.join(base_dir, f)

                if not os.path.exists(f):
                    raise RuntimeError(f'File does not exist. "{f}"')

                self.paths.append(f)
        else:
            if base_dir is None:
                raise RuntimeError(f'No file list and no base dir were specified, at least one is required.')

            file_list = []
            for f in os.listdir(base_dir):
                if os.path.splitext(f)[-1].lower() in extensions:
                    file_list.append(os.path.join(base_dir, f))

        self.paths = file_list

    def load_frame(self):
        if self._frame_num >= len(self):
            return None
        path = str(self.paths[self._frame_num])
        frame = cv2.imread(path)
        self._frame_num += 1
        return frame

    def __len__(self):
        return len(self.paths)


class VideoFrameSource(SizedFrameSource):
    """
    Frame source that reads frames from a video file.
    """

    def __init__(self, file, **kwargs):
        super().__init__(**kwargs)
        self.file = str(file)

        self.cap = cv2.VideoCapture(self.file)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video file. {:s}".format(self.file))
        self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def load_frame(self):
        is_success, frame = self.cap.read()
        if not is_success:
            return None
        return frame

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __len__(self):
        return self._frame_count


def make_frame_source(path, **kwargs) -> FrameSource:
    if os.path.isdir(path):
        return ImageFrameSource(path, **kwargs)
    elif os.path.isfile(path):
        return VideoFrameSource(path, **kwargs)
    elif path is None:
        return CameraFrameSource(**kwargs)
    else:
        raise ValueError('Could not determine type of frame source.')
