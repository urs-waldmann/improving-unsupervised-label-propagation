from abc import ABC, abstractmethod
from typing import Sized, Iterable

import numpy as np


class KeypointVideo(ABC, Sized):
    """
    Base class for a keypoint video.
    """

    @abstractmethod
    def frame_at(self, i) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def keypoints_at(self, i) -> np.ndarray:
        """
        :param i:
        :return: shape:  [2, num_keypoints]
        """
        raise NotImplementedError()


class KeypointVideoDataset(ABC, Sized, Iterable[KeypointVideo]):
    """
    Base class for Keypoint datasets.
    """
    pass
