"""
The VideoIO class family forms an abstraction layer that translates between problem domain and label domains.

TODO: These classes need a major refactoring to remove the close coupling of video data source and translation layers.
  Further, it would be better to implement the result saving in a more flexible way for example with one general-purpose
  LabelPropVideoIO class that contains a list of result savers. The actual translation between the problem domains is
  performed in the corresponding LabelCodec anyway, so it would be beneficial to remove this domain-specific code, too.
  *
  This would also allow us to implement the tracking label propagation without the strong coupling to unrelated code.
"""
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union, Callable, Tuple

import numpy as np
import torch

from features import FeatureExtractor

from label_prop.label_codec import AbstractLabelCodec
from utils.vis_utils import ColorMap

LabelCodecFactory = Callable[[Tuple[int, int], Tuple[int, int]], AbstractLabelCodec]


class LabelPropVideoIO(ABC):
    """
    Abstraction layer between problem domain and label domain. Implementations use a LabelCodec instance to perform the
    translation between label functions and problem-specific representation. The implementations wrap a data source,
    mostly a video with ground truth annotations, and provides accessors for the encoded ground truth labels. Further,
    it provides a way to save result labels, which are decoded prior to saving.

    The class acts as a context manager. This allows implementations to defer the persisting of data to the end of the
    video which improves the performance.
    """

    def __init__(self, label_codec: AbstractLabelCodec, only_first_mask):
        """
        :param label_codec: AbstractLabelCodec instance used for translation between problem and label domain.
        :param only_first_mask: True to return only the first mask during iteration. Useful for inference where we are
                                only interested in the initial mask.
        """
        assert label_codec is not None

        self.label_codec = label_codec
        self.only_first_mask = only_first_mask

    def __iter__(self) -> Iterator[Union[np.ndarray, Optional[np.ndarray]]]:
        for i in range(len(self)):
            frame = self.frame_at(i)
            mask = self.label_at(i) if i < 1 or not self.only_first_mask else None

            yield frame, mask

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def frame_at(self, i: int) -> np.ndarray:
        """
        :return: Image of shape [3, h, w] in BGR format.
        """
        raise NotImplementedError()

    def label_at(self, i: int) -> torch.Tensor:
        """
        Loads the labels for the given frame number.

        :param i: Frame number.
        :return: Labels as returned by the label_codec encoding function.
        """
        annot = self.annotation_at(i)

        return self.label_codec.encode(annot)

    @abstractmethod
    def annotation_at(self, i: int):
        """
        Implementations should return the annotations in the problem domain for the given frame number.

        :param i: Frame number.
        :return: Annotations instance.
        """
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def save_frame_result(self, index: int, frame: np.ndarray, labels: torch.Tensor) -> None:
        raise NotImplementedError()


class DatasetIterator(ABC):
    """
    The DatasetIterator class wraps a video dataset and yields video name and mask codec factory tuples for each video.
    There is no longer a real use for this abstraction layer and it only exists for legacy reasons. It would be better
    to get rid it entirely to simplify the code paths.
    """

    def __init__(self,
                 feat_extractor: FeatureExtractor,
                 color_map: ColorMap,
                 label_codec_factory: LabelCodecFactory):
        assert feat_extractor is not None
        assert color_map is not None
        assert label_codec_factory is not None

        self.feat_extractor = feat_extractor
        self.color_map = color_map
        self.label_codec_factory = label_codec_factory

    @abstractmethod
    def iter_videos(self) \
            -> Iterator[Tuple[str, Callable[[str], LabelPropVideoIO]]]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()
