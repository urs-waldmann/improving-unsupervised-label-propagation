from abc import ABC, abstractmethod

import torch


class LabelInitializer(ABC):
    """
    Base class for label initialization methods.
    """

    def __init__(self, device):
        self.device = device

    @abstractmethod
    def initialize_labels(self, frame_source) -> torch.Tensor:
        """
        Computes a label function for the first frame of the given frame source.
        :param frame_source: Input video frame source
        :return: Label tensor.
        """
        raise NotImplementedError()


class GroundTruthLabelInitializer(LabelInitializer):
    """
    Label initializer that returns the ground truth label function for the first frame.
    """

    def initialize_labels(self, frame_source):
        labels = frame_source.label_at(0)
        labels = labels.to(device=self.device)

        return labels
