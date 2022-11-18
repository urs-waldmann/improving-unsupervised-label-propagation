from abc import ABC, abstractmethod

from torch import nn as nn


class AbstractLabelPropagation(nn.Module, ABC):
    @abstractmethod
    def forward(self, feat_tar, feat_refs, label_refs):
        """
        :param feat_tar: [h*w, num_feats]
        :param feat_refs: [num_ctx, num_feats, h*w]
        :param label_refs: [num_ctx, num_labels, h, w]
        :return: [1, num_labels, h, w]
        """
        raise NotImplementedError()

    def reset_for_next_video(self):
        """
        Some label-propagation methods might memorize information about previous frames. Therefore, it is necessary to
        reset this information whenever a video boundary is reached.
        """
        pass
