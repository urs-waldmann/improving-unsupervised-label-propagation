import abc

import torch
from torch import nn
from torch.nn import functional as F


class AffinityNorm(nn.Module, abc.ABC):
    """
    Base class for affinity norms.
    """

    @abc.abstractmethod
    def forward(self, affinities, *, dim):
        """
        Implementations should maintain the input shape and normalize over the given dimension.

        :param affinities: Affinity matrix to normalize.
        :param dim: Dimension to normalize.
        :return: Normalized affinities of the same shape as the input affinities.
        """
        raise NotImplementedError()


class NoOpAffinityNorm(AffinityNorm):
    """
    Affinity normalization implementation that does nothing. Exists for backward compatibility with older
    implementations.
    """

    def forward(self, affinities, *, dim):
        return affinities


class BasicAffinityNorm(AffinityNorm):
    """
    Affinity normalization that ensures that all values in the given dimension sum to 1. Performs division by the sum of
    all values.
    """

    def forward(self, affinities, *, dim):
        return affinities.div_(torch.sum(affinities, keepdim=True, dim=dim))


class SoftmaxAffinityNorm(AffinityNorm):
    """
    Affinity normalization that ensures that all values in the given dimension sum to 1 by applying a softmax function.
    Optionally, a multiplicative temperature can be applied to the affinity prior to normalization.
    """

    def __init__(self, temperature=None):
        """
        :param temperature: Multiplicative temperature parameter that is applied prior to the softmax function.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, affinities, *, dim):
        if self.temperature is not None:
            affinities = affinities * self.temperature

        return F.softmax(affinities, dim=dim)
