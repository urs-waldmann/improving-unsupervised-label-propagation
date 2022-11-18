from abc import ABC, abstractmethod
from typing import TypeVar, Generic

import torch

T = TypeVar("T")


class AbstractLabelCodec(ABC, Generic[T]):
    @abstractmethod
    def encode(self, data: T) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def decode(self, labels: torch.Tensor) -> T:
        raise NotImplementedError()

    def recode(self, labels: torch.Tensor) -> torch.Tensor:
        return self.encode(self.decode(labels)).to(device=labels.device)
