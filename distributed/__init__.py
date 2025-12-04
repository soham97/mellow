import abc
import logging
import sys
from typing import Dict, List

import torch
import torch.nn as nn

__all__ = ["IDistributedContext", "get_local_context"]


class IDistributedContext(abc.ABC):
    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # noinspection PyBroadException
        try:
            self.cleanup()
        except Exception as e:
            if exc_type is None:
                raise

            logging.warning("exception in distributed cleanup", exc_info=sys.exc_info())

    @abc.abstractmethod
    def initialize(self) -> None:
        pass

    @abc.abstractmethod
    def cleanup(self) -> None:
        pass

    @abc.abstractmethod
    def world_size(self) -> int:
        return 1

    @abc.abstractmethod
    def rank(self) -> int:
        return 0

    @abc.abstractmethod
    def local_rank(self) -> int:
        return 0

    @abc.abstractmethod
    def broadcast(self, tensor: torch.Tensor):
        pass

    @abc.abstractmethod
    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        pass

    @abc.abstractmethod
    def broadcast_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    @abc.abstractmethod
    def create_distributed_model(self, model: nn.Module) -> nn.Module:
        pass

    @abc.abstractmethod
    def get_distributed_model_state(self, model: nn.Module):
        return model.state_dict()

    @abc.abstractmethod
    def all_reduce(self, tensor: torch.Tensor, *, average: bool = True) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def all_gather_object(self, obj) -> List:
        pass

    @abc.abstractmethod
    def barrier(self):
        pass


class LocalDistributedContext(IDistributedContext):
    def initialize(self) -> None:
        pass

    def cleanup(self) -> None:
        pass

    def world_size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0

    def local_rank(self) -> int:
        return 0

    def broadcast(self, tensor: torch.Tensor):
        pass

    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor]):
        pass

    def create_distributed_model(self, model: nn.Module) -> nn.Module:
        return model

    def get_distributed_model_state(self, model: nn.Module):
        return model.state_dict()

    def broadcast_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        pass

    def all_reduce(self, tensor: torch.Tensor, *, average: bool = True) -> torch.Tensor:
        return tensor

    def all_gather_object(self, obj) -> List:
        return [obj]

    def barrier(self):
        pass


_local_context = LocalDistributedContext()


def get_local_context() -> IDistributedContext:
    return _local_context
