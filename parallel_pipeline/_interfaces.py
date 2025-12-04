import abc
from typing import Callable, Any, Iterable, Optional, Generic, TypeVar

__all__ = ["IBatchProcessor", "IParallelPipeline", "IParallelPipelineHost"]

T = TypeVar('T')
V = TypeVar('V')


class IBatchProcessor(Generic[T, V]):
    @abc.abstractmethod
    def put(self, task: T) -> None:
        pass

    @abc.abstractmethod
    def flush(self, discard=False) -> None:
        pass

    @property
    @abc.abstractmethod
    def n_available(self) -> int:
        pass

    @abc.abstractmethod
    def get(self) -> V:
        pass


class IParallelPipeline(abc.ABC):
    @abc.abstractmethod
    def add_parallel_stage(self, processor: Callable[[Any], Any], *,
                           skip_check: Optional[Callable[[Any], bool]] = None,
                           thread_pool: Optional[str] = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def add_batched_stage(self, processor: IBatchProcessor, *,
                          thread_name: str, max_batch_size: int,
                          skip_check: Optional[Callable[[Any], bool]] = None):
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, task_list: Iterable):
        raise NotImplementedError()


class IParallelPipelineHost(abc.ABC):
    @property
    @abc.abstractmethod
    def process_count(self) -> int:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def configure_thread_pool(self, name: str, thread_count: int = 1) -> None:
        pass

    @abc.abstractmethod
    def create_pipeline(self, *, active_task_limit: int) -> IParallelPipeline:
        pass
