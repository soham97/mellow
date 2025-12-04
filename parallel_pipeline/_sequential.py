import dataclasses
from typing import Callable, Any, Iterable, Optional, List, Tuple

from ._interfaces import IParallelPipeline, IBatchProcessor

__all__ = ["SequentialPipeline"]


@dataclasses.dataclass
class StageData:
    skip_check: Optional[Callable[[Any], bool]]
    max_batch_size: int
    parallel_processor: Callable[[Any], Any] = None
    batch_processor: IBatchProcessor = None


class SequentialPipeline(IParallelPipeline):
    _stage_list: List[StageData]

    def __init__(self):
        self._stage_list = list()

    def add_parallel_stage(self, processor: Callable[[Any], Any], *,
                           skip_check: Optional[Callable[[Any], bool]] = None,
                           thread_pool: Optional[str] = None):
        self._stage_list.append(StageData(skip_check=skip_check, max_batch_size=1, parallel_processor=processor))

    def add_batched_stage(self, processor: IBatchProcessor, *,
                          thread_name: str, max_batch_size: int,
                          skip_check: Optional[Callable[[Any], bool]] = None):
        self._stage_list.append(StageData(skip_check=skip_check, max_batch_size=max_batch_size,
                                          batch_processor=processor))

    def __call__(self, task_list: Iterable):
        cursor = 0
        for idx in range(len(self._stage_list)):
            stage = self._stage_list[idx]
            if stage.parallel_processor is not None:
                continue

            task_list = self._run_parallel(cursor, idx, task_list)
            task_list = self._run_batched(stage.batch_processor, task_list,
                                          max_batch_size=stage.max_batch_size,
                                          skip_check=stage.skip_check)
            cursor = idx + 1

        return self._run_parallel(cursor, len(self._stage_list), task_list)

    def _run_parallel(self, start: int, end: int, task_list: Iterable):
        if start < end:
            return self._run_parallel_impl(tuple(self._stage_list[start:end]), task_list)
        else:
            return task_list

    @staticmethod
    def _run_parallel_impl(stage_list: Tuple[StageData], task_list: Iterable):
        for task in task_list:
            for stage in stage_list:
                if stage.skip_check is not None and stage.skip_check(task):
                    continue

                task = stage.parallel_processor(task)

            yield task

    @staticmethod
    def _run_batched(executor: IBatchProcessor, task_list: Iterable, *,
                     max_batch_size: int, skip_check: Optional[Callable[[Any], bool]] = None):
        batched_tasks = 0
        for task in task_list:
            if skip_check is not None and skip_check(task):
                yield task
                continue

            executor.put(task)
            batched_tasks += 1

            if batched_tasks >= max_batch_size:
                executor.flush()
            while executor.n_available > 0:
                assert batched_tasks > 0
                yield executor.get()
                batched_tasks -= 1

        executor.flush()
        while executor.n_available > 0:
            assert batched_tasks > 0
            yield executor.get()
            batched_tasks -= 1

        assert batched_tasks == 0
