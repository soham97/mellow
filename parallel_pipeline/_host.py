import dataclasses
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import Callable, Optional, Dict

from ._interfaces import IParallelPipelineHost, IParallelPipeline


@dataclasses.dataclass
class ThreadPoolData:
    name: str
    thread_count: int = 1
    executor: Optional[ThreadPoolExecutor] = None


class ParallelPipelineHost(IParallelPipelineHost):
    _process_pool: Optional[Pool]
    _thread_pool_map: Dict[str, ThreadPoolData]

    def __init__(self, process_count: int, *,
                 initializer: Optional[Callable[[], None]] = None):
        process_count = int(process_count)
        if process_count < 0:
            raise ValueError(f'invalid process count {process_count}')

        self._lock = threading.Lock()
        self._process_count = process_count
        self._process_intializer = initializer
        self._process_pool = None
        self._thread_pool_map = dict()

    @property
    def process_count(self):
        return self._process_count

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        terminate_list = list()
        with self._lock:
            pool = self._process_pool
            if pool is not None:
                self._process_pool = None
                terminate_list.append(pool.terminate)

            for data in self._thread_pool_map.values():
                executor: Optional[ThreadPoolExecutor] = data.executor
                if executor is not None:
                    data.executor = None
                    terminate_list.append(executor.shutdown)

            self._thread_pool_map.clear()

        exception = None
        for f in terminate_list:
            try:
                f()
            except BaseException as e:
                if exception is None:
                    exception = e

        if exception is not None and sys.exc_info()[0] is None:
            raise exception

    def get_process_pool(self):
        with self._lock:
            pool = self._process_pool
            if pool is None:
                if self.process_count == 0:
                    raise ValueError(f'parallel processing is disabled')

                self._process_pool = pool = Pool(processes=self.process_count,
                                                 initializer=self._process_intializer)

            return pool

    def configure_thread_pool(self, name: str, thread_count: int = 1):
        name = str(name)
        if not name:
            raise ValueError(f'invalid thread pool name')

        thread_count = int(thread_count)
        if thread_count <= 0:
            raise ValueError(f'invalid thread count {thread_count}')

        with self._lock:
            data = self._thread_pool_map.get(name)
            if data is not None:
                raise ValueError(f'thread pool {name} is already configured')

            self._thread_pool_map[name] = ThreadPoolData(name, thread_count)

    def get_thread_executor(self, name: str):
        name = str(name)
        if not name:
            raise ValueError(f'invalid thread pool name')

        with self._lock:
            data = self._thread_pool_map.get(name)
            if data is None:
                self._thread_pool_map[name] = data = ThreadPoolData(name)

            executor = data.executor
            thread_count = data.thread_count
            if data.executor is None:
                data.executor = executor = ThreadPoolExecutor(thread_count, thread_name_prefix=f'pph-{name}')

            return executor, thread_count

    def create_pipeline(self, *, active_task_limit: int) -> IParallelPipeline:
        active_task_limit = int(active_task_limit)
        if active_task_limit <= 0:
            raise ValueError(f'invalid active task limit {active_task_limit}')

        if self.process_count == 0:
            from ._sequential import SequentialPipeline
            return SequentialPipeline()
        else:
            from ._parallel import ParallelPipeline
            return ParallelPipeline(self, active_task_limit)
