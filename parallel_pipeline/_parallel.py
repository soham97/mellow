import abc
import functools
import queue
import sys
import threading
from concurrent import futures
from multiprocessing.pool import Pool
from typing import Callable, Any, Optional, Iterable, List, Set, Dict

from ._host import ParallelPipelineHost
from ._interfaces import IParallelPipeline, IBatchProcessor

__all__ = ["ParallelPipeline"]

CompletionCallback = Callable[[Any], None]


class CancellationToken:
    """
    Holds cancellation status and root cause exception
    """
    __slots__ = ["_lock", "_cancelled", "_exception", "_error_callback"]

    def __init__(self):
        self._lock = threading.Lock()
        self._cancelled = True
        self._exception = None
        self._error_callback = None

    def activate(self, error_callback: Callable[[], None]):
        """
        Activates token and sets error callback (introducing circular ref count dependency)

        Args:
            error_callback: callback to call on reported exception
        """
        with self._lock:
            assert self._error_callback is None
            assert error_callback is not None
            self._cancelled = False
            self._error_callback = error_callback

    def deactivate(self):
        """
        Deactivates token and removes error callback (breaking circular reference)
        """
        with self._lock:
            if self._error_callback is None:
                return

            self._cancelled = True
            self._exception = None
            self._error_callback = None

    @property
    def is_cancelled(self):
        return self._cancelled

    def rethrow(self):
        with self._lock:
            exception = self._exception
            if exception is not None:
                self._exception = None
                raise exception

    def cancel(self):
        with self._lock:
            assert self._error_callback is not None
            self._cancelled = True

    def on_exception(self, exception: BaseException):
        with self._lock:
            if self._cancelled:
                return

            self._exception = exception
            self._cancelled = True
            self._error_callback()


class PriorityTaskQueue(abc.ABC):
    """
    Base class for priority task queue

    Task priorities are required to give run tasks on later pipeline stages first, reducing number of half-processed
    tasks in pipeline
    """
    def __init__(self, *, cancellation_token: CancellationToken, active_task_limit: int):
        self._cancellation_token = cancellation_token
        self._active_task_limit = int(active_task_limit)

        self._task_queue = queue.PriorityQueue()
        self._lock = self._create_lock()
        self._serial = 0
        self._cancelled = False

    @staticmethod
    def _create_lock():
        """
        Create lock to use by queue (ExecutorTaskQueue requires recurrent lock)

        Returns:
            Lock object
        """
        return threading.Lock()

    def apply(self, priority, func: Callable[[Any], Any], task, completion_callback: Optional[CompletionCallback]):
        """
        Queue task and with completion callback.

        Tasks beyond active_task_limit are kept in priority queue which allows
        to schedule higher priority task first.

        Tasks already submitted to underlying implementation can not be rescheduled.

        Args:
            priority: task priority
            func: processor function
            task: task (processor argument)
            completion_callback: completion callback
        """
        with self._lock:
            if self._check_cancelled():
                return

            try:
                if self._active_task_count < self._active_task_limit:
                    self._apply(func, task, completion_callback)
                else:
                    self._serial = serial = self._serial + 1
                    self._task_queue.put((priority, serial, func, task, completion_callback))
            except BaseException as e:
                self._cancel(e)

    @property
    @abc.abstractmethod
    def _active_task_count(self) -> int:
        """
        Number of tasks submitted to underlying implementation (which does not support priorities)
        """
        pass

    def cancel(self):
        """
        Cancel all queued tasks
        """
        with self._lock:
            self._cancel(None)

    def _check_cancelled(self):
        if self._cancelled:
            return

        if self._cancellation_token.is_cancelled:
            self._cancel(None)

    def _pump(self):
        """
        Pump priority queue
        """
        while self._active_task_count < self._active_task_limit and not self._task_queue.empty():
            _, _, func, task, completion_callback = self._task_queue.get()
            self._apply(func, task, completion_callback)

    @abc.abstractmethod
    def _apply(self, func: Callable[[Any], Any], task, completion_callback: Optional[CompletionCallback]) -> None:
        """
        Submit task to underlying implementation

        Args:
            func: task processor
            task: task object
            completion_callback: completion callback
        """
        pass

    def _cancel(self, exception):
        if self._cancelled:
            return

        self._cancelled = True

        while not self._task_queue.empty():
            self._task_queue.get()

        self._cancel_active_tasks()

        if exception is not None:
            self._cancellation_token.on_exception(exception)

    @abc.abstractmethod
    def _cancel_active_tasks(self):
        """
        Cancel tasks already submitted to underlying imlementation
        """
        pass


class PoolTaskQueue(PriorityTaskQueue):
    """
    Priority task queue for multiprocessing.pool.Pool
    """
    def __init__(self, pool: Pool, *, cancellation_token: CancellationToken, active_task_limit: int):
        super().__init__(cancellation_token=cancellation_token, active_task_limit=active_task_limit)
        self._pool = pool
        self._async_callback_list = set()

    @property
    def _active_task_count(self):
        return len(self._async_callback_list)

    def _on_task_complete(self, callback, result):
        with self._lock:
            if callback.async_result is None:
                return False

            try:
                callback.async_result = None
                self._async_callback_list.discard(callback)

                if self._check_cancelled():
                    return False

                if isinstance(result, BaseException):
                    self._cancel(result)
                    return False

                self._pump()
                return True
            except BaseException as e:
                self._cancel(e)
                return False

    def _apply(self, func: Callable[[Any], Any], task, completion_callback: Optional[CompletionCallback]):
        callback = AsyncCallback(self, completion_callback)
        callback.async_result = self._pool.apply_async(func, (task,), callback=callback, error_callback=callback)
        self._async_callback_list.add(callback)

    def _cancel_active_tasks(self):
        for callback in self._async_callback_list:
            callback.async_result = None

        self._async_callback_list.clear()


class AsyncCallback:
    """
    Multiprocessing pool task callback object
    """
    def __init__(self, task_queue: PoolTaskQueue, completion_callback: Optional[CompletionCallback]):
        self._task_queue = task_queue
        self._completion_callback = completion_callback
        self.async_result = None

    def __call__(self, result):
        task_queue = self._task_queue
        if task_queue is None:
            return
        self._task_queue = None
        # noinspection PyProtectedMember
        if not task_queue._on_task_complete(self, result):
            return
        if self._completion_callback is not None:
            self._completion_callback(result)


class ExecutorTaskQueue(PriorityTaskQueue):
    """
    Priority task queue for thread pool executor
    """
    _future_set: Set[futures.Future]

    def __init__(self, executor: futures.Executor, *, thread_count: int, cancellation_token: CancellationToken,
                 active_task_limit: int):
        super().__init__(cancellation_token=cancellation_token, active_task_limit=active_task_limit)
        self._executor = executor
        self._thread_count = thread_count
        self._future_set = set()

    @staticmethod
    def _create_lock():
        return threading.RLock()

    @property
    def thread_count(self):
        return self._thread_count

    @property
    def executor(self):
        return self._executor

    @property
    def _active_task_count(self):
        return len(self._future_set)

    def _apply(self, func: Callable[[Any], Any], task, completion_callback: CompletionCallback):
        with self._lock:
            future = self._executor.submit(self._execute, func, task, completion_callback)
            future.add_done_callback(self._done)
            if not future.done():
                self._future_set.add(future)
                return

        try:
            future.result()
        except futures.CancelledError:
            pass

    def _execute(self, func: Callable[[Any], Any], task, completion_callback: CompletionCallback):
        if self._cancellation_token.is_cancelled:
            return

        task = func(task)
        if completion_callback is not None:
            completion_callback(task)

    def _done(self, future: futures.Future):
        with self._lock:
            try:
                if future not in self._future_set:
                    return

                self._future_set.remove(future)
                future.result()
                self._pump()
            except BaseException as e:
                self._cancel(e)

    def _cancel_active_tasks(self):
        for future in self._future_set:
            future.cancel()
        self._future_set.clear()


class IStage(abc.ABC):
    """
    Pipeline stage interface
    """
    @property
    @abc.abstractmethod
    def max_batch_size(self) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def start(self, completion_callback: CompletionCallback) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def join(self) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(self, task) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def on_input_closed(self, total_count: int) -> None:
        raise NotImplementedError()


class ParallelStageImpl(IStage):
    """
    Parallel pipeline stage implementation
    """
    def __init__(self, *,
                 task_queue: PriorityTaskQueue,
                 priority: int,
                 processor: Callable[[Any], Any],
                 skip_check: Optional[Callable[[Any], bool]]):
        self._task_queue = task_queue
        self._priority = priority
        self._processor = processor
        self._skip_check = skip_check
        self._completion_callback = None

    @property
    def max_batch_size(self):
        return 1

    def start(self, completion_callback: CompletionCallback):
        assert self._completion_callback is None
        self._completion_callback = completion_callback

    def join(self):
        self._completion_callback = None

    def apply(self, task):
        if self._skip_check is not None and self._skip_check(task):
            return False

        completion_callback = self._completion_callback
        if completion_callback is None:
            return True

        self._task_queue.apply(self._priority, self._processor, task, completion_callback)
        return True

    def on_input_closed(self, total_count: int):
        pass


class BatchedStageImpl(IStage):
    """
    Batched stage implementation
    """
    _close_future: Optional[futures.Future]

    def __init__(self, *,
                 task_queue: ExecutorTaskQueue,
                 priority: int,
                 processor: IBatchProcessor,
                 skip_check: Optional[Callable[[Any], bool]],
                 max_batch_size: int,
                 cancellation_token: CancellationToken):
        self._task_queue = task_queue
        self._lock = threading.Lock()
        self._priority = priority
        self._processor = processor
        self._skip_check = skip_check
        self._max_batch_size = max_batch_size
        self._cancellation_token = cancellation_token
        self._completion_callback = None

        # number of received tasks to track end of input sequence
        self._received_tasks = 0
        self._total_tasks = None

        # number of tasks accepted for processing (after skip_check)
        self._accepted_tasks = 0
        self._is_accept_closed = False

        # number of tasks submitted to IBatchExecutor
        self._submitted_tasks = 0
        # number of tasks currently batched in IBatchExecutor
        self._batched_tasks = 0
        # indicator of closed IBatchExecutor
        self._is_closed = False

        # future to wait for IBatchExecutor closure
        self._close_future = None

    @property
    def max_batch_size(self):
        return self._max_batch_size

    def start(self, completion_callback: CompletionCallback):
        with self._lock:
            assert self._completion_callback is None
            self._completion_callback = completion_callback

    def join(self):
        with self._lock:
            if self._completion_callback is None:
                return

            self._completion_callback = None
            # we should be already closed except for error handing scenarios
            if self._accepted_tasks > 0 and not self._is_closed:
                self._submit_close(True)

        if self._close_future is not None:
            try:
                self._close_future.result()
            except futures.CancelledError:
                pass

    def apply(self, task):
        is_applied = self._skip_check is None or not self._skip_check(task)
        with self._lock:
            if self._completion_callback is None:
                return True

            self._received_tasks += 1
            if is_applied:
                self._accepted_tasks += 1
            self._check_accept_closed()

        if is_applied:
            self._task_queue.apply(self._priority, self._put, task, None)

        return is_applied

    def on_input_closed(self, total_count: int):
        with self._lock:
            assert self._completion_callback is not None and self._total_tasks is None
            self._total_tasks = total_count
            self._check_accept_closed()

    def _check_accept_closed(self):
        """
        Check whether the number of accepted tasks is finally known
        """
        assert not self._is_accept_closed
        if self._total_tasks is None or self._total_tasks != self._received_tasks:
            return
        self._is_accept_closed = True

        # if we already submitted all the accepted tasks, we need to flush executor to get buffered tasks done
        if self._submitted_tasks == self._accepted_tasks and not self._is_closed:
            if self._accepted_tasks == 0:
                # no task was ever accepted
                self._is_closed = True
            else:
                # we can not be sure that there is no task batched from other thread
                self._submit_close(False)

    def _submit_close(self, cancelled: bool):
        """
        Send close signal to executor thread
        Args:
            cancelled: whether cancel is requested
        """
        if self._close_future is not None:
            return
        self._close_future = self._task_queue.executor.submit(self._close, cancelled)

    def _put(self, task):
        """
        Main processing method on executor thread

        Args:
            task: task to process
        """
        if self._is_closed:
            # if there are tasks to closed executor - processing must be cancelled
            assert self._cancellation_token.is_cancelled
            return

        # submit and count task
        self._submitted_tasks += 1
        self._processor.put(task)
        self._batched_tasks += 1

        # early cancellation check
        if self._cancellation_token.is_cancelled:
            self._processor.flush(True)
            self._is_closed = True
            return

        # check for forced buffer flush: too many tasks or last task submitted
        if self._batched_tasks >= self._max_batch_size or \
                (self._is_accept_closed and self._submitted_tasks == self._accepted_tasks):
            self._processor.flush()

        self._pump()

    def _pump(self):
        """
        Pump processed tasks
        """
        while self._processor.n_available > 0:
            assert self._batched_tasks > 0
            task = self._processor.get()

            self._batched_tasks -= 1
            if self._batched_tasks == 0 and self._is_accept_closed and self._submitted_tasks == self._accepted_tasks:
                # mark processing as closed as soon as all tasks are processed
                self._is_closed = True

            self._completion_callback(task)

    def _close(self, cancelled):
        """
        Close stage by flushing buffered tasks

        Args:
            cancelled: whether cancel is requested
        """
        if self._is_closed:
            return

        if not cancelled:
            cancelled = self._cancellation_token.is_cancelled
        try:
            if cancelled:
                self._processor.flush(True)
                self._is_closed = True
            else:
                self._processor.flush()
                self._pump()
                assert self._is_closed
        except BaseException as e:
            self._cancellation_token.on_exception(e)


class ParallelPipeline(IParallelPipeline):
    """
    Parallel pipeline implementation
    """
    _pool_task_queue: Optional[PoolTaskQueue]
    _executor_task_queue_map: Dict[str, ExecutorTaskQueue]
    _stage_list: List[IStage]

    def __init__(self, host: ParallelPipelineHost, active_task_limit: int):
        self._host = host
        self._cancellation_token = CancellationToken()
        self._active_task_limit = active_task_limit
        self._pool_task_queue = None
        self._executor_task_queue_map = dict()
        self._stage_list = list()
        self._result_queue = queue.SimpleQueue()
        self._is_started = False

    def add_parallel_stage(self, processor: Callable[[Any], Any], *,
                           skip_check: Optional[Callable[[Any], bool]] = None,
                           thread_pool: Optional[str] = None):
        if thread_pool is None:
            task_queue = self._pool_task_queue
            if task_queue is None:
                self._pool_task_queue = task_queue = PoolTaskQueue(self._host.get_process_pool(),
                                                                   cancellation_token=self._cancellation_token,
                                                                   active_task_limit=self._host.process_count * 4)
        else:
            task_queue = self._get_executor_task_queue(thread_pool)

        stage = ParallelStageImpl(task_queue=task_queue,
                                  priority=-len(self._stage_list),
                                  processor=processor,
                                  skip_check=skip_check)
        self._add_stage(stage)

    def add_batched_stage(self, processor: IBatchProcessor, *,
                          thread_name: str, max_batch_size: int,
                          skip_check: Optional[Callable[[Any], bool]] = None):
        task_queue = self._get_executor_task_queue(thread_name)
        if task_queue.thread_count != 1:
            raise ValueError(f'thread pool {thread_name} for batched operation must have only one thread')

        stage = BatchedStageImpl(task_queue=task_queue,
                                 priority=-len(self._stage_list),
                                 processor=processor,
                                 skip_check=skip_check,
                                 max_batch_size=max_batch_size,
                                 cancellation_token=self._cancellation_token)
        self._add_stage(stage)

    def _get_executor_task_queue(self, name: str):
        task_queue = self._executor_task_queue_map.get(name)
        if task_queue is None:
            thread_pool, thread_count = self._host.get_thread_executor(name)
            task_queue = ExecutorTaskQueue(executor=thread_pool,
                                           thread_count=thread_count,
                                           cancellation_token=self._cancellation_token,
                                           active_task_limit=thread_count)
            self._executor_task_queue_map[name] = task_queue

        return task_queue

    def _add_stage(self, stage: IStage):
        assert not self._is_started
        self._stage_list.append(stage)

    def __call__(self, task_list: Iterable):
        assert not self._is_started
        try:
            self._is_started = True
            self._cancellation_token.activate(self._on_error)

            max_queued_size = 0
            for idx, stage in enumerate(self._stage_list):
                stage.start(functools.partial(self._apply_next_stage, idx + 1))
                max_queued_size += stage.max_batch_size - 1

            max_queued_size = max(max_queued_size + 1, self._active_task_limit)

            task_count = 0
            completed_task_count = 0
            for task in task_list:
                task_count += 1
                self._apply_next_stage(0, task)

                while True:
                    result = self._get_task_result(task_count - completed_task_count >= max_queued_size)
                    if result is None:
                        break

                    completed_task_count += 1
                    yield result

            for stage in self._stage_list:
                stage.on_input_closed(task_count)

            while completed_task_count < task_count:
                result = self._get_task_result(True)
                completed_task_count += 1
                yield result

        finally:
            self._join()

    def _apply_next_stage(self, idx: int, task):
        if self._cancellation_token.is_cancelled:
            return

        try:
            while idx < len(self._stage_list):
                stage = self._stage_list[idx]
                if stage.apply(task):
                    return

                idx += 1

            self._result_queue.put(task)
        except BaseException as e:
            self._cancellation_token.on_exception(e)

    def _join(self):
        self._cancellation_token.deactivate()

        exception = None
        if self._pool_task_queue is not None:
            try:
                self._pool_task_queue.cancel()
            except BaseException as e:
                if exception is None:
                    exception = e

        for task_queue in self._executor_task_queue_map.values():
            try:
                task_queue.cancel()
            except BaseException as e:
                if exception is None:
                    exception = e

        for stage in self._stage_list:
            try:
                stage.join()
            except BaseException as e:
                if exception is None:
                    exception = e

        while not self._result_queue.empty():
            self._result_queue.get()

        if exception is not None and sys.exc_info()[0] is None:
            raise exception

    def _on_error(self):
        self._result_queue.put(self._cancellation_token)

    def _get_task_result(self, block: bool):
        self._cancellation_token.rethrow()
        assert not self._cancellation_token.is_cancelled

        if not block and self._result_queue.empty():
            return None

        try:
            result = self._result_queue.get(block=block)
        except queue.Empty:
            return None

        if result == self._cancellation_token:
            self._cancellation_token.rethrow()
            assert False, 'cancellation token must hold an exception'

        return result
