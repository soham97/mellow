"""
Configure Python logger to work properly in AzureML.
"""
import functools
import logging
import multiprocessing
import os
import sys
import threading
from io import RawIOBase, TextIOWrapper, BufferedWriter
from typing import Optional

__all__ = ["configure_logging", "WorkerLogSink"]


def configure_logging(*,
                      level: Optional[int] = None,
                      log_format: Optional[str] = None,
                      file_name: Optional[str] = None):
    if log_format is None:
        log_format = "%(asctime)s P%(process)05d %(levelname).1s: %(message)s"

    if level is None:
        # Logging level DEBUG should be set only selectively.
        # Otherwise there is too much output from azureml packages
        level = logging.INFO

    warn_handler = logging.StreamHandler(stream=sys.stderr)
    warn_handler.setLevel(logging.WARNING)

    info_handler = logging.StreamHandler(stream=sys.stdout)
    info_handler.addFilter(lambda record: record.levelno < logging.WARNING)

    handlers = [
        info_handler,
        warn_handler

    ]

    if file_name is not None:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        handlers.append(logging.FileHandler(file_name, encoding="utf-8"))

    # noinspection PyArgumentList
    logging.basicConfig(format=log_format, level=level, handlers=handlers, force=True)


class _LogQueueHandler(logging.Handler):
    def __init__(self, put):
        super().__init__()
        self._put = put

    def emit(self, record: logging.LogRecord):
        d = dict(record.__dict__)
        d.pop('message', None)
        d['msg'] = record.getMessage()
        d['args'] = None
        d['exc_info'] = None
        self._put(d)


class _LogSink(RawIOBase):
    def __init__(self, tag):
        self._tag = tag

    def writable(self) -> bool:
        return True

    def write(self, msg):
        text = str(msg, encoding='utf-8', errors='replace').strip('\r\n')
        logging.info(f"{self._tag}: {text}")
        return len(msg)


def _init_worker(log_level, queue):
    handler = _LogQueueHandler(queue.put)
    logging.basicConfig(level=log_level, handlers=(handler,))

    sys.stdout.close()
    sys.stdout = TextIOWrapper(BufferedWriter(_LogSink("STDOUT")),
                               encoding='utf-8', line_buffering=True)
    sys.stderr.close()
    sys.stderr = TextIOWrapper(BufferedWriter(_LogSink("STDERR")),
                               encoding='utf-8', line_buffering=True)

    import platform
    if platform.system() == 'Linux':
        # close standard handles as well
        nul = os.open('/dev/null', os.O_RDWR)
        try:
            os.dup2(nul, 0)
        finally:
            os.close(nul)

        os.dup2(0, 1)
        os.dup2(0, 2)


class WorkerLogSink:
    """
    Class to collect logs from worker processess
    """

    def __init__(self):
        self._queue = multiprocessing.get_context().SimpleQueue()
        self._sink_thread = None
        self._init_worker = functools.partial(_init_worker, logging.root.level, self._queue)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def start(self):
        sink_thread = self._sink_thread
        if sink_thread is not None:
            return

        queue = self._queue
        assert queue is not None

        sink_thread = threading.Thread(name="worker_log_sink", daemon=True,
                                       target=self._sink_thread_main, args=(queue,))
        sink_thread.start()

        self._sink_thread = sink_thread

    def close(self):
        queue = self._queue
        if queue is None:
            return

        self._queue = None
        sink_thread = self._sink_thread
        self._sink_thread = None

        if sink_thread is not None:
            queue.put(None)
            sink_thread.join()

    @staticmethod
    def _sink_thread_main(queue):
        while True:
            msg = queue.get()
            if msg is None:
                break

            msg = logging.makeLogRecord(msg)
            # noinspection PyBroadException
            try:
                logging.getLogger(msg.name).handle(msg)
            except Exception:
                logging.error("exception in log sink thread", exc_info=True, stack_info=True)

    @property
    def init_worker(self):
        assert self._sink_thread is not None
        return self._init_worker
