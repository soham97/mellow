from typing import Optional, Callable

from ._interfaces import *

__all__ = ["IBatchProcessor", "IParallelPipeline", "IParallelPipelineHost", "create_parallel_pipeline_host"]


def create_parallel_pipeline_host(process_count: int, *,
                                  initializer: Optional[Callable[[], None]] = None):
    from ._host import ParallelPipelineHost
    return ParallelPipelineHost(process_count, initializer=initializer)
