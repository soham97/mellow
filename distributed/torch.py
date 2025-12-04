import logging
import os
from typing import Dict, List

import torch.distributed
import torch.nn as nn
import torch.nn.parallel
import re

from . import IDistributedContext

__all__ = ["TorchDistributedContext"]


class TorchDistributedContext(IDistributedContext):
    def __init__(self, backend: str):
        self._backend = backend
        self._local_rank = 0
        self._initialized = False

    def size(self):
        """Find world size from SLURM or torch distributed environment
        :rtype: int
        """
        # Try SLURM environment variables first
        if "SLURM_NTASKS" in os.environ:
            return int(os.environ["SLURM_NTASKS"])
        return int(os.environ.get("WORLD_SIZE") or 1)

    def local_size(self):
        """Find local size (GPUs per node) from SLURM or torch distributed environment
        :rtype: int
        """
        # Try SLURM environment variables first
        if "SLURM_GPUS_PER_NODE" in os.environ:
            # Handle format like "gpu:4" or just "4"
            gpus = os.environ["SLURM_GPUS_PER_NODE"]
            if ":" in gpus:
                return int(gpus.split(":")[-1])
            return int(gpus)
        elif "SLURM_NTASKS_PER_NODE" in os.environ:
            return int(os.environ["SLURM_NTASKS_PER_NODE"])
        return int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE") or 1)

    def _get_init_method_itp(self):
        # For SLURM, use environment variable initialization
        if "SLURM_JOB_ID" in os.environ:
            # SLURM-based initialization
            # Master address is typically set by SLURM or needs to be determined
            if "MASTER_ADDR" not in os.environ:
                # Get the first node from SLURM_NODELIST
                import subprocess
                try:
                    # Use scontrol to get the first node
                    nodelist = os.environ.get("SLURM_NODELIST", "")
                    if nodelist:
                        cmd = f"scontrol show hostnames {nodelist}"
                        result = subprocess.run(cmd.split(), capture_output=True, text=True)
                        nodes = result.stdout.strip().split('\n')
                        if nodes:
                            os.environ["MASTER_ADDR"] = nodes[0]
                            logging.info(f"Set MASTER_ADDR to {nodes[0]} from SLURM_NODELIST")
                except Exception as e:
                    logging.warning(f"Failed to determine MASTER_ADDR from SLURM: {e}")
                    
            if "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "29500"
        
        assert("MASTER_ADDR" in os.environ)
        init_method = f"tcp://{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT', '29500')}"
        # Only print from rank 0 to avoid spam (check LOCAL_RANK or SLURM_LOCALID since rank() isn't available yet)
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        if local_rank == 0:
            print(f"init_method for torch.distributed.init_process_group: {init_method}", flush=True)
        return init_method

    def initialize(self) -> None:
        # Determine local rank from environment
        # Priority: LOCAL_RANK > SLURM_LOCALID > error
        if "LOCAL_RANK" in os.environ:
            self._local_rank = int(os.environ["LOCAL_RANK"])
        elif "SLURM_LOCALID" in os.environ:
            # SLURM environment
            self._local_rank = int(os.environ["SLURM_LOCALID"])
        else:
            # Try to infer from RANK and SLURM_NTASKS_PER_NODE
            if "RANK" in os.environ and "SLURM_NTASKS_PER_NODE" in os.environ:
                rank = int(os.environ["RANK"])
                tasks_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
                self._local_rank = rank % tasks_per_node
            else:
                raise ValueError(
                    "Unable to determine local rank. "
                    "Expected LOCAL_RANK, SLURM_LOCALID, or RANK+SLURM_NTASKS_PER_NODE in environment. "
                    f"Found: {', '.join([k for k in os.environ.keys() if 'RANK' in k or 'SLURM' in k])}"
                )

        # Set RANK and WORLD_SIZE if not set (for SLURM)
        if "RANK" not in os.environ and "SLURM_PROCID" in os.environ:
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
        
        if "WORLD_SIZE" not in os.environ and "SLURM_NTASKS" in os.environ:
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        
        # Update LOCAL_RANK in environment for consistency
        os.environ["LOCAL_RANK"] = str(self._local_rank)
        
        # DON'T set CUDA_VISIBLE_DEVICES here - torchrun/DDP handles device assignment
        # Instead, explicitly set the device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank())
        
        # Initialize master address if needed
        self._get_init_method_itp()
        
        torch.distributed.init_process_group(self._backend)
        
        # Only log from rank 0 to avoid duplicate messages
        if self.rank() == 0:
            logging.info(f"Initializing distributed process group with {self._backend} backend...")
            logging.info(f"Initialized distributed process group: world_size = {self.world_size()}, "
                        f"local_rank = {self.local_rank()}, rank = {self.rank()}")
            if torch.cuda.is_available():
                logging.info(f"CUDA device set for each process")
        
        self._initialized = True

    def cleanup(self) -> None:
        if self._initialized:
            self._initialized = False
            torch.distributed.destroy_process_group()

    def world_size(self) -> int:
        return torch.distributed.get_world_size()

    def rank(self) -> int:
        return torch.distributed.get_rank()

    def local_rank(self) -> int:
        return self._local_rank

    def broadcast(self, tensor: torch.Tensor):
        torch.distributed.broadcast(tensor, src=0)

    def broadcast_parameters(self, parameters: Dict[str, torch.Tensor]):
        wait_list = list()
        for _, tensor in sorted(parameters.items()):
            wait_list.append(torch.distributed.broadcast(tensor.contiguous(), src=0, async_op=True))

        for op in wait_list:
            op.wait()

    def create_distributed_model(self, model: nn.Module) -> nn.Module:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # Set find_unused_parameters=False for better performance
        # Only set to True if your model has control flow that leads to unused parameters
        model = nn.parallel.DistributedDataParallel(
            model, 
            find_unused_parameters=False
        )
        self.broadcast_parameters(model.state_dict())
        return model

    def get_distributed_model_state(self, model: nn.Module):
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        return model.module.state_dict()

    def broadcast_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        wait_list = list()

        bcast_device = None
        if torch.distributed.get_backend() == torch.distributed.Backend.NCCL:
            bcast_device = torch.device('cuda', torch.cuda.current_device())
        sync_list = list()

        def process_dict(obj: dict):
            for _, value in sorted(obj.items()):
                if isinstance(value, dict):
                    process_dict(value)
                elif isinstance(value, torch.Tensor):
                    if bcast_device is not None and value.device.type != bcast_device.type:
                        # move CPU tensors to broadcast device
                        bcast_value = value.to(bcast_device)
                        sync_list.append((value, bcast_value))
                        value = bcast_value

                    wait_list.append(torch.distributed.broadcast(value, src=0, async_op=True))

        process_dict(optimizer.state_dict()['state'])
        for op in wait_list:
            op.wait()

        for tensor, bcast_tensor in sync_list:
            tensor[...] = bcast_tensor[...]

    def all_reduce(self, tensor: torch.Tensor, *, average: bool = True) -> torch.Tensor:
        torch.distributed.all_reduce(tensor)
        if average:
            tensor /= self.world_size()
        return tensor

    def all_gather_object(self, obj) -> List:
        obj_list = [None] * self.world_size()
        torch.distributed.all_gather_object(obj_list, obj)
        return obj_list

    def barrier(self):
        torch.distributed.barrier()
