import os
from datetime import datetime
import logging
import sys

import distributed
from utils.launch_utils import parse_args, multiprocessing_init, parse_mode
from training.log import configure_logging
from training.trainer import Trainer, TrainerMode

def main():
    args, conf = parse_args()
    args = parse_mode(args)
    args.job_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_dir = os.path.join(args.save_dir, args.job_id)
    args.save_adir = os.path.join(args.save_dir, "audio")
    # create output folder
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_adir, exist_ok=True)

    configure_logging()
    multiprocessing_init()

    if args.distributed_backend is None:
        distributed_ctx = distributed.get_local_context()
    else:
        import distributed.torch as impl
        distributed_ctx = impl.TorchDistributedContext(args.distributed_backend)

    with distributed_ctx:
        # Suppress logging from non-zero ranks IMMEDIATELY after distributed context is initialized
        # This prevents duplicate "Proceeding with config" and other messages
        if distributed_ctx.rank() > 0:
            logging.getLogger().setLevel(logging.ERROR)  # Changed to ERROR to suppress warnings too
        
        # copy configs to output folder (only rank 0)
        if distributed_ctx.rank() == 0:
            import shutil
            shutil.copy(conf, os.path.join(args.save_dir, "conf.yaml"))

        logging.info(f"Proceeding with config {vars(args)}")

        with Trainer(vars(args), distributed_ctx=distributed_ctx) as trainer:
            # noinspection PyBroadException
            try:
                return run_command(trainer, args)
            except Exception:
                # make sure to print error to per-process log file as well, not only to console output
                logging.error("error running command", exc_info=sys.exc_info())
                if args.reraise_exceptions:
                    raise

                return -1

def run_command(trainer: Trainer, args):
    if args.mode is TrainerMode.Train:
        trainer.train()
    elif args.mode is TrainerMode.EvaluateCheckpoint:
        trainer.evaluate_checkpoint()
    else:
        assert False, f"Unknown operation mode {args.mode}"

if __name__ == "__main__":
    exit(main())