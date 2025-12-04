import os
import argparse
import yaml
from enum import Enum
from numbers import Number
from typing import Optional, Dict, Iterable, Tuple, Iterator, TypeVar, Any, Sequence, Set, Callable
import numpy as np
import logging
import torch
import random
import pathlib
import functools
from training.trainer import TrainerMode
import time

def retry(_func=None, *, max_tries=5):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            for attempt in range(max_tries - 1):
                # noinspection PyBroadException
                try:
                    return func(*args, **kwargs)
                except BaseException:
                    if attempt == 0:
                        if len(args) > 0:
                            logging.info(f"args: {args}")
                        if len(kwargs) > 0:
                            logging.info(f"kwargs: {kwargs}")
                    logging.warning(f"exception caught, retry #{attempt+1}", exc_info=True)
                    time.sleep(random.random())

            return func(*args, **kwargs)

        return wrapper_retry

    if _func is None:
        return decorator_retry
    else:
        return decorator_retry(_func)

def add_config_file_args(parser: argparse.ArgumentParser):
    """
    Add configuration file options
    """
    parser.add_argument('--config', default="config.yaml")
    parser.add_argument('--dataset-config', default=None)

def support_old_config(config):
    wer_config = config["wer_eval"]
    if 'cognitive' not in wer_config:
        cog_config = {}
        for k in ['endpoint', 'language', 'speech2text_region', 'wer_delay', 'retries_on_err', 'threads']:
            cog_config[k] = wer_config[k]
        wer_config['cognitive'] = cog_config
    if wer_config['wer_engine'] == "jiwer":
        if 'jiwer' not in wer_config:
            jiwer_config = {}
            jiwer_config['calc_cer'] = wer_config['jiwer_metric_type'] == 'cer'
            jiwer_config['calc_wer'] = wer_config['jiwer_metric_type'] == 'wer'
            jiwer_config['normalization'] = 'none'
            wer_config['jiwer'] = jiwer_config
    return config

def add_cmdline_args(parser: argparse.ArgumentParser):
    """
    Add command line options
    """
    add_config_file_args(parser)

    parser.add_argument('--save-dir', type=str, default="outputs", help="folder to save results")
    parser.add_argument('--music-detection-dir', type=str, default="music_detection", 
                        help="folder to save music detection results")
    parser.add_argument('--log-file-name', type=str, help="log file name (or filenames per local rank)")
    parser.add_argument('--num_workers', type=int, help="Number of data generation workers")
    parser.add_argument('--distributed-backend', help="run in distributed mode with specified backend")
    parser.add_argument('--checkpoint_path', help="Path to saved model")
    parser.add_argument('--optimizer_state', help="Path to saved optimizer state", default=argparse.SUPPRESS)
    parser.add_argument('--epoch_marker', help="Path to epoch marker file", default=argparse.SUPPRESS)
    parser.add_argument('--farend_path')
    parser.add_argument('--mic_path')
    parser.add_argument('--out_path')
    parser.add_argument('--enrollment', help="speaker enrollment as floating point array")
    parser.add_argument('--train_data_mount', action='append', required=False,
                        help='train-data container mount point for each Azure region')
    parser.add_argument('--static_data_mount', action='append', required=False,
                        help='static-data container mount point for each Azure region')
    parser.add_argument(
        '--data_dir',
        help='Directory path that contains loopback and micin signals. Used to run batched inference.')
    parser.add_argument(
        '--static_dir', default="",
        help='Directory path that contains speaker embedder models. Used for producing embeddings.')
    parser.add_argument(
        '--lpb_id',
        default='lpb',
        help='Filename identifier for loopback signal. Used when running batched inference from a directory.')
    parser.add_argument(
        '--mic_id',
        default='mic',
        help='Filename identifier for microphone signal. Used when running batched inference from a directory.')
    parser.add_argument(
        '--out_id',
        default='dec',
        help='Filename identifier for enhanced speech. Used when running batched inference from a directory.')
    parser.add_argument(
        '--test_dir',
        help='Directory path that contains loopback and micin signals for mturk test signals.')
    parser.add_argument('--data_cfg')
    parser.add_argument(
        '--repro',
        default=None,
        type=int,
        help='Index of the repro study when using --nb_repro to trigger Azure ML Run')
    parser.add_argument('--skip-decmos-eval', action='store_true')
    parser.add_argument('--user-managed-identity', type=str, default=None, help="User managed identity client ID")
    parser.add_argument(
        '--functiontest',
        action='store_true',
        help="Break after a single training batch, save model and run eval")
    parser.add_argument('--synth-audio-types', default=None, help="audio types to store")
    parser.add_argument('--enhanced_dir_url',
                        help='Blob container url of directory that contains enhanced files for validation')
    parser.add_argument('--enhanced_dir_sas',
                        help='SAS for accessing enhanced_dir_url')
    parser.add_argument(
        '--skip-cp-eval',
        action='store_true',
        help="Skip final checkpoint evaluation")
    parser.add_argument("--fpie-model", help="FPIE model for validation")
    parser.add_argument("--frame-size", type=int, help="FPIE model frame size")
    parser.add_argument("--ns-only", action="store_true", help="whether FPIE model is NS only")
    parser.add_argument("--reraise-exceptions", action="store_true",
                        help="re-raise exceptions to let debugger stop at raise statement")
    
def add_config_args(parser: argparse.ArgumentParser, config: Dict[str, Any],
                    prefix: str, explicit_args: Optional[Set[str]] = None):
    """
    Add arguments for configuration settings found in config
    """
    if explicit_args is None:
        explicit_args = set()
        # noinspection PyProtectedMember
        for n, a in parser._option_string_actions.items():
            explicit_args.add(n)
            explicit_args.add('--' + a.dest)

    for name, option in config.items():
        if isinstance(option, dict):
            add_config_args(parser, option, prefix + name + ".", explicit_args)
            continue

        if not isinstance(option, (Number, str, bool)):
            # skip all unsupported types for now
            continue

        option_name = prefix + str(name)
        if option_name in explicit_args:
            continue

        option_type = type(option)
        if option_type is bool:
            def parse_bool(s):
                v = s.lower()
                if v in ('true', 'on', 'yes', '1'):
                    return True
                if v in ('false', 'off', 'no', '0'):
                    return False
                raise ValueError(f"Can not convert value to bool: '{s}'")

            option_type = parse_bool

        parser.add_argument(option_name, type=option_type, default=argparse.SUPPRESS, )

def set_cmdline_value(config, name, value):
    """
    Set option value from command line in config
    """
    name_list = name.split('.')
    for n in name_list[:-1]:
        config = config[n]

    config[name_list[-1]] = value

def parse_args():
    """
    Parse command line arguments and load configuration files
    """
    cfg_parser = argparse.ArgumentParser(add_help=False)
    add_config_file_args(cfg_parser)
    cfg_arg, _ = cfg_parser.parse_known_args()
    # Only print from rank 0 (check environment variables set by torchrun/SLURM)
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    if local_rank == 0:
        print(f"Using config from {cfg_arg.config}")

    main_config_filename = cfg_arg.config
    with open(main_config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['main_config_dirname'] = os.path.dirname(main_config_filename)
    parser = argparse.ArgumentParser(description='Train or validate source separation models.')
    add_cmdline_args(parser)
    add_config_args(parser, config, "--")

    args = parser.parse_args()
    for name, value in vars(args).items():
        set_cmdline_value(config, name, value)

    return argparse.Namespace(**config), main_config_filename

def parse_mode(args):
    if args.mode == "train":
        args.mode = TrainerMode.Train
    elif args.mode == "evaluate_checkpoint":
        args.mode = TrainerMode.EvaluateCheckpoint
    elif args.mode == "evaluate_experiment":
        args.mode = TrainerMode.EvaluateExperiment
    else:
        print(f"Mode not found. Reverting to {TrainerMode.Train}")
        args.mode = TrainerMode.Train
    return args

def multiprocessing_init():
    """
    Multiprocessing intitialization
    """
    import torch.multiprocessing as mp

    mp_start_method = "forkserver"
    if mp_start_method in mp.get_all_start_methods():
        # Only print from rank 0
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
        if local_rank == 0:
            print(f"Switching to multiprocessing strategy {mp_start_method}")
        mp.set_start_method(mp_start_method)
        mp.set_forkserver_preload(["torch", "audiolib", "audiolib.sig", "pandas", "librosa", "joblib"])

        # try:
        #    mp.set_start_method(mp_start_method)
        #    mp.set_forkserver_preload(["torch", "audiolib", "audiolib.sig", "pandas", "librosa", "joblib"])
        #    print("spawned")
        # except RuntimeError:
        #    ctx = mp.get_context("spawn")
        #    ctx.set_start_method(mp_start_method)
        #    ctx.set_forkserver_preload(["torch", "audiolib", "audiolib.sig", "pandas", "librosa", "joblib"])

    mp.freeze_support()


def sync_random_seed(seed):
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)


def worker_init_fn(logging_initializer, worker_id):
    logging_initializer()

    seed = torch.utils.data.get_worker_info().seed
    sync_random_seed(seed)

    logging.info(f"worker #{worker_id}: pid = {os.getpid()}, seed = {seed}")
    logging.info(f"rand numbers: {np.random.randint(0, 10, 5)}")
