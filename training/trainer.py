import logging
import torch
import os
import sys
from typing import Optional, Dict, Iterable, Tuple, Iterator, TypeVar, Any, Sequence, Set, Callable
import numpy as np
import random
import functools
from functools import partial
import math
import json
from pathlib import Path
import time
from tqdm import tqdm
from enum import Enum
import io
import gzip
import pandas as pd
import glob
from pandas import Series
from scipy.io.wavfile import write
import traceback
from torch.nn import functional as F
import distributed
from training import log
from models.model import get_model_class
from data.sampler import CustomDistributedSampler
from utils.utils import retry, numparams, group_weight_decay_params
from utils.utils import GradNormTracker, LossTrackingLRScheduler, LazyConversionDict
from metrics.get_metrics import Metric
from models.generate import generate_greedy, generate_greedy_batch

class TrainerMode(Enum):
    Train = "train"
    EvaluateCheckpoint = "evaluate_checkpoint"

def worker_init_fn(logging_initializer, worker_id):
    # Initialize logging for this worker
    # This prevents "I/O operation on closed file" errors
    try:
        logging_initializer()
    except Exception as e:
        # If logging initialization fails, continue without it
        # to avoid breaking the data loading
        pass  # Silently fail - don't print to avoid spam
    
    # Suppress ALL logging from workers to avoid spam
    # Workers should not log - only main process should log
    logging.getLogger().setLevel(logging.CRITICAL)  # Only critical errors
    
    # Suppress transformers and other library warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set random seed for reproducibility
    seed = torch.utils.data.get_worker_info().seed
    sync_random_seed(seed)

def sync_random_seed(seed):
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)

class Trainer:

    def __init__(self, config, distributed_ctx: distributed.IDistributedContext = distributed.get_local_context()):
        self.distributed = distributed_ctx
        # noinspection PyPackageRequirements
        self.logger = logging.getLogger(__name__)
        self.config = config

        # init device
        self.device = None
        self.device_type = None
        if config["gpu"] and torch.cuda.is_available():
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.device = torch.device(self.device_type)

        self.use_mixed_precision = config["train"]["mixed_precision"]["use_mixed_precision"]
        if self.use_mixed_precision:
            amp_dtype = config.get("mixed_precision_dtype", "float16")
            if amp_dtype == "float16":
                dtype = torch.float16
            elif amp_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown mixed precision dtype: {amp_dtype}")
        else:
            if self.device_type == "cuda":
                dtype = torch.float16
            elif self.device_type == "cpu":
                dtype = torch.bfloat16
        self.fast_dtype = dtype

        log_file_name = self.config.get('log_file_name')
        if log_file_name is not None:
            # select log file name by local rank
            log_file_name = log_file_name.split(os.path.pathsep)
            if len(log_file_name) == 0:
                log_file_name = None
            else:
                log_file_name = log_file_name[distributed_ctx.local_rank() % len(log_file_name)]

        # Fix level for all installed handlers
        logger = logging.getLogger()
        if len(logger.handlers) == 0:
            from log import configure_logging
            configure_logging(file_name=log_file_name)
        elif log_file_name is not None:
            # initializing separate file logging
            for handler in logger.handlers:
                level = logger.getEffectiveLevel()
                if level > handler.level:
                    handler.setLevel(level)

            # And enable INFO to separate log file
            logger.setLevel(logging.INFO)

            os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(next(iter(logger.handlers)).formatter)
            logger.addHandler(file_handler)

        self._worker_log_sink = None

        seed = config["train"]["random_seed"]
        if seed is None:
            seed = torch.seed()
            logging.info(f"Random seed is {seed}")
        else:
            if isinstance(seed, Sequence):
                seed = seed[self.distributed.rank()]
            else:
                seed += self.distributed.rank()
            logging.info(f"Setting random seed to {seed}")
            torch.manual_seed(seed)
        sync_random_seed(seed)

        self._is_cleanup_enabled = 0
        self._cleanup_hook_list = list()

        self._parallel_pipeline_host = None
        self._fpie_inference_test = None
        self._fpie_temp_dir = None

    def get_model(self):
        # model
        model_type = self.config['model']['model_type']
        Model = get_model_class(model_type=model_type)

        model = Model(
            audioenc_name = self.config['model']['encoder']['audioenc_name'],
            d_in = self.config['model']['encoder']['out_emb'],
            text_decoder = self.config['model']['decoder']['text_decoder'],
            prefix_length = self.config['model']['decoder']['prefix_length'],
            freeze_text_decoder_weights = self.config['model']['decoder']['freeze_gpt_weights'],
            d_out = self.config['model']['encoder']['d_proj'],
            use_pretrained_audioencoder = self.config['model']['encoder']['use_pretrained_audioencoder'],
            freeze_audio_encoder_weights= self.config['model']['encoder']['freeze_audio_encoder_weights'],
            pretrained_audioencoder_path = self.config['model']['encoder']['pretrained_audioencoder_path'],
        )
        
        return model

    def get_num_data_workers(self):
        if self.config["train"]["num_workers"] is not None:
            # backward compatibility
            return self.config["train"]["num_workers"]

        num_workers = os.cpu_count() * self.config["num_data_workers_per_cpu"]
        num_workers /= self.distributed.world_size()
        num_workers = math.ceil(num_workers - 0.05)
        return num_workers

    def _cleanup_worker_log_sink(self):
        sink = self._worker_log_sink
        if sink is None:
            return

        self._worker_log_sink = None
        sink.close()

    def get_worker_logging_initializer(self):
        sink = self._worker_log_sink
        if sink is None:
            from .log import WorkerLogSink
            self._worker_log_sink = sink = WorkerLogSink()
            self._cleanup_hook_list.append(self._cleanup_worker_log_sink)

        sink.start()
        return sink.init_worker

    def _get_data_worker_init_fn(self):
        return functools.partial(worker_init_fn, self.get_worker_logging_initializer())

    def _cleanup_parallel_pipeline_host(self):
        parallel_pipeline_host = self._parallel_pipeline_host
        if parallel_pipeline_host is None:
            return

        self._parallel_pipeline_host = None
        parallel_pipeline_host.close()

    def get_parallel_pipeline_host(self):
        assert self._is_cleanup_enabled > 0
        host = self._parallel_pipeline_host
        if host is None:
            from parallel_pipeline import create_parallel_pipeline_host
            host = create_parallel_pipeline_host(self.get_num_data_workers(),
                                                 initializer=self.get_worker_logging_initializer())

            host.configure_thread_pool("cognitive", self.config['wer_eval']['cognitive']['threads'])
            self._cleanup_hook_list.append(self._cleanup_parallel_pipeline_host)
            self._parallel_pipeline_host = host

        return host

    def __enter__(self):
        self._is_cleanup_enabled += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._is_cleanup_enabled > 0
        self._is_cleanup_enabled -= 1
        if self._is_cleanup_enabled > 0:
            return

        cleanup_hook_list = self._cleanup_hook_list
        self._cleanup_hook_list = list()
        for hook in reversed(cleanup_hook_list):
            # noinspection PyBroadException
            try:
                hook()
            except Exception:
                logging.warning("Exception while runinning cleanup hook", exc_info=True, stack_info=True)
                
    def get_data(self, key):
        # creating and return dataset + sampler
        sampling_rate = self.config['data']['sampling_rate']
        segment_seconds = self.config['data']['segment_seconds']
        datafiles = self.config['data'][key]
        data_path = self.config['data']['datapath']
        sampling_rate = self.config['data']['sampling_rate']
        tokenizer_type = self.config['data']['tokenizer_type']
        ip_text_len = self.config['data']['ip_text_len']
        op_text_len = self.config['data']['op_text_len']
        num_workers = self.get_num_data_workers()

        if self.config["mode"] is TrainerMode.Train:
            from data.audiotext_dataset import AudioTextDataset, collate_fn

            dataset = AudioTextDataset(
                data_path=data_path,
                datafiles=datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                tokenizer_type=tokenizer_type,
                ip_text_len=ip_text_len,
                op_text_len=op_text_len,
            )

            data_sampler = CustomDistributedSampler(
                        dataset, shuffle=True, 
                        num_replicas=self.distributed.world_size(), 
                        rank=self.distributed.rank(), 
                        drop_last=True
                        )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config["train"]["batch_size"], num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True, sampler=data_sampler,
                drop_last=True, worker_init_fn=self._get_data_worker_init_fn(),
                persistent_workers=self.config["train"]["persistent_data_workers"] if num_workers > 0 else False,
            )
        elif self.config["mode"] is TrainerMode.EvaluateCheckpoint:
            from data.audiotext_eval_dataset import AudioTextEvalDataset, collate_fn
            dataset = AudioTextEvalDataset(
                data_path=data_path,
                datafiles=datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                tokenizer_type=tokenizer_type,
                ip_text_len=ip_text_len,
                op_text_len=op_text_len,
            )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config["train"]["batch_size"], num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True,
                drop_last=False, worker_init_fn=self._get_data_worker_init_fn(),
                persistent_workers=self.config["train"]["persistent_data_workers"] if num_workers > 0 else False,
            )

            data_sampler = None
        else:
            mode = self.config["mode"]
            raise ValueError(f"{mode} dataloader mode not supported'")
        
        return dataset, data_sampler, data_loader

    def train(self):
        self.logger.info("Training Mellow with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        # creating dataset
        dataset, data_sampler, data_loader = self.get_data("datafiles")   
        start_epoch = 0

        # Construct NN model
        model = self.get_model()
        model = model.to(self.device)
        if self.config["resume_checkpoint"] and self.config["resume_checkpoint"] != "":
            checkpoint = torch.load(self.config["resume_checkpoint"], map_location=self.device)
            checkpoint = checkpoint['state_dict']
            model.load_state_dict(checkpoint, strict=False)

        model = self.distributed.create_distributed_model(model)
        model.train()

        if self.distributed.rank() == 0:
            self.logger.info("Mellow has %d parameters of which %d are trainable" % numparams(model))
            self.logger.info("%s", model)

        # add weight decay to appropriate layers
        weight_decay = self.config["train"]["optimizer"]["weight_decay"]
        parameters = group_weight_decay_params(
            model,
            weight_decay=weight_decay,
            rnn_weight_decay=None,
            exclude_bias_bn_from_weight_decay=self.config.get("exclude_bias_bn_from_weight_decay", False)
        )

        optimizer_type = self.config["train"]["optimizer"]["optimizer_type"]
        optimizer = getattr(torch.optim, optimizer_type)(
            parameters,
            lr=float(self.config["train"]["optimizer"]["learning_rate"]), # * self.distributed.world_size(), # disabled multiplying the world size with the learning rate, this creates hardship for multi-node training
            weight_decay=weight_decay,
        )
        del parameters

        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        lr_scheduler = None
        loss_tracker = None
        if self.config["train"]["optimizer"]["scheduler"] is not None:
            lr_schedule = self.config["train"]["optimizer"]["scheduler"]
            if lr_schedule == "step":
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 100, 150], gamma=0.5)
            elif lr_schedule == "cosine":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config["train"]["num_epochs"])
            elif lr_schedule == "cosine_restarts":
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15)
            elif lr_schedule == "loss_tracking":
                lr_scheduler = loss_tracker = LossTrackingLRScheduler(
                    optimizer, lr_min=self.config.get("lr_min", 0.))
            else:
                raise ValueError(f"No such lr schedule: {lr_schedule}")

        # broadcast optimizer state to all other processes
        self.distributed.broadcast_optimizer_state(optimizer)
        # Train the model
        num_batches_per_epoch = len(data_loader)

        t0 = time.time()
        lowest_accerr_epo = 1000.0
        max_grad_norm = self.config["train"]["max_grad_norm"]
        grad_norm_tracker = GradNormTracker(initial_l2_norm=max_grad_norm, initial_max_norm=10 * max_grad_norm)
    
        loss_history = dict(
            loss=[],
            total_grad_norm=[],
            grad_scale=[]
        )

        os.makedirs(self.config["save_dir"], exist_ok=True)

        total_step = 0
        ignore_index = dataset.tokenizer.encode(dataset.tokenizer.pad_token)[0]
        for epoch in range(start_epoch, self.config["train"]["num_epochs"]):
            # set epoch to use different seeds for different epochs during sampling
            data_sampler.set_epoch(epoch)
            tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)
            metrics_train = {"epoch": epoch}
            accerr_epo = 0  # accumulated error per epoch

            if epoch > 0 and lr_scheduler is not None:
                lr_scheduler.step()
                lr = lr_scheduler.get_last_lr()[0]
            elif lr_scheduler is None:
                lr = optimizer.param_groups[0]["lr"]
                if epoch == 0:
                    print("Starting the training with a learning rate of {}".format(lr))
            
            for ii, batch_data_dict in enumerate(data_loader):
                batch_audio1 = batch_data_dict['waveform1']
                batch_audio2 = batch_data_dict['waveform2']
                batch_input = batch_data_dict['input']
                batch_answer = batch_data_dict['answer']

                batch_answer['attention_mask'] = torch.stack([torch.cat((torch.ones(self.config["model"]["decoder"]["total_prefix_length"]), text), dim=0) for text in batch_answer['attention_mask']])

                input_dict = {
                    "audio1":batch_audio1,
                    "audio2": batch_audio2,
                    "input":batch_input,
                    "answer":batch_answer,
                }
                input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                
                model_outputs = model(input_dict)
                logits = model_outputs.logits[:, self.config["model"]["decoder"]["total_prefix_length"] - 1: -1]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), input_dict['answer']['input_ids'].flatten(), ignore_index=ignore_index)

                optimizer.zero_grad()

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)  # to use the same max_grad_norm value for gradient clipping

                total_norm, grad_scale = grad_norm_tracker.track_and_clip_(list(model.named_parameters()))
                
                grad_scaler.step(optimizer)
                grad_scaler.update()

                loss = self.distributed.all_reduce(loss.detach()).item()
                accerr_epo += loss

                if loss_tracker is not None:
                    loss_tracker.track_loss(loss)

                loss_history["loss"].append(loss)
                loss_history["total_grad_norm"].append(total_norm)
                loss_history["grad_scale"].append(grad_scale)

                # Print log for current step
                total_step += 1
                if total_step % self.config["train"]["log_step"] == 0 and self.distributed.rank() == 0:
                    errdict = {
                        "accerr_epo": accerr_epo,
                        "loss": loss,
                    }

                    errstr = ", ".join(
                        "{}: {:6.3f}(e-6)".format(k, v * 1e6) for k, v in errdict.items()
                    )
                    self.logger.info(
                        "Epoch [%3d/%3d], Step [%3d/%3d], %s",
                        epoch + 1, self.config["train"]["num_epochs"], ii + 1,
                        num_batches_per_epoch, errstr
                    )
                    tqdm_handler.update(1)

                del batch_audio1, batch_audio2, batch_input, batch_answer
                del input_dict, model_outputs


            metrics_train["accerr"] = float(accerr_epo)
            if accerr_epo < lowest_accerr_epo and self.distributed.rank() == 0:
                lowest_accerr_epo = accerr_epo
                self.logger.info("The lowest accumulated error so far is {%f}", accerr_epo)

            # Save the MODEL checkpoint
            is_save_epoch = (epoch + 1) % self.config["train"]["sav_per_num_epochs"] == 0
            if is_save_epoch:

                fname = self._get_checkpoint_name(epoch)
                save_dir = self.config["save_dir"]

                model_fpath = os.path.join(save_dir, "model-" + fname)
                metrics_train["checkpoint"] = model_fpath

                if self.distributed.rank() == 0:
                    os.makedirs(save_dir, exist_ok=True)

                    self.logger.info("Saving model to: %s Total training time: %f hours",
                                        model_fpath, (time.time() - t0) / 3600.0)

                    self._save_model_state(model_fpath, model)
                    
            # distributed: broadcast parameters to ensure that models do not diverge
            self.distributed.broadcast_parameters(model.state_dict())
            self.distributed.broadcast_optimizer_state(optimizer)

    # pylint: disable=too-many-locals
    def evaluate_checkpoint(self):
        self.logger.info("Validate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        model = self.get_model()
        model = model.to(self.device)
        checkpoint = torch.load(self.config["checkpoint_path"], map_location=self.device)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()

        tasks = self.config["data"]["datafiles"]
        val_score = 0
        for task in tasks:
            self.logger.info("Evaluating task %s", task)
            self.config["data"]["datafiles"] = [task]

            metric = Metric(task, self.config["data"]["sampling_rate"])
            dataset, _, data_loader = self.get_data("datafiles")
            num_batches_per_epoch = len(data_loader)
            tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)

            generations, answers, filepaths, inputs = [], [], [], []
            with torch.no_grad():
                for batch_data_dict in tqdm(data_loader):
                    batch_audio1 = batch_data_dict['waveform1']
                    batch_audio2 = batch_data_dict['waveform2']
                    batch_input = batch_data_dict['input']
                    batch_answer = batch_data_dict['answer']
                    batch_answer_text = batch_data_dict['answer_text']
                    batch_input_text = batch_data_dict['input_text']
                    batch_file_paths = batch_data_dict['file_path1']

                    input_dict = {
                        "audio1":batch_audio1,
                        "audio2": batch_audio2,
                        "input":batch_input,
                        "answer":batch_answer,
                    }
                    input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                
                    prefix, _, _ = model.generate_prefix_inference(input_dict)
                    generated_text = generate_greedy_batch(model, data_loader.dataset.tokenizer, embed=prefix)
                    generations += generated_text
                    answers += batch_answer_text
                    inputs += batch_input_text
                    filepaths += batch_file_paths
                    #break

            metric.get_metrics(generations, answers, filepaths)
            if self.distributed.rank() == 0:
                self.logger.info("Task %s results", task)
                for key in metric.metrics.keys():
                    self.logger.info("%s: %f", key, metric.metrics[key]["score"])
            
            # azure logging
            val_score += metric.metrics["main"]["score"]
            taskname = task.split(os.path.sep)[-1].split(".json")[0]
            
    # pylint: disable=too-many-locals
    def evaluate_experiment(self):
        self.logger.info("Evaluate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        model = self.get_model()
        model = model.to(self.device)

        foldername = f"{os.path.sep}".join(self.config["checkpoint_path"].split(os.path.sep)[:-1])
        max_epochs = max([int(f.split("-epo-")[-1].split(".ckpt")[0]) for f in glob.glob(os.path.join(foldername,"*.ckpt"))])

        tasks = self.config["data"]["datafiles"]
        for e in range(1, max_epochs+1):
            checkpoint_path = self.config["checkpoint_path"].replace("-epo-1",f"-epo-{e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=True)
            model.eval()

            val_score = 0
            for task in tasks:
                self.logger.info("Evaluating task %s", task)
                self.config["data"]["datafiles"] = [task]

                metric = Metric(task, self.config["data"]["sampling_rate"])
                dataset, _, data_loader = self.get_data("datafiles")
                num_batches_per_epoch = len(data_loader)
                # tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)

                generations, answers, filepaths, inputs = [], [], [], []
                with torch.no_grad():
                    for batch_data_dict in tqdm(data_loader):
                        batch_audio1 = batch_data_dict['waveform1']
                        batch_audio2 = batch_data_dict['waveform2']
                        batch_input = batch_data_dict['input']
                        batch_answer = batch_data_dict['answer']
                        batch_answer_text = batch_data_dict['answer_text']
                        batch_input_text = batch_data_dict['input_text']
                        batch_file_paths = batch_data_dict['file_path1']

                        input_dict = {
                            "audio1":batch_audio1,
                            "audio2": batch_audio2,
                            "input":batch_input,
                            "answer":batch_answer,
                        }
                        input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                    
                        prefix, _, _ = model.generate_prefix_inference(input_dict)
                        generated_text = generate_greedy_batch(model, data_loader.dataset.tokenizer, embed=prefix)
                        generations += generated_text
                        answers += batch_answer_text
                        inputs += batch_input_text
                        filepaths += batch_file_paths

                metric.get_metrics(generations, answers, filepaths)
                if self.distributed.rank() == 0:
                    self.logger.info(f"Epoch {e}, Task %s results", task)
                    for key in metric.metrics.keys():
                        self.logger.info("%s: %f", key, metric.metrics[key]["score"])
                
                # azure logging
                val_score += metric.metrics["main"]["score"]
                taskname = task.split(os.path.sep)[-1].split(".json")[0]
                if self.distributed.rank() == 0:
                    self.log_step_metric(taskname, metric.metrics["main"]["score"])
                    
            if self.distributed.rank() == 0:
                self.log_step_metric(f"val_score", val_score)

    def _get_checkpoint_name(self, epoch: int):
        prefix = self.config.get("myconfig")
        prefix = prefix + '-' if prefix else ''
        return f"{prefix}-epo-{epoch + 1}.ckpt"

    @retry
    def _save_model_state(self, model_fpath, model):
        with open(model_fpath, "wb") as f:
            torch.save(self.distributed.get_distributed_model_state(model), f)
            # make sure data is sent to blobstorage
            f.flush()
            f.close()
