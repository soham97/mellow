import io
import functools
import logging
import random
import sys
import time
import traceback
import numpy as np
import tarfile
import json
from typing import Callable, Any, Sequence, Tuple, Union, List, BinaryIO, Iterable, Dict, Optional
import math
import re
from torch import nn
import torch
import os
import yaml

__all__ = [
    "read_epoch_marker", "write_epoch_marker","LossTrackingLRScheduler",\
    "GradNormTracker","LazyConverter",
    "NumpyJSONEncoder",
]

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

def numparams(model):
    """Calculate the total number of parameters of a NN model."""
    tot = 0
    tot_trainable = 0
    for p in model.parameters():
        tot += math.prod(p.size())
        if p.requires_grad:
            tot_trainable += math.prod(p.size())
    return tot, tot_trainable

def group_weight_decay_params(model: nn.Module, *,
                              weight_decay=1e-5,
                              rnn_weight_decay=None,
                              exclude_bias_bn_from_weight_decay=True,
                              no_weight_decay_list=()):
    """Group model parameters to exclude biases and BN from weight decay."""

    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    rnn_decay = []
    no_decay = []
    decay_names = []
    rnn_decay_names = []
    no_decay_names = []

    rnn_regexp = None
    if rnn_weight_decay is not None:
        rnn_regexp = re.compile(r".(?:weight_(?:ih|hh|hr)|bias_(?:ih|hh))_l\d+$")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if rnn_regexp is not None and rnn_regexp.search(name):
            rnn_decay.append(param)
            rnn_decay_names.append(name)
        elif name in no_weight_decay_list or (
                exclude_bias_bn_from_weight_decay and (param.ndim <= 1 or name.endswith(".bias"))):
            no_decay.append(param)
            no_decay_names.append(name)
        else:
            decay.append(param)
            decay_names.append(name)

    if len(no_decay_names) == 0 and len(rnn_decay_names) == 0:
        return decay

    params = list()
    if len(decay_names) > 0:
        logging.info(f"apply weight decay ({weight_decay}) to: {decay_names}")
        params.append(dict(
            params=decay,
            weight_decay=weight_decay
        ))
    if len(rnn_decay_names) > 0:
        logging.info(f"apply rnn weight decay ({rnn_weight_decay}) to: {rnn_decay_names}")
        params.append(dict(
            params=rnn_decay,
            weight_decay=rnn_weight_decay
        ))
    if len(no_decay_names) > 0:
        logging.info(f"ignore weight decay for: {no_decay_names}")
        params.append(dict(
            params=no_decay,
            weight_decay=0.
        ))

    return params

class LossTrackingLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, *,
                 window_size: int = 4000, threshold: float = 0.1, scale_factor: float = 0.5,
                 lr_min: float = 0.):
        self.optimizer = optimizer
        self.scale_factor = scale_factor
        self.lr_min = lr_min

        self.threshold = threshold
        self.loss_history = np.zeros(window_size)
        self.loss_mean_history = np.zeros(window_size)
        self._loss_sum = 0.
        self._loss_squared_sum = 0.
        self._cursor = window_size - 1
        self._reset_index = self._cursor

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    @property
    def window_size(self):
        return len(self.loss_history)

    def track_loss(self, loss: float):
        window_size = self.window_size
        self._cursor = cursor = (self._cursor + 1) % window_size

        last_loss = self.loss_history[cursor]
        self.loss_history[cursor] = loss

        self._loss_sum = loss_sum = self._loss_sum + loss - last_loss
        self.loss_mean_history[cursor] = loss_sum / window_size

        self._loss_squared_sum += loss * loss - last_loss * last_loss

        if self._cursor == self._reset_index:
            self._reset_index = -1

    @property
    def loss_mean(self):
        return float(self.loss_mean_history[self._cursor])

    @property
    def loss_var(self):
        loss_mean = self.loss_mean
        return math.sqrt(max(self._loss_squared_sum / self.window_size - loss_mean * loss_mean, 0))

    @property
    def is_loss_unmoved(self):
        if self._reset_index >= 0:
            # Haven't collected all the window
            return False

        loss_min = np.min(self.loss_mean_history)
        loss_max = np.max(self.loss_mean_history)
        return loss_max - loss_min < self.threshold * self.loss_var

    def reset(self):
        self._reset_index = self._cursor

    def step(self):
        if not self.is_loss_unmoved:
            return

        self.reset()

        changed = False
        for g in self.optimizer.param_groups:
            lr = g['lr']
            lr_new = max(lr * self.scale_factor, self.lr_min)
            if lr_new < lr:
                g['lr'] = lr_new
                changed = True

        if changed:
            lr_str = ', '.join(str(x) for x in self.get_last_lr())
            logging.info(f"scaling down LR due to unmoved average loss value to {lr_str}")

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
class GradNormTracker:
    def __init__(self, *, initial_l2_norm, initial_max_norm, overdrive_factor: float = 2.5, momentum=0.995):
        self.initial_norm = initial_l2_norm, initial_max_norm
        self.overdrive_factor = overdrive_factor
        self.momentum = momentum
        self.running_norm = dict()
        self.history = dict()

    def track_and_clip_(self, parameters: Sequence[Tuple[str, nn.Parameter]]):
        min_scale = 1.

        l2_norm_list = list()
        for n, p in parameters:
            g = p.grad
            if g is None:
                continue

            running_l2_norm, running_max_norm = self.running_norm.get(n, self.initial_norm)
            g = g.detach().flatten()

            l2_norm = torch.linalg.norm(g, 2.0).item()
            self._add_history(n + ".l2", l2_norm)
            l2_norm_list.append(l2_norm)
            running_l2_norm, min_scale = self._clip_grad(l2_norm, running_l2_norm, min_scale)

            max_norm = torch.linalg.norm(g, np.inf).item()
            self._add_history(n + ".max", max_norm)
            running_max_norm, min_scale = self._clip_grad(max_norm, running_max_norm, min_scale)

            self.running_norm[n] = running_l2_norm, running_max_norm

        if min_scale < 1.:
            logging.info(f"scaling gradients by {min_scale}")
            min_scale_tensor = torch.tensor(min_scale)
            for _, p in parameters:
                g = p.grad
                if g is None:
                    continue

                g = g.detach()
                min_scale_tensor = min_scale_tensor.to(g.device)
                g.mul_(min_scale_tensor)

        l2_norm = torch.linalg.norm(torch.tensor(l2_norm_list), 2.0).item()
        return l2_norm, min_scale

    def _clip_grad(self, norm, running_norm, min_scale):
        norm_limit = running_norm * self.overdrive_factor
        if norm > norm_limit:
            min_scale = min(min_scale, norm_limit / norm)
            # limit running average impact
            norm = min(norm, norm_limit * self.overdrive_factor)

        running_norm = self.momentum * running_norm + (1 - self.momentum) * norm
        return running_norm, min_scale

    def _add_history(self, name, value):
        history = self.history.get(name)
        if history is None:
            self.history[name] = history = list()

        history.append(value)

    def truncate_history(self):
        for history in self.history.values():
            history.clear()

    def state_dict(self):
        return self.running_norm

    def load_state_dict(self, state_dict):
        self.running_norm.clear()
        self.running_norm.update(state_dict)

class LazyConversionDict:
    """
    Dictionary of lazily converted items
    """

    def __init__(self, source: dict, converter: Callable[[Any], Any]):
        self._source = source
        self._converter = converter
        self._cache = dict()

    def __getitem__(self, item):
        value = self._cache.get(item, None)
        if value is None:
            try:
                self._cache[item] = value = self._converter(self._source[item])
            except:
                self._cache[item] = value = {key:self._converter(self._source[item][key]) for key in self._source[item].keys()}

        return value

class LazyConverter:
    """
    Object with lazily converted attributes
    """

    def __init__(self, source, converter: Callable[[Any], Any]):
        self._source = source
        self._converter = converter

    def __getattr__(self, name):
        value = self._converter(getattr(self._source, name))
        setattr(self, name, value)
        return value
    
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        elif isinstance(obj, np.bool):
            return bool(obj)

        return super().default(obj)
    
def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)
