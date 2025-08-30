"""Common utilities."""

import argparse
import functools
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import torch


def stopwatch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Measure the execution time of any function"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> Callable[..., Any]:
        start_time = time.time()
        result = fn(*args, **kwargs)
        end_time = time.time()
        logging.info(
            "Function %s took %.3f seconds to execute",
            fn.__name__,
            end_time - start_time,
        )
        print(
            "Function %s took %.3f seconds to execute" %
            (fn.__name__,
            end_time - start_time,
            )
        )
        return result

    return wrapper


def save_argparse(args: argparse.Namespace, out_path: str) -> None:
    """Serializes the argparse.Namespace to a JSON file.

    Args:
        args: The parsed command-line arguments.
        out_path: The path to save the JSON file.
    """
    config_dict = vars(args)
    with open(out_path, "w") as f:
        json.dump(config_dict, f)


def get_device() -> torch.device:
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device_str = "mps"
    return torch.device(device_str)
