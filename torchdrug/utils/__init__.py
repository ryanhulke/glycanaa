import ast
import functools
import hashlib
import inspect
import os
from typing import Any, Callable, Mapping

import torch

from . import comm

__all__ = [
    "literal_eval",
    "copy_args",
    "cached_property",
    "cuda",
    "cpu",
    "get_line_count",
    "download",
    "pretty",
    "comm",
]


def literal_eval(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def copy_args(parent: Callable) -> Callable:
    parent_signature = inspect.signature(parent)

    def decorator(func: Callable) -> Callable:
        func.__signature__ = parent_signature
        func.__doc__ = getattr(parent, "__doc__", None)
        return func

    return decorator


class cached_property(property):
    def __init__(self, func: Callable[[Any], Any]) -> None:
        super().__init__(func)
        self.func = func

    def __get__(self, obj: Any, owner: Any) -> Any:
        if obj is None:
            return self
        value = obj.__dict__.get(self.func.__name__, None)
        if value is None:
            value = self.func(obj)
            obj.__dict__[self.func.__name__] = value
        return value


def _apply(data: Any, op: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    if isinstance(data, torch.Tensor):
        return op(data)
    if isinstance(data, Mapping):
        return type(data)({k: _apply(v, op) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(_apply(v, op) for v in data)
    return data


def cuda(data: Any, *args, **kwargs) -> Any:
    return _apply(data, lambda t: t.cuda(*args, **kwargs))


def cpu(data: Any) -> Any:
    return _apply(data, lambda t: t.cpu())


def get_line_count(path: str) -> int:
    with open(path, "r") as fin:
        return sum(1 for _ in fin)


def download(url: str, directory: str, filename: str = None, md5: str = None) -> str:
    directory = os.path.expanduser(directory)
    if filename is None:
        filename = os.path.basename(url)
    dest = os.path.join(directory, filename)
    if not os.path.exists(dest):
        raise RuntimeError(
            f"Automatic download disabled in this lightweight TorchDrug substitute. "
            f"Please manually place the file `{filename}` at `{directory}`."
        )
    if md5 is not None:
        with open(dest, "rb") as fin:
            file_hash = hashlib.md5(fin.read()).hexdigest()
        if file_hash != md5:
            raise RuntimeError(f"MD5 checksum mismatch for {dest} (expected {md5}, got {file_hash})")
    return dest


class pretty:
    @staticmethod
    def long_array(values):
        return "[" + ", ".join(str(int(v)) for v in values) + "]"
