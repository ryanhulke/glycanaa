"""Minimal internal implementation to replace the external TorchDrug dependency.
This module implements a very small subset of the original API that is used by the
GlycANAA code base.  The goal is to provide compatibility with PyTorch 2.x without
requiring the third party TorchDrug package at runtime.
"""

from . import core, utils, layers, data, tasks, metrics

__all__ = ["core", "utils", "layers", "data", "tasks", "metrics"]
