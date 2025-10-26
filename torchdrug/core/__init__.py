import copy
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar

import torch
from torch import nn
from torch.utils import data as torch_data
from torch.utils.data._utils.collate import default_collate

from ..utils import comm
from ..data import Graph, PackedGraph


class Registry:
    """Light-weight string to class/function registry."""

    def __init__(self):
        self._map: Dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(obj: Type[Any]) -> Type[Any]:
            if name in self._map and self._map[name] is not obj:
                raise KeyError(f"Registry key {name} has been registered already")
            self._map[name] = obj
            setattr(obj, "registry_key", name)
            return obj

        return decorator

    def get(self, name: str) -> Any:
        if name not in self._map:
            raise KeyError(f"Unknown registry key {name}")
        return self._map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._map


R = Registry()


T = TypeVar("T")


class Configurable:
    """Utility class that knows how to build objects from configuration dictionaries."""

    registry = R

    @classmethod
    def _build(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "class" in value:
            return cls.load_config_dict(value)
        if isinstance(value, list):
            return [cls._build(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls._build(v) for v in value)
        return value

    @classmethod
    def load_config_dict(cls, cfg: Mapping[str, Any]) -> Any:
        if "class" not in cfg:
            raise ValueError("Configuration dictionary must contain a `class` entry")
        cfg = dict(cfg)
        class_name = cfg.pop("class")
        target = cls.registry.get(class_name)
        params = {k: cls._build(v) for k, v in cfg.items() if k != "name"}
        return target(**params)


@dataclass
class _State:
    epoch: int = 0
    iteration: int = 0


class Engine:
    """A very small training helper that mimics the original TorchDrug Engine API."""

    def __init__(
        self,
        task: nn.Module,
        train_set: Optional[torch_data.Dataset],
        valid_set: Optional[torch_data.Dataset],
        test_set: Optional[torch_data.Dataset],
        optimizer: torch.optim.Optimizer,
        batch_size: int = 1,
        gpus: Optional[Iterable[int]] = None,
        num_worker: int = 0,
        **kwargs,
    ) -> None:
        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.scheduler = kwargs.get("scheduler")
        self._state = _State()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if gpus is not None and torch.cuda.is_available():
            gpu_list = list(gpus)
            if len(gpu_list) > 1:
                self.model = nn.DataParallel(self.model, device_ids=gpu_list)
            self.device = torch.device(f"cuda:{gpu_list[0]}")
        self.model.to(self.device)

    @property
    def epoch(self) -> int:
        return self._state.epoch

    def _collate(self, samples: List[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not samples:
            return {}
        result: Dict[str, Any] = {}
        keys = samples[0].keys()
        for key in keys:
            values = [sample[key] for sample in samples]
            first = values[0]
            if isinstance(first, Graph):
                result[key] = first.__class__.pack(values)
            else:
                try:
                    result[key] = default_collate(values)
                except TypeError:
                    result[key] = values
        return result

    def _data_loader(self, dataset: torch_data.Dataset, training: bool) -> Iterator[Any]:
        if dataset is None:
            return iter([])
        loader = torch_data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=training,
            num_workers=self.num_worker,
            collate_fn=getattr(dataset, "collate", self._collate),
        )
        return iter(loader)

    def train(self, num_epoch: int = 1, log_interval: int = 10) -> None:
        self.model.train()
        for _ in range(num_epoch):
            self._state.epoch += 1
            data_iter = self._data_loader(self.train_set, training=True)
            for i, batch in enumerate(data_iter, start=1):
                self._state.iteration += 1
                batch = self._move_to_device(batch)
                loss, metric = self.model(batch)
                if isinstance(loss, Mapping):
                    loss = loss["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if log_interval and i % log_interval == 0 and comm.get_rank() == 0:
                    print(f"epoch {self.epoch:04d} iter {i:04d} | loss {loss.item():.4f}")
            if self.scheduler is not None:
                self.scheduler.step()

    def evaluate(self, split: str) -> Dict[str, float]:
        if split == "train":
            dataset = self.train_set
        elif split == "valid":
            dataset = self.valid_set
        elif split == "test":
            dataset = self.test_set
        else:
            raise ValueError(f"Unknown split `{split}`")
        if dataset is None:
            return {}
        self.model.eval()
        metrics: Dict[str, float] = {}
        with torch.no_grad():
            for batch in self._data_loader(dataset, training=False):
                batch = self._move_to_device(batch)
                loss, metric = self.model(batch)
                for key, value in metric.items():
                    metrics.setdefault(key, 0.0)
                    metrics[key] += float(value)
        for key in list(metrics.keys()):
            metrics[key] /= max(len(dataset), 1)
        return metrics

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def _move_to_device(self, batch: Any) -> Any:
        if isinstance(batch, Graph):
            return batch.to(self.device)
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        if isinstance(batch, Mapping):
            return {k: self._move_to_device(v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(v) for v in batch)
        return batch
