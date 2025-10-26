from typing import Dict, Iterable, List, Mapping, Optional

import torch
from torch import nn

from ..core import Configurable
from ..layers import MLP, functional

__all__ = [
    "Task",
    "PropertyPrediction",
    "_get_criterion_name",
    "_get_metric_name",
]


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get_criterion_name(name: str) -> str:
    mapping = {
        "mse": "mean squared error",
        "bce": "binary cross entropy",
        "ce": "cross entropy",
    }
    return mapping.get(name, name)


def _get_metric_name(name: str) -> str:
    mapping = {
        "acc": "accuracy",
        "mcc": "matthews correlation",
        "macrof1": "macro f1",
    }
    return mapping.get(name, name)


class Task(nn.Module, Configurable):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return self

    def preprocess(self, train_set, valid_set, test_set):
        pass


class PropertyPrediction(Task):
    def __init__(
        self,
        model: nn.Module,
        task: Optional[Iterable[str]] = None,
        criterion: Mapping[str, float] | str = "mse",
        metric: Optional[Iterable[str]] = None,
        num_class: Optional[Iterable[int]] = None,
        num_mlp_layer: int = 2,
        normalization: bool = False,
        mean: float = 0.0,
        std: float = 1.0,
        weight: Optional[Iterable[float]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.model = model
        self.task = list(task) if task is not None else [0]
        self.verbose = verbose
        if isinstance(criterion, str):
            criterion = {criterion: 1.0}
        self.criterion = criterion
        self.metric = _ensure_list(metric)
        self.normalization = normalization
        self.mean = mean
        self.std = std
        self.num_class = list(num_class) if num_class is not None else [1] * len(self.task)
        if weight is None:
            weight_tensor = torch.ones(len(self.task))
        else:
            weight_tensor = torch.as_tensor(list(weight), dtype=torch.float32)
        self.register_buffer("weight", weight_tensor)

        model_output_dim = getattr(model, "output_dim", None)
        if model_output_dim is None:
            raise ValueError("Model must define `output_dim`")
        final_dim = sum(self.num_class) if "ce" in self.criterion else len(self.task)
        hidden_dims = [model_output_dim] * (num_mlp_layer - 1) + [final_dim]
        self.mlp = MLP(model_output_dim, hidden_dims)

    def parameters(self):  # type: ignore[override]
        for p in self.model.parameters():
            yield p
        for p in self.mlp.parameters():
            yield p

    def target(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        targets = [batch[field] for field in self.task]
        targets = [torch.as_tensor(t, device=self.device) for t in targets]
        target = torch.stack(targets, dim=-1)
        return target

    def predict(self, batch: Mapping[str, torch.Tensor], all_loss=None, metric=None) -> torch.Tensor:
        graph = batch["graph"]
        input_feature = getattr(graph, "node_feature", None)
        output = self.model(graph, input_feature, all_loss, metric)
        graph_feature = output.get("graph_feature")
        if graph_feature is None:
            raise ValueError("Model output must contain `graph_feature`")
        pred = self.mlp(graph_feature)
        return pred

    def forward(self, batch: Mapping[str, torch.Tensor]):
        pred = self.predict(batch)
        target = self.target(batch)
        loss = torch.nn.functional.mse_loss(pred, target)
        metric = {"loss": loss}
        return loss, metric
