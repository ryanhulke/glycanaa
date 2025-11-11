import torch
from torch.nn import functional as F

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.GlycanProteinInteraction")
class GlycanProteinInteraction(tasks.Task, core.Configurable):
    """Interaction prediction task that consumes paired protein / glycan inputs."""

    def __init__(self, model, criterion="mse", metric=("mae", "rmse", "spearmanr")):
        super().__init__()
        self.model = model
        if isinstance(criterion, (list, tuple)):
            self.criterion = {name: 1.0 for name in criterion}
        elif isinstance(criterion, dict):
            self.criterion = criterion
        else:
            self.criterion = {criterion: 1.0}
        self.metric = list(metric) if metric is not None else []
        self.weight = None

    def preprocess(self, train_set, valid_set, test_set):
        num_tasks = len(self.task)
        if self.weight is None:
            weight = torch.ones(num_tasks, dtype=torch.float)
        else:
            weight = torch.as_tensor(self.weight, dtype=torch.float)
            if weight.ndim == 0:
                weight = weight.repeat(num_tasks)
        self.register_buffer("weight", weight)

    def predict(self, batch, all_loss=None, metric=None):
        protein = batch["graph1"]
        glycan = batch["graph2"]
        pred = self.model(protein, glycan)
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        return pred

    def target(self, batch):
        targets = []
        for name in self.task:
            value = batch[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float, device=self.device)
            else:
                value = value.to(self.device).float()
            targets.append(value)
        target = torch.stack(targets, dim=-1)
        return target

    def evaluate(self, pred, target):
        metric_dict = {}
        labeled = ~torch.isnan(target)
        for name in self.metric:
            if name == "mae":
                score = F.l1_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0)
            elif name == "rmse":
                score = F.mse_loss(pred, target, reduction="none")
                score = functional.masked_mean(score, labeled, dim=0).sqrt()
            elif name == "spearmanr":
                score = metrics.spearmanr(pred, target, dim=0)
            else:
                raise ValueError(f"Unknown metric `{name}`")
            metric_name = tasks._get_metric_name(name)
            if score.ndim == 0:
                metric_dict[metric_name] = score
            else:
                for idx, task_name in enumerate(self.task):
                    metric_dict[f"{metric_name} [{task_name}]"] = score[idx]
        return metric_dict

    def forward(self, batch):
        all_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target = target.clone()
        target[~labeled] = 0

        weight = self.weight.to(pred.device)
        for name, scale in self.criterion.items():
            if name == "mse":
                loss = F.mse_loss(pred, target, reduction="none")
            elif name == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError(f"Unknown criterion `{name}`")
            loss = functional.masked_mean(loss, labeled, dim=0)
            if loss.ndim == 0:
                loss = loss.unsqueeze(0)
            cur_weight = weight[: loss.numel()]
            normalized = (loss * cur_weight).sum() / cur_weight.sum().clamp_min(1e-12)
            metric_name = tasks._get_criterion_name(name)
            metric[metric_name] = normalized
            all_loss = all_loss + normalized * scale

        metric.update(self.evaluate(pred, target))
        return all_loss, metric
