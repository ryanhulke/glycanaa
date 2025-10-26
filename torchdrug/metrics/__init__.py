import torch

__all__ = [
    "accuracy",
    "matthews_corrcoef",
    "area_under_roc",
    "area_under_prc",
    "r2",
    "spearmanr",
    "pearsonr",
]


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    return torch.as_tensor(tensor)


def _binary_predictions(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 1 or pred.shape[-1] == 1:
        return (pred.view(-1) >= 0).long()
    return pred.argmax(dim=-1)


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_label = _binary_predictions(pred)
    correct = (pred_label == target).float()
    return correct.mean()


def matthews_corrcoef(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_label = _binary_predictions(pred)
    tp = ((pred_label == 1) & (target == 1)).sum().float()
    tn = ((pred_label == 0) & (target == 0)).sum().float()
    fp = ((pred_label == 1) & (target == 0)).sum().float()
    fn = ((pred_label == 0) & (target == 1)).sum().float()
    numerator = tp * tn - fp * fn
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
    return numerator / denominator.clamp(min=1e-12)


def area_under_roc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim > 1 and pred.shape[-1] > 1:
        pred = pred[..., 1]
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    sorted_indices = torch.argsort(pred, descending=True)
    target = target[sorted_indices]
    positives = target.sum()
    negatives = len(target) - positives
    if positives == 0 or negatives == 0:
        return torch.tensor(0.0)
    tpr = torch.cumsum(target, dim=0) / positives
    fpr = torch.cumsum(1 - target, dim=0) / negatives
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])
    auc = torch.trapz(tpr, fpr)
    return auc


def area_under_prc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.ndim > 1 and pred.shape[-1] > 1:
        pred = pred[..., 1]
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    sorted_indices = torch.argsort(pred, descending=True)
    target = target[sorted_indices]
    cum_tp = torch.cumsum(target, dim=0)
    denom = torch.arange(1, len(target) + 1, dtype=torch.float32)
    precision = cum_tp / denom
    recall = cum_tp / target.sum().clamp(min=1.0)
    precision = torch.cat([torch.tensor([1.0]), precision])
    recall = torch.cat([torch.tensor([0.0]), recall])
    auc = torch.trapz(precision, recall)
    return auc


def r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    return 1 - ss_res / ss_tot.clamp(min=1e-12)


def spearmanr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rank = torch.argsort(torch.argsort(pred))
    target_rank = torch.argsort(torch.argsort(target))
    pred_rank = pred_rank.float()
    target_rank = target_rank.float()
    cov = ((pred_rank - pred_rank.mean()) * (target_rank - target_rank.mean())).mean()
    std = pred_rank.std() * target_rank.std() + 1e-12
    return cov / std


def pearsonr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_mean = pred.mean()
    target_mean = target.mean()
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    return cov / (pred.std() * target.std() + 1e-12)
