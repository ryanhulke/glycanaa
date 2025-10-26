from typing import Callable, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_max, scatter_mean

from . import functional

__all__ = [
    "MLP",
    "SinusoidalPositionEmbedding",
    "MessagePassingBase",
    "RelationalGraphConv",
    "GeometricRelationalGraphConv",
    "ProteinResNetBlock",
    "ProteinBERTBlock",
    "SumReadout",
    "MeanReadout",
    "MaxReadout",
    "functional",
]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.activation = getattr(F, activation)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.output_dim = dims[-1]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = input
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i < len(self.layers) - 1:
                hidden = self.activation(hidden)
                if self.dropout:
                    hidden = self.dropout(hidden)
        return hidden


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        device = positions.device
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=device, dtype=torch.float32)
        emb = positions.unsqueeze(-1) / (10000 ** (emb / half_dim))
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class MessagePassingBase(nn.Module):
    def __init__(self):
        super().__init__()

    def message(self, graph, input):
        raise NotImplementedError

    def aggregate(self, message, graph):
        node_out = graph.edge_list[:, 1]
        return scatter_add(message, node_out, dim=0, dim_size=graph.num_node)

    def update(self, node_feature, input):
        return node_feature

    def forward(self, graph, input):
        message = self.message(graph, input)
        node_feature = self.aggregate(message, graph)
        return self.update(node_feature, input)


class RelationalGraphConv(MessagePassingBase):
    def __init__(self, input_dim: int, output_dim: int, num_relation: int, batch_norm: bool = False, activation: str = "relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.node_linear = nn.Linear(input_dim, output_dim)
        self.rel_emb = nn.Embedding(num_relation, output_dim)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else None
        self.activation = getattr(F, activation)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 0]
        relation = graph.edge_list[:, 2]
        message = self.node_linear(input[node_in]) + self.rel_emb(relation)
        if hasattr(graph, "edge_feature") and graph.edge_feature is not None:
            message = message + graph.edge_feature.float()
        weight = graph.edge_weight
        if weight.dim() == 1:
            weight = weight.unsqueeze(-1)
        message = message * weight
        return message

    def aggregate(self, message, graph):
        node_out = graph.edge_list[:, 1]
        update = scatter_add(message, node_out, dim=0, dim_size=graph.num_node)
        degree = scatter_add(graph.edge_weight, node_out, dim=0, dim_size=graph.num_node) + 1e-6
        if degree.dim() == 1:
            degree = degree.unsqueeze(-1)
        update = update / degree
        if self.batch_norm is not None:
            update = self.batch_norm(update)
        return self.activation(update)

    def forward(self, graph, input):
        return super().forward(graph, input)


class GeometricRelationalGraphConv(RelationalGraphConv):
    pass


class ProteinResNetBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, stride: int, padding: int, activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, stride=1, padding=padding)
        self.shortcut = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None
        self.activation = getattr(F, activation)
        self.norm1 = nn.BatchNorm1d(output_dim)
        self.norm2 = nn.BatchNorm1d(output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(input.transpose(1, 2))
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        if self.shortcut is not None:
            residual = self.shortcut(input.transpose(1, 2))
        else:
            residual = input.transpose(1, 2)
        hidden = self.activation(hidden + residual)
        return hidden.transpose(1, 2)


class ProteinBERTBlock(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask = ~mask
        return self.encoder(input, src_key_padding_mask=mask)


class SumReadout(nn.Module):
    def forward(self, graph, input):
        index = graph.node2graph
        return scatter_add(input, index, dim=0, dim_size=graph.batch_size)


class MeanReadout(nn.Module):
    def forward(self, graph, input):
        index = graph.node2graph
        return scatter_mean(input, index, dim=0, dim_size=graph.batch_size)


class MaxReadout(nn.Module):
    def forward(self, graph, input):
        index = graph.node2graph
        return scatter_max(input, index, dim=0, dim_size=graph.batch_size)[0]
