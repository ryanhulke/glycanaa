from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch


class _AttributeContext:
    def __init__(self, graph: "Graph", name: str):
        self.graph = graph
        self.name = name

    def __enter__(self) -> "Graph":
        self.graph._active_namespace = self.name
        return self.graph

    def __exit__(self, exc_type, exc, tb) -> None:
        self.graph._active_namespace = None


class Graph:
    def __init__(
        self,
        edge_list: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        num_nodes: Optional[int] = None,
        num_edges: Optional[int] = None,
        offsets: Optional[Sequence[int]] = None,
        num_relation: Optional[int] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        self.device = edge_list.device if isinstance(edge_list, torch.Tensor) else torch.device("cpu")
        self._active_namespace: Optional[str] = None
        self.meta_dict = meta_dict or {}
        self.data_dict: Dict[str, Any] = {}

        self._edge_list = torch.as_tensor(edge_list, dtype=torch.long, device=self.device)
        self._edge_weight = (
            torch.as_tensor(edge_weight, dtype=torch.float32, device=self.device)
            if edge_weight is not None
            else torch.ones(self._edge_list.shape[0], 1, device=self.device)
        )
        self.num_relation = num_relation or int(self._edge_list[:, -1].max().item() + 1) if self._edge_list.numel() > 0 else 0

        if num_nodes is None:
            self.num_node = int(self._edge_list[:, :2].max().item() + 1) if self._edge_list.numel() else 0
        else:
            self.num_node = int(num_nodes)
        self.num_edge = int(num_edges) if num_edges is not None else int(self._edge_list.shape[0])

        self.batch_size = 1
        self._num_nodes = torch.tensor([self.num_node], device=self.device)
        self._num_edges = torch.tensor([self.num_edge], device=self.device)
        self.node2graph = torch.zeros(self.num_node, dtype=torch.long, device=self.device)
        self.edge2graph = torch.zeros(self.num_edge, dtype=torch.long, device=self.device)
        self.cumsums = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    def clone(self) -> "Graph":
        new_graph = copy.copy(self)
        new_graph._edge_list = self._edge_list.clone()
        new_graph._edge_weight = self._edge_weight.clone()
        new_graph.meta_dict = copy.deepcopy(self.meta_dict)
        new_graph.data_dict = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v) for k, v in self.data_dict.items()}
        return new_graph

    def to(self, device):
        self._edge_list = self._edge_list.to(device)
        self._edge_weight = self._edge_weight.to(device)
        self._num_nodes = self._num_nodes.to(device)
        self._num_edges = self._num_edges.to(device)
        self.node2graph = self.node2graph.to(device)
        self.edge2graph = self.edge2graph.to(device)
        if self.cumsums is not None:
            self.cumsums = [c.to(device) if isinstance(c, torch.Tensor) else c for c in self.cumsums]
        new_data = {}
        for key, value in self.data_dict.items():
            if isinstance(value, torch.Tensor):
                moved = value.to(device)
                object.__setattr__(self, key, moved)
                new_data[key] = moved
            else:
                new_data[key] = value
        self.data_dict = new_data
        self.device = device
        return self

    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self, device: Optional[int] = None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if device is None:
            device = torch.cuda.current_device()
        return self.to(torch.device(f"cuda:{device}"))

    # Namespace helpers -------------------------------------------------
    def unit(self) -> _AttributeContext:
        return _AttributeContext(self, "unit")

    def link(self) -> _AttributeContext:
        return _AttributeContext(self, "link")

    def glycan(self) -> _AttributeContext:
        return _AttributeContext(self, "glycan")

    def glycoword(self) -> _AttributeContext:
        return _AttributeContext(self, "glycoword")

    @classmethod
    def pack(cls, graphs: List["Graph"]):
        if not graphs:
            raise ValueError("Cannot pack an empty list of graphs")
        if hasattr(cls, "packed_type"):
            total_edge = sum(g.edge_list.shape[0] for g in graphs)
            if total_edge:
                edge_list = torch.cat([g.edge_list for g in graphs], dim=0)
                edge_weight = torch.cat([g.edge_weight for g in graphs], dim=0)
            else:
                device = graphs[0].edge_list.device
                edge_list = torch.zeros((0, graphs[0].edge_list.shape[1]), dtype=torch.long, device=device)
                edge_weight = torch.zeros((0,) + graphs[0].edge_weight.shape[1:], device=device)
            num_nodes = [g.num_node for g in graphs]
            num_edges = [g.num_edge for g in graphs]
            return cls.packed_type(edge_list, edge_weight=edge_weight, num_nodes=num_nodes, num_edges=num_edges)
        return graphs

    # Properties --------------------------------------------------------
    @property
    def edge_list(self) -> torch.Tensor:
        return self._edge_list

    @property
    def edge_weight(self) -> torch.Tensor:
        return self._edge_weight

    @property
    def num_nodes(self) -> torch.Tensor:
        return self._num_nodes

    @property
    def num_edges(self) -> torch.Tensor:
        return self._num_edges

    @property
    def num_cum_nodes(self) -> torch.Tensor:
        return self._num_nodes.cumsum(0)

    @property
    def num_cum_edges(self) -> torch.Tensor:
        return self._num_edges.cumsum(0)

    # Data attribute helpers --------------------------------------------
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name in {"device", "meta_dict", "data_dict", "_edge_list", "_edge_weight", "num_relation", "num_node", "num_edge", "batch_size", "_num_nodes", "_num_edges", "node2graph", "edge2graph", "_active_namespace", "cumsums"}:
            return
        if isinstance(value, torch.Tensor):
            self.data_dict[name] = value
            namespace = self._active_namespace
            if namespace is not None:
                meta_value = {
                    "unit": "node",
                    "link": "edge",
                    "glycan": "graph",
                    "glycoword": "glycoword",
                }.get(namespace)
                if meta_value is not None:
                    self.meta_dict.setdefault(name, [])
                    if meta_value not in self.meta_dict[name]:
                        self.meta_dict[name].append(meta_value)

    def node_mask(self, mask: torch.Tensor, compact: bool = False) -> "Graph":
        if mask.dtype != torch.bool:
            mask = mask.bool()
        node_index = torch.nonzero(mask, as_tuple=False).view(-1)
        sub_edge_mask = mask[self._edge_list[:, 0]] & mask[self._edge_list[:, 1]]
        sub_edge_list = self._edge_list[sub_edge_mask]
        if compact:
            remap = torch.full((self.num_node,), -1, dtype=torch.long, device=self.device)
            remap[node_index] = torch.arange(node_index.numel(), device=self.device)
            sub_edge_list = remap[sub_edge_list]
        subgraph = Graph(sub_edge_list, num_nodes=node_index.numel(), num_edges=sub_edge_list.shape[0], num_relation=self.num_relation)
        for key, value in self.data_dict.items():
            if value.dim() > 0 and value.size(0) == self.num_node:
                subgraph.__dict__[key] = value[node_index]
                subgraph.data_dict[key] = subgraph.__dict__[key]
        return subgraph

    def line_graph(self) -> "Graph":
        if self._edge_list.numel() == 0:
            return Graph(torch.zeros((0, 3), dtype=torch.long))
        node_in = self._edge_list[:, 0]
        node_out = self._edge_list[:, 1]
        num_edges = self._edge_list.shape[0]
        adjacency = []
        relation = []
        for i in range(num_edges):
            tail = node_out[i]
            candidates = (node_in == tail).nonzero(as_tuple=False).view(-1)
            for j in candidates:
                adjacency.append([i, int(j.item())])
                relation.append(int(self._edge_list[j, -1].item()))
        if adjacency:
            edge_tensor = torch.tensor(adjacency, dtype=torch.long, device=self.device)
            relation = torch.tensor(relation, dtype=torch.long, device=self.device)
            edge_tensor = torch.cat([edge_tensor, relation.unsqueeze(-1)], dim=-1)
        else:
            edge_tensor = torch.zeros((0, 3), dtype=torch.long, device=self.device)
        return Graph(edge_tensor, num_nodes=num_edges, num_relation=self.num_relation)


class PackedGraph(Graph):
    def __init__(
        self,
        edge_list: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        num_nodes: Optional[Sequence[int]] = None,
        num_edges: Optional[Sequence[int]] = None,
        offsets: Optional[Sequence[int]] = None,
        meta_dict: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(edge_list, edge_weight=edge_weight, num_nodes=sum(num_nodes) if num_nodes else None, num_edges=sum(num_edges) if num_edges else None, meta_dict=meta_dict, **kwargs)
        if num_nodes is None:
            raise ValueError("PackedGraph requires `num_nodes`")
        self.batch_size = len(num_nodes)
        self._num_nodes = torch.as_tensor(num_nodes, dtype=torch.long, device=self.device)
        if num_edges is None:
            num_edges = [edge_list.shape[0]] * self.batch_size
        self._num_edges = torch.as_tensor(num_edges, dtype=torch.long, device=self.device)
        indices = torch.arange(self.batch_size, device=self.device)
        self.node2graph = torch.repeat_interleave(indices, self._num_nodes)
        self.edge2graph = torch.repeat_interleave(indices, self._num_edges)
        self.cumsums = kwargs.get("cumsums")


class Molecule(Graph):
    pass


class PackedMolecule(PackedGraph):
    pass


class constant:
    ATOM_SYMBOL = ["Null"]


Graph.packed_type = PackedGraph
Molecule.packed_type = PackedMolecule
