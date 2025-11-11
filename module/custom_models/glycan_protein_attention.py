from enum import IntEnum
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core
from torchdrug.core import Registry as R
from module.custom_data.glycan import Glycan


ACTIVATIONS = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "ELU": nn.ELU,
    "LeakyReLU": nn.LeakyReLU,
}


class CrossAttention(nn.Module):
    """Multi-head cross-attention used to fuse glycan and protein tokens."""

    def __init__(self, q_dim: int, kv_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        inner = heads * dim_head
        self.to_q = nn.Linear(q_dim, inner, bias=False)
        self.to_k = nn.Linear(kv_dim, inner, bias=False)
        self.to_v = nn.Linear(kv_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, q_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(q_dim)

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        q_mask: Optional[torch.Tensor],
        kv_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            q_tokens: Query tokens of shape ``(B, Lq, Dq)``.
            kv_tokens: Key / value tokens of shape ``(B, Lkv, Dkv)``.
            q_mask: Boolean mask for queries with shape ``(B, Lq)`` where ``True`` means the token is valid.
            kv_mask: Boolean mask for keys with shape ``(B, Lkv)``.
        """

        B, Lq, _ = q_tokens.shape
        Lkv = kv_tokens.size(1)
        h = self.heads
        dh = self.to_q.out_features // h

        q = self.to_q(q_tokens).view(B, Lq, h, dh).transpose(1, 2)
        k = self.to_k(kv_tokens).view(B, Lkv, h, dh).transpose(1, 2)
        v = self.to_v(kv_tokens).view(B, Lkv, h, dh).transpose(1, 2)

        if q_mask is not None and kv_mask is not None:
            allow = q_mask[:, None, :, None] & kv_mask[:, None, None, :]
            attn_mask = ~allow.expand(B, h, Lq, Lkv)
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, Lq, h * dh)
        out = self.ln(self.dropout(self.to_out(out)))
        return out + q_tokens


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    num_hidden_layers: int,
    activation: nn.Module,
    dropout: float,
) -> nn.Sequential:
    layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), activation(), nn.Dropout(dropout)]
    for _ in range(int(num_hidden_layers)):
        layers.extend(
            [
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                activation(),
                nn.Dropout(dropout),
            ]
        )
    seq = nn.Sequential(*layers)
    for module in seq.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
    return seq


class TrainingStage(IntEnum):
    STAGE0 = 0
    STAGE1 = 1


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for param in module.parameters():
        param.requires_grad = flag


def freeze(module: nn.Module) -> None:
    module.eval()
    _set_requires_grad(module, False)


def unfreeze(module: nn.Module) -> None:
    module.train()
    _set_requires_grad(module, True)


class ProteinEmbeddingEncoder(nn.Module):
    """Simple encoder that looks up cached protein embeddings."""

    def __init__(self, cache_path: Optional[str] = None, device: Optional[torch.device] = None):
        super().__init__()
        self.device_type = device or torch.device("cpu")
        self.cache_path = cache_path
        self.inmem_protein: Dict[str, torch.Tensor] = {}
        if cache_path is not None:
            self.load_cache(cache_path)

    def load_cache(self, path: Optional[str] = None) -> None:
        cache_path = path or self.cache_path
        if cache_path is None:
            raise ValueError("Protein cache path must be provided before loading embeddings.")
        data = torch.load(cache_path, map_location="cpu")
        embeddings = data.get("embeddings", data)
        self.inmem_protein = {seq: tensor.to(self.device_type) for seq, tensor in embeddings.items()}
        self.cache_path = cache_path

    def _normalize_sequences(self, proteins) -> List[str]:
        if isinstance(proteins, (list, tuple)):
            return [self._extract_sequence(p) for p in proteins]
        if hasattr(proteins, "to_sequence"):
            seqs = proteins.to_sequence()
            if isinstance(seqs, str):
                return [seqs]
            if isinstance(seqs, Sequence):
                return [self._ensure_string(seq) for seq in seqs]
        if hasattr(proteins, "sequence"):
            seqs = proteins.sequence
            if isinstance(seqs, str):
                return [seqs]
            if isinstance(seqs, Sequence):
                return [self._ensure_string(seq) for seq in seqs]
        raise ValueError("Unsupported protein representation for cached embeddings.")

    def _ensure_string(self, sequence) -> str:
        if isinstance(sequence, str):
            return sequence
        if isinstance(sequence, (list, tuple)):
            return "".join(sequence)
        if torch.is_tensor(sequence):
            raise ValueError(
                "Cannot infer amino acid sequence from tensor representation. Please provide string sequences."
            )
        raise ValueError("Cannot convert protein sequence representation to string.")

    def _extract_sequence(self, protein) -> str:
        if hasattr(protein, "to_sequence"):
            seq = protein.to_sequence()
            if isinstance(seq, str):
                return seq
            if isinstance(seq, Sequence):
                return self._ensure_string(seq)
        if hasattr(protein, "sequence"):
            seq = protein.sequence
            return self._ensure_string(seq)
        if isinstance(protein, str):
            return protein
        raise ValueError("Protein object does not expose a sequence string.")

    def forward(self, proteins) -> Tuple[torch.Tensor, torch.Tensor]:
        sequences = self._normalize_sequences(proteins)
        if not sequences:
            raise ValueError("No protein sequences provided for encoding.")
        if not self.inmem_protein:
            raise RuntimeError("Protein embedding cache has not been loaded.")

        embeddings: List[torch.Tensor] = []
        lengths: List[int] = []
        for sequence in sequences:
            if sequence not in self.inmem_protein:
                raise KeyError(f"Sequence `{sequence}` not found in protein embedding cache.")
            tensor = self.inmem_protein[sequence]
            embeddings.append(tensor)
            lengths.append(int(tensor.size(0)))

        batch_size = len(embeddings)
        max_len = max(lengths)
        dim = embeddings[0].size(-1)
        padded = embeddings[0].new_zeros(batch_size, max_len, dim)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=padded.device)
        for idx, tensor in enumerate(embeddings):
            length = lengths[idx]
            padded[idx, :length] = tensor
            mask[idx, :length] = True
        return padded, mask


class GlycanTokenEncoder(nn.Module):
    """Wrapper around a Glycan encoder that returns padded monosaccharide features."""

    def __init__(self, glycan_model: nn.Module, device: torch.device):
        super().__init__()
        self.model = glycan_model
        self.device_type = device

    def _infer_batch_size(self, graph, mono_graph_ids: torch.Tensor) -> int:
        for attr in ("batch_size", "num_graphs", "num_graph"):
            if hasattr(graph, attr):
                value = getattr(graph, attr)
                if isinstance(value, int):
                    return value
                if torch.is_tensor(value):
                    return int(value.item())
        if mono_graph_ids.numel() > 0:
            return int(mono_graph_ids.max().item()) + 1
        raise ValueError("Unable to infer batch size from glycan graph.")

    def forward(self, graph) -> Tuple[torch.Tensor, torch.Tensor]:
        input_feature = getattr(graph, "node_feature", None)
        if torch.is_tensor(input_feature):
            input_feature = input_feature.float()
        elif input_feature is not None:
            input_feature = torch.as_tensor(input_feature, device=self.device_type).float()

        output = self.model(graph, input_feature)
        if "all_node_feature" not in output:
            raise ValueError("Glycan encoder output must contain `all_node_feature`.")
        all_node_feature = output["all_node_feature"]
        if not hasattr(graph, "unit_type"):
            raise ValueError("Glycan graph must contain `unit_type` information.")

        mono_mask = graph.unit_type < len(Glycan.units)
        mono_features = all_node_feature[mono_mask]
        node2graph = getattr(graph, "node2graph", None)
        if node2graph is None and hasattr(graph, "unit2graph"):
            node2graph = graph.unit2graph
        if node2graph is None:
            raise ValueError("Packed glycan graph must expose `node2graph` or `unit2graph` mapping.")
        mono_graph_ids = node2graph[mono_mask]

        batch_size = self._infer_batch_size(graph, mono_graph_ids)
        lengths = torch.bincount(mono_graph_ids, minlength=batch_size)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        dim = mono_features.size(-1)
        padded = mono_features.new_zeros(batch_size, max_len, dim)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=mono_features.device)

        cursor = 0
        for idx, length in enumerate(lengths.tolist()):
            if length == 0:
                continue
            end = cursor + length
            padded[idx, :length] = mono_features[cursor:end]
            mask[idx, :length] = True
            cursor = end
        return padded, mask


@R.register("models.GlycanProteinAttention")
class GlycanProteinAttention(nn.Module, core.Configurable):
    """Cross-attention fusion between protein and glycan representations."""

    def __init__(
        self,
        glycan_model: nn.Module,
        protein_cache_path: Optional[str] = None,
        glycan_dim: Optional[int] = None,
        protein_dim: Optional[int] = None,
        latent_activation: str = "GELU",
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
        coattention: bool = False,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        self.glycan_model = glycan_model
        self.model = glycan_model
        self.latent_activation = ACTIVATIONS[latent_activation]
        self.num_hidden_layers = int(num_hidden_layers)
        self.dropout = float(dropout)
        self.coattention = coattention

        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = torch.float16

        self.glycan_dim = glycan_dim or getattr(glycan_model, "output_dim", None)
        if self.glycan_dim is None:
            raise ValueError("`glycan_dim` must be provided if the glycan model does not expose `output_dim`.")

        self.glycan_encoder = GlycanTokenEncoder(glycan_model, self.device_type)
        self.protein_encoder = ProteinEmbeddingEncoder(protein_cache_path, self.device_type)
        inferred_dim = None
        if self.protein_encoder.inmem_protein:
            inferred_dim = next(iter(self.protein_encoder.inmem_protein.values())).size(-1)
        self.protein_dim = protein_dim or inferred_dim
        if self.protein_dim is None:
            raise ValueError("`protein_dim` must be provided when protein cache does not define embeddings.")

        self.g2p = CrossAttention(self.glycan_dim, self.protein_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        if self.coattention:
            self.p2g = CrossAttention(self.protein_dim, self.glycan_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            fusion_in_dim = self.glycan_dim + self.protein_dim
        else:
            self.p2g = None
            fusion_in_dim = self.glycan_dim
        self.fusion_block = build_mlp(fusion_in_dim, 2 * self.glycan_dim, self.num_hidden_layers, self.latent_activation, self.dropout)
        self.regression_head = nn.Linear(2 * self.glycan_dim, 1)

        self.training_stage: TrainingStage = TrainingStage.STAGE0
        self.apply_stage_freeze()

    def load_protein_cache(self, cache_path: str) -> None:
        self.protein_encoder.load_cache(cache_path)
        if self.protein_dim is None:
            first = next(iter(self.protein_encoder.inmem_protein.values()))
            self.protein_dim = int(first.size(-1))

    def set_stage(self, stage: int) -> None:
        if stage not in (0, 1):
            raise ValueError("Only Stage 0 and Stage 1 are implemented.")
        self.training_stage = TrainingStage(stage)
        self.apply_stage_freeze()

    def apply_stage_freeze(self) -> None:
        if self.training_stage == TrainingStage.STAGE0:
            freeze(self.glycan_model)
        elif self.training_stage == TrainingStage.STAGE1:
            unfreeze(self.glycan_model)

    def forward(self, protein_graph, glycan_graph) -> torch.Tensor:
        with torch.autocast(
            device_type=("cuda" if self.device_type.type == "cuda" else "cpu"),
            dtype=self.amp_dtype,
            enabled=(self.device_type.type == "cuda"),
        ):
            p_pad, p_mask = self.protein_encoder(protein_graph)
        with torch.autocast(
            device_type=("cuda" if self.device_type.type == "cuda" else "cpu"),
            enabled=False,
        ):
            g_pad, g_mask = self.glycan_encoder(glycan_graph)
        with torch.autocast(
            device_type=("cuda" if self.device_type.type == "cuda" else "cpu"),
            dtype=self.amp_dtype,
            enabled=(self.device_type.type == "cuda"),
        ):
            g_att = self.g2p(g_pad, p_pad, g_mask, p_mask)
            g_vec = (g_att * g_mask.unsqueeze(-1)).sum(1) / g_mask.sum(1, keepdim=True).clamp_min(1)

            if self.coattention and self.p2g is not None:
                p_att = self.p2g(p_pad, g_pad, p_mask, g_mask)
                p_vec = (p_att * p_mask.unsqueeze(-1)).sum(1) / p_mask.sum(1, keepdim=True).clamp_min(1)
                fusion_input = torch.cat([g_vec, p_vec], dim=-1)
            else:
                fusion_input = g_vec

            fused = self.fusion_block(fusion_input)
            out = self.regression_head(fused)
            return out.squeeze(-1)
