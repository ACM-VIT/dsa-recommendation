"""
DSA Engine -- RGCN Step 2: Model
================================
A compact heterogeneous R-GCN implemented in pure PyTorch -- no
torch-geometric dependency. Built for the Knode problem/concept graph:

    node types : problem, concept
    relations  : uses, used_by, similar, cooccurs (each gets its own weight matrix)

Message passing per layer, for every destination node type t:

    h'_t = SelfLoop(h_t) + Σ_{relations r into t}  Â_r · (h_src · W_r)

where Â_r is the (optionally weight-scaled) degree-normalised adjacency of
relation r. LayerNorm + residual + ReLU + dropout wrap each layer.

This is the standard R-GCN formulation (Schlichtkrull et al., 2018); with only
four relations we skip basis decomposition -- one dense matrix per relation is
cheap and more expressive.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Normalised sparse adjacency cache
# ---------------------------------------------------------------------------

def build_norm_adj(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None,
    num_src: int,
    num_dst: int,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Symmetric-ish row-normalised sparse adjacency  (dst <- src):
        Â[d, s] = w(s,d) / deg(d)
    Returned as a torch.sparse_coo_tensor of shape (num_dst, num_src).
    None if the relation has no edges.
    """
    if edge_index.numel() == 0:
        return None
    src, dst = edge_index[0], edge_index[1]
    w = edge_weight if edge_weight is not None else torch.ones(src.shape[0])
    w = w.to(device).float()
    src = src.to(device); dst = dst.to(device)

    # destination in-degree (weighted) for normalisation
    deg = torch.zeros(num_dst, device=device).index_add_(0, dst, w)
    deg = torch.where(deg == 0, torch.ones_like(deg), deg)
    norm_w = w / deg[dst]

    idx = torch.stack([dst, src])  # rows = dst, cols = src
    adj = torch.sparse_coo_tensor(idx, norm_w, (num_dst, num_src)).coalesce()
    return adj


# ---------------------------------------------------------------------------
# One R-GCN layer over the whole hetero graph
# ---------------------------------------------------------------------------

class RGCNLayer(nn.Module):
    def __init__(self, in_dims: Dict[str, int], out_dim: int,
                 relations: Dict[str, Tuple[str, str]]):
        super().__init__()
        self.relations = relations          # rel -> (src_type, dst_type)
        self.node_types = list(in_dims)
        # self-loop weight per node type
        self.self_w = nn.ModuleDict(
            {t: nn.Linear(in_dims[t], out_dim, bias=True) for t in in_dims}
        )
        # one relation weight matrix per relation
        self.rel_w = nn.ModuleDict(
            {rel: nn.Linear(in_dims[src], out_dim, bias=False)
             for rel, (src, _dst) in relations.items()}
        )

    def forward(self, h: Dict[str, torch.Tensor],
                adj: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # self-loop contribution
        out = {t: self.self_w[t](h[t]) for t in self.node_types}
        # relational messages
        for rel, (src, dst) in self.relations.items():
            a = adj.get(rel)
            if a is None:
                continue
            msg = self.rel_w[rel](h[src])      # (num_src, out)
            out[dst] = out[dst] + torch.sparse.mm(a, msg)
        return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HeteroRGCN(nn.Module):
    def __init__(
        self,
        in_dims: Dict[str, int],
        relations: Dict[str, Tuple[str, str]],
        hidden_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.node_types = list(in_dims)
        self.relations = relations
        self.dropout = dropout

        # input projection per node type -> common hidden space
        self.input_proj = nn.ModuleDict(
            {t: nn.Linear(in_dims[t], hidden_dim) for t in in_dims}
        )
        hid = {t: hidden_dim for t in in_dims}

        self.layer1 = RGCNLayer(hid, hidden_dim, relations)
        self.norm1  = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in in_dims})

        self.layer2 = RGCNLayer({t: hidden_dim for t in in_dims}, out_dim, relations)
        self.norm2  = nn.ModuleDict({t: nn.LayerNorm(out_dim) for t in in_dims})

        # residual projection when hidden_dim != out_dim
        self.res2 = nn.ModuleDict(
            {t: (nn.Linear(hidden_dim, out_dim) if hidden_dim != out_dim
                 else nn.Identity()) for t in in_dims}
        )

        # learnable temperature for cosine link-prediction logits (init ~10x),
        # keeps BCE well-conditioned regardless of embedding magnitude
        self.logit_scale = nn.Parameter(torch.tensor(2.3026))  # ln(10)

    def link_logits(self, zp: torch.Tensor, zc: torch.Tensor,
                    p_idx: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled cosine score between problem and concept nodes."""
        zp = F.normalize(zp, dim=-1)
        zc = F.normalize(zc, dim=-1)
        scale = self.logit_scale.clamp(max=4.6052).exp()  # cap at 100x
        return scale * (zp[p_idx] * zc[c_idx]).sum(-1)

    def forward(self, features: Dict[str, torch.Tensor],
                adj: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = {t: F.relu(self.input_proj[t](features[t])) for t in self.node_types}

        # layer 1 (residual: hidden->hidden, dims match)
        h1 = self.layer1(h, adj)
        h1 = {t: self.norm1[t](h1[t]) + h[t] for t in self.node_types}
        h1 = {t: F.dropout(F.relu(v), p=self.dropout, training=self.training)
              for t, v in h1.items()}

        # layer 2 (residual through projection if needed)
        h2 = self.layer2(h1, adj)
        h2 = {t: self.norm2[t](h2[t]) + self.res2[t](h1[t])
              for t in self.node_types}
        return h2


# ---------------------------------------------------------------------------
# Adjacency precompute from a saved graph dict
# ---------------------------------------------------------------------------

def precompute_adj(graph: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    adj = {}
    for rel, (st, dt, ei, ew) in graph["relations"].items():
        adj[rel] = build_norm_adj(
            ei, ew, graph["num_nodes"][st], graph["num_nodes"][dt], device
        )
    return adj


def relation_types(graph: dict) -> Dict[str, Tuple[str, str]]:
    return {rel: (st, dt) for rel, (st, dt, _ei, _ew) in graph["relations"].items()}


# ---------------------------------------------------------------------------
# Multi-relation link decoder (DistMult with a learned diagonal per relation)
# ---------------------------------------------------------------------------

class LinkDecoder(nn.Module):
    """
    score(src, rel, dst) = temp * < norm(z_src) ⊙ diag_rel , norm(z_dst) >

    DistMult generalises the plain dot product: each scored relation gets its
    own diagonal, so `uses` (problem-concept) and `cooccurs` (concept-concept)
    are decoded differently while sharing one encoder.
    """

    def __init__(self, dim: int, scored_relations: list[str]):
        super().__init__()
        self.rel = nn.ParameterDict(
            {r: nn.Parameter(torch.ones(dim)) for r in scored_relations}
        )
        self.logit_scale = nn.Parameter(torch.tensor(2.3026))  # ln(10)

    def forward(self, z_src, z_dst, src_idx, dst_idx, rel: str) -> torch.Tensor:
        zs = F.normalize(z_src[src_idx], dim=-1)
        zd = F.normalize(z_dst[dst_idx], dim=-1)
        scale = self.logit_scale.clamp(max=4.6052).exp()        # cap at 100x
        return scale * (zs * self.rel[rel] * zd).sum(-1)
