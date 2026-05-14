"""Shared fixtures for the critnet test suite.

The fixtures intentionally avoid HuggingFace model downloads: a minimal
``nn.Module`` mimicking LLaMA module names (``q_proj`` / ``k_proj`` /
``v_proj`` / ``o_proj`` / ``gate_proj`` / ``up_proj`` / ``down_proj`` +
``input_layernorm``) is enough to exercise every code path in the
toolkit.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class _RMSNorm(nn.Module):
    """Minimal RMSNorm with a 1-D weight (class name is matched by ``_is_norm``)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return x / rms * self.weight


class TinyAttn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x); k = self.k_proj(x); v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (x.size(-1) ** 0.5), dim=-1)
        return self.o_proj(attn @ v)


class TinyMLP(nn.Module):
    def __init__(self, hidden: int, mlp: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, mlp, bias=False)
        self.up_proj = nn.Linear(hidden, mlp, bias=False)
        self.down_proj = nn.Linear(mlp, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(nn.Module):
    def __init__(self, hidden: int, mlp: int) -> None:
        super().__init__()
        self.input_layernorm = _RMSNorm(hidden)
        self.self_attn = TinyAttn(hidden)
        self.post_attention_layernorm = _RMSNorm(hidden)
        self.mlp = TinyMLP(hidden, mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLM(nn.Module):
    """Synthetic LLaMA-shaped LM used in every test.

    Two hidden=8 / mlp=12 / vocab=16 layers.  Total scalar parameters
    are well under 10 K so the entire suite runs in milliseconds.
    """

    HIDDEN, MLP, VOCAB, N_LAYERS = 8, 12, 16, 2

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(self.VOCAB, self.HIDDEN)
        self.layers = nn.ModuleList(
            [TinyLayer(self.HIDDEN, self.MLP) for _ in range(self.N_LAYERS)]
        )
        self.norm = _RMSNorm(self.HIDDEN)
        self.lm_head = nn.Linear(self.HIDDEN, self.VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.VOCAB)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=-100
            )

        class _Out:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        return _Out(loss, logits)


@pytest.fixture
def tiny_lm() -> TinyLM:
    torch.manual_seed(0)
    return TinyLM()


@pytest.fixture
def tiny_loader(tiny_lm: TinyLM):
    """Three-batch deterministic loader of (input_ids, attention_mask, labels)."""
    torch.manual_seed(1)
    batches = []
    for _ in range(3):
        ids = torch.randint(0, tiny_lm.VOCAB, (2, 6))
        batches.append({
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        })
    return batches
