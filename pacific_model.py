"""
Pacific-mini: a ~300M parameter decoder-only transformer, Llama-style.

Architecture choices (see design notes at bottom of file):
- RoPE positional encoding (no learned/absolute positions)
- RMSNorm, pre-norm placement
- Grouped-Query Attention (GQA) — fewer KV heads than query heads
- SwiGLU feed-forward
- Tied input/output embeddings

This file only defines the model. Training loop, data pipeline, and
tokenizer are separate pieces (build those next).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PacificConfig:
    vocab_size: int = 32000
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16          # query heads
    n_kv_heads: int = 4        # GQA: fewer kv heads, saves KV-cache memory
    ffn_hidden: int = 2816     # SwiGLU hidden dim (~2.67x d_model, not 4x)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0       # pretraining typically uses 0
    tie_embeddings: bool = True


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Root-mean-square layer norm. Cheaper than LayerNorm (no mean-centering,
    no bias), and empirically works as well or better for LLM pretraining."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(dtype)


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Precompute the rotation frequencies used by RoPE."""
    assert head_dim % 2 == 0, "RoPE requires an even head_dim"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, freqs)  # (max_seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to a (batch, seq_len, n_heads, head_dim) tensor."""
    b, s, h, d = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(b, s, h, d // 2, 2))
    freqs = rope_freqs[:s].view(1, s, 1, d // 2)
    x_rotated = x_complex * freqs
    x_out = torch.view_as_real(x_rotated).reshape(b, s, h, d)
    return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------
class GQAAttention(nn.Module):
    """Multi-head attention where key/value heads are shared across groups
    of query heads. n_heads must be divisible by n_kv_heads.

    With n_heads=16, n_kv_heads=4: every 4 query heads share 1 kv head.
    """

    def __init__(self, cfg: PacificConfig):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.wq(x).view(b, s, self.n_heads, self.head_dim)
        k = self.wk(x).view(b, s, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(b, s, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------
class SwiGLU(nn.Module):
    """SwiGLU FFN: gate(x) * up(x), then down-projected."""

    def __init__(self, cfg: PacificConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg: PacificConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_freqs)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class PacificModel(nn.Module):
    def __init__(self, cfg: PacificConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        head_dim = cfg.d_model // cfg.n_heads
        rope_freqs = precompute_rope_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        b, s = input_ids.shape
        assert s <= self.cfg.max_seq_len, (
            f"sequence length {s} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x = self.tok_embeddings(input_ids)
        rope_freqs = self.rope_freqs.to(x.device)

        for layer in self.layers:
            x = layer(x, rope_freqs)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings and self.cfg.tie_embeddings:
            n -= self.tok_embeddings.weight.numel()
        return n


if __name__ == "__main__":
    cfg = PacificConfig()
    model = PacificModel(cfg)
    print(f"Total params: {model.num_params():,}")

    x = torch.randint(0, cfg.vocab_size, (2, 128))
    y = torch.randint(0, cfg.vocab_size, (2, 128))
    logits, loss = model(x, y)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")