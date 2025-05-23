import logging
from dataclasses import dataclass
from typing import Callable, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy.physics.units import current

from models.conf import ModelConfig
from models.base import BaseModel

log = logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

@dataclass
class Cache:
    key_cache: torch.Tensor
    value_cache: torch.Tensor

class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embed // config.n_heads
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.inv_freq = 1.0 / (t ** (r / d)).float()

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(x.device)

        position_ids = position_ids.unsqueeze(-1)
        freqs = position_ids * inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.n_embed = config.n_embed
        self.n_embed_per_head = config.n_embed // config.n_heads
        self.n_kv_embed = config.n_kv_heads * self.n_embed_per_head

        self.q_proj = nn.Linear(self.n_embed, self.n_embed, bias=True)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=True)
        self.o_proj = nn.Linear(self.n_embed, self.n_embed, bias=False)

    def forward(self, x, cos, sin, past_cache, use_cache):
        B, T, C = x.size()

        q = self.q_proj(x)

        if past_cache is not None:
            x_kv = x[:, -1:, :]
        else:
            x_kv = x

        k = self.k_proj(x_kv)
        v = self.v_proj(x_kv)

        q = q.view(B, T, self.n_heads, self.n_embed_per_head).transpose(1, 2)
        k = k.view(B, k.size(1), self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)
        v = v.view(B, v.size(1), self.n_kv_heads, self.n_embed_per_head).transpose(1, 2)

        q = self._apply_rotary_pos_emb(q, cos, sin)
        if past_cache is not None:
            cos = cos[:, -1:, :]
            sin = sin[:, -1:, :]
        k = self._apply_rotary_pos_emb(k, cos, sin)

        if past_cache is not None:
            past_k, past_v = past_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)

        return y, new_cache

    @staticmethod
    def _apply_rotary_pos_emb(x, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        x_embed = (x * cos) + (CausalSelfAttention._rotate_half(x) * sin)
        return x_embed

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, past_cache, use_cache):
        attn_out, present_cache = self.self_attn(self.input_layernorm(x), cos, sin, past_cache, use_cache)
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present_cache


class Qwen2(nn.Module, BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = nn.ModuleDict(dict(
            embed_tokens=nn.Embedding(config.vocab_size, config.n_embed),
            rotary_emb=RotaryEmbedding(config),
            layers=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            norm=RMSNorm(config.n_embed, eps=config.rms_norm_eps)
        ))
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    @staticmethod
    def _get_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.LongTensor,
                target: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                past_cache: Optional[List[Cache]] = None,
                ):
        if target is not None and use_cache :
            log.error('use_cache=True не совместим с обучением. Выставите use_cache=False')
            raise 'use_cache=True. Must by False'
        use_cache = False if use_cache is None else use_cache

        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        cos, sin = self.model.rotary_emb(x, position_ids)

        new_cache: List[Cache] = []
        for i, layer in enumerate(self.model.layers):
            current_cache = None if past_cache is None else self._get_kv_cache(past_cache[i], input_ids.device)
            x, present_cache = layer(x, cos, sin, current_cache, use_cache)
            self._set_cache(new_cache, present_cache)

        x = self.model.norm(x)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)

        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        new_cache = new_cache if use_cache else None
        return logits, loss, new_cache

    @staticmethod
    def _set_cache(cache, new_cache):
        if new_cache is not None:
            present = Cache(
                new_cache[0].detach().cpu(),
                new_cache[1].detach().cpu()
            )
            cache.append(present)

    @staticmethod
    def _get_kv_cache(cache, device):
        if cache is not None:
            cache_on_device = (
                cache.key_cache.to(device),
                cache.value_cache.to(device)
            )
            return cache_on_device
        return None

