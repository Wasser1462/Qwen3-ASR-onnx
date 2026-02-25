#!/usr/bin/env python3
#
# Decoder wrapper for Qwen3-ASR ONNX export (KV cache delta outputs).

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import operators as onnx_ops


def _get_first_attr(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            try:
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        v = int(v.item())
                    else:
                        continue
                return v
            except Exception:
                continue
    return default


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_llama(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    cos = cos.to(dtype=dtype)
    sin = sin.to(dtype=dtype)
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbeddingFallback(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0, rope_scaling=None):
        super().__init__()
        self.head_dim = int(head_dim)
        self.base = float(base)
        self.rope_scaling = rope_scaling

        if self.head_dim % 2 != 0:
            raise RuntimeError(f"RoPE requires even head_dim, got {self.head_dim}")

        half = self.head_dim // 2
        inv_freq = 1.0 / (self.base ** (torch.arange(0, half, dtype=torch.float32) / half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = position_ids_1d.to(torch.float32)
        if isinstance(self.rope_scaling, dict):
            t = str(self.rope_scaling.get("type", "")).lower()
            if t == "linear":
                factor = float(self.rope_scaling.get("factor", 1.0))
                if factor != 1.0:
                    pos = pos / factor

        freqs = torch.einsum("s,d->sd", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb), torch.sin(emb)


class DecoderCoreWrapper(nn.Module):
    # input_ids(B,S), audio_features(B,A,H), attention_mask(B,S), cache_position(S), caches... -> logits + kv_deltas
    def __init__(self, thinker: nn.Module, audio_token_id: int, hidden_size: int, max_total_len: int = 512):
        super().__init__()
        self.thinker = thinker
        self.audio_token_id = int(audio_token_id)
        self.hidden_size = int(hidden_size)
        self.max_total_len = int(max_total_len)

        self.core = getattr(thinker, "model", None) or getattr(thinker, "core", None)
        if self.core is None:
            raise RuntimeError("Cannot find thinker.model/core")
        self.layers = self.core.layers
        self.norm = self.core.norm
        self.lm_head = getattr(thinker, "lm_head", None)
        if self.lm_head is None:
            raise RuntimeError("Cannot find lm_head on thinker")

        self.embed_tokens = getattr(self.core, "embed_tokens", None)
        if self.embed_tokens is None:
            raise RuntimeError("Cannot find embed_tokens")

        cfg = getattr(thinker, "config", None)
        if cfg is not None and hasattr(cfg, "text_config"):
            cfg = cfg.text_config

        self.num_heads = int(_get_first_attr(cfg, ["num_attention_heads", "num_heads", "n_head"], 0))
        if self.num_heads <= 0:
            attn0 = self.layers[0].self_attn
            self.num_heads = int(_get_first_attr(attn0, ["num_heads", "n_heads"], 0))

        attn0 = self.layers[0].self_attn
        q_out = int(attn0.q_proj.weight.shape[0])
        self._head_dim = int(q_out // self.num_heads)

        k_out = int(attn0.k_proj.weight.shape[0])
        self._num_kv_heads = int(k_out // self._head_dim)

        self.group_size = int(self.num_heads // self._num_kv_heads)
        self.qkv_dim = int(self.num_heads * self._head_dim)

        rope_theta = float(_get_first_attr(cfg, ["rope_theta"], 10000.0))
        rope_scaling = _get_first_attr(cfg, ["rope_scaling"], None)
        self.rope_fallback = RotaryEmbeddingFallback(self._head_dim, base=rope_theta, rope_scaling=rope_scaling)

    def _get_cos_sin(self, attn_mod: nn.Module, cache_position_1d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        rotary = getattr(attn_mod, "rotary_emb", None)
        if rotary is None:
            rotary = getattr(self.core, "rotary_emb", None)

        if rotary is not None:
            pos_ids = cache_position_1d.view(1, -1)
            try:
                dummy = torch.zeros((1, 1, pos_ids.shape[1], self._head_dim),
                                    device=cache_position_1d.device, dtype=torch.float32)
                cos, sin = rotary(dummy, pos_ids)
                return cos, sin
            except Exception:
                pass
            try:
                cos, sin = rotary(pos_ids)
                return cos, sin
            except Exception:
                pass

        return self.rope_fallback(cache_position_1d)

    def _apply_qk_norm_if_any(self, attn_mod: nn.Module, q: torch.Tensor, k_kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = getattr(attn_mod, "q_norm", None)
        k_norm = getattr(attn_mod, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k_kv = k_norm(k_kv)
        return q, k_kv

    def _inject_audio_features(self, tok: torch.Tensor, input_ids: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        # ONNX-friendly masked_scatter sequential: use cumsum rank + gather
        B, S, H = tok.shape
        mask = (input_ids == self.audio_token_id)  # (B,S)
        m64 = mask.to(torch.int64)
        rank = torch.cumsum(m64, dim=1) - 1

        a_shape = onnx_ops.shape_as_tensor(audio_features)
        A = a_shape[1].to(torch.int64)
        A1 = torch.clamp(A - 1, min=0)

        zero = torch.zeros_like(rank)
        rank0 = torch.maximum(rank, zero)
        rankc = torch.minimum(rank0, A1)

        idx = rankc.unsqueeze(-1).expand(B, S, H)
        gathered = torch.gather(audio_features, dim=1, index=idx)
        return torch.where(mask.unsqueeze(-1), gathered.to(tok.dtype), tok)

    def _attn_chunk(
        self,
        attn_mod: nn.Module,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        cache_k_full: torch.Tensor,
        cache_v_full: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # keep attention_mask in graph
        x = x + (attention_mask[:, :1].to(dtype=x.dtype) * 0.0)

        x_shape = onnx_ops.shape_as_tensor(x)
        B = x_shape[0]
        S = x_shape[1]

        past_len = cache_position[0]
        past_len_cache = torch.clamp(past_len, min=1)
        cache_k = cache_k_full[:, :past_len_cache]
        cache_v = cache_v_full[:, :past_len_cache]

        q = attn_mod.q_proj(x)
        k = attn_mod.k_proj(x)
        v = attn_mod.v_proj(x)

        i64 = x_shape.dtype
        dev = x.device
        t_num_heads = torch.tensor(self.num_heads, device=dev, dtype=i64)
        t_num_kv_heads = torch.tensor(self.num_kv_heads, device=dev, dtype=i64)
        t_head_dim = torch.tensor(self.head_dim, device=dev, dtype=i64)

        q = onnx_ops.reshape_from_tensor_shape(q, torch.stack([B, S, t_num_heads, t_head_dim]))
        k_kv = onnx_ops.reshape_from_tensor_shape(k, torch.stack([B, S, t_num_kv_heads, t_head_dim]))
        v_kv = onnx_ops.reshape_from_tensor_shape(v, torch.stack([B, S, t_num_kv_heads, t_head_dim]))

        q = q.permute(0, 2, 1, 3)
        k_kv = k_kv.permute(0, 2, 1, 3)
        v_kv = v_kv.permute(0, 2, 1, 3)

        q, k_kv = self._apply_qk_norm_if_any(attn_mod, q, k_kv)

        cos, sin = self._get_cos_sin(attn_mod, cache_position)
        q = _apply_rope_llama(q, cos, sin)
        k_kv = _apply_rope_llama(k_kv, cos, sin)

        k_h = k_kv.repeat_interleave(self.group_size, dim=1)
        v_h = v_kv.repeat_interleave(self.group_size, dim=1)

        cache_k_h = cache_k.permute(0, 2, 1, 3).repeat_interleave(self.group_size, dim=1)
        cache_v_h = cache_v.permute(0, 2, 1, 3).repeat_interleave(self.group_size, dim=1)

        scaling = getattr(attn_mod, "scaling", None)
        try:
            scale = float(scaling) if scaling is not None else float(self.head_dim) ** -0.5
        except Exception:
            scale = float(self.head_dim) ** -0.5

        scores_cache = torch.matmul(q.float(), cache_k_h.float().transpose(-1, -2)) * scale
        scores_new = torch.matmul(q.float(), k_h.float().transpose(-1, -2)) * scale

        cache_valid = (past_len > 0).to(dtype=scores_cache.dtype)
        scores_cache = scores_cache + (1.0 - cache_valid) * (-1e4)

        q_pos = cache_position.unsqueeze(1)
        k_pos = cache_position.unsqueeze(0)
        causal_ok = (k_pos <= q_pos)
        causal_new = torch.where(
            causal_ok,
            torch.zeros((), device=scores_new.device, dtype=scores_new.dtype),
            torch.full((), -1e4, device=scores_new.device, dtype=scores_new.dtype),
        ).unsqueeze(0).unsqueeze(0)
        scores_new = scores_new + causal_new

        scores = torch.cat([scores_cache, scores_new], dim=-1)
        attn = torch.softmax(scores, dim=-1).to(dtype=q.dtype)

        v_total = torch.cat([cache_v_h, v_h], dim=2)
        out = torch.matmul(attn, v_total)

        out = out.permute(0, 2, 1, 3)
        t_qkv_dim = torch.tensor(self.qkv_dim, device=dev, dtype=i64)
        out = onnx_ops.reshape_from_tensor_shape(out, torch.stack([B, S, t_qkv_dim]))
        out = attn_mod.o_proj(out)

        k_delta = k_kv.permute(0, 2, 1, 3).contiguous()
        v_delta = v_kv.permute(0, 2, 1, 3).contiguous()
        return out, k_delta, v_delta

    def _mlp(self, mlp_mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
        gate = mlp_mod.gate_proj(x)
        up = mlp_mod.up_proj(x)
        h = F.silu(gate) * up
        return mlp_mod.down_proj(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        *cache_flat: torch.Tensor,
    ):
        L = len(self.layers)
        if len(cache_flat) != 2 * L:
            raise RuntimeError(f"expect {2*L} cache tensors, got {len(cache_flat)}")

        model_dtype = next(self.thinker.parameters()).dtype
        tok = self.embed_tokens(input_ids).to(model_dtype)
        tok = self._inject_audio_features(tok, input_ids, audio_features.to(model_dtype))

        x = tok
        deltas = []
        for i, layer in enumerate(self.layers):
            residual = x
            x_norm = layer.input_layernorm(x)

            cache_k = cache_flat[2 * i].to(model_dtype)
            cache_v = cache_flat[2 * i + 1].to(model_dtype)

            attn_out, k_delta, v_delta = self._attn_chunk(
                layer.self_attn,
                x_norm,
                attention_mask,
                cache_position,
                cache_k,
                cache_v,
            )

            x = residual + attn_out
            residual = x

            x_norm2 = layer.post_attention_layernorm(x)
            mlp_out = self._mlp(layer.mlp, x_norm2)
            x = residual + mlp_out

            deltas.append(k_delta.to(model_dtype))
            deltas.append(v_delta.to(model_dtype))

        x = self.norm(x)
        logits = self.lm_head(x).float()
        return (logits, *deltas)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim
