# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import time
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional
import transformer_engine.jax as te

def speedometer(
        module: nn.Module,
        params: jnp.ndarray,
        input: jnp.ndarray,
        output_grad: jnp.ndarray,
        forward_kwargs: dict = {},
        fp8_autocast_kwargs: Optional[dict] = None,
        timing_iters: int = 50,
        warmup_iters: int = 50,
) -> None:
    """Measure average run time for a Flax module

    Performs forward and backward passes.
    """
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = { "enabled": False }

    fwd_bwd_func = jax.value_and_grad(module.apply)

    # Warmup runs
    for _ in range(warmup_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output, output_grad = fwd_bwd_func(params, input, **forward_kwargs)

    # Timing runs
    jax.block_until_ready()
    start = time.time()
    for _ in range(timing_iters):
        with te.fp8_autocast(**fp8_autocast_kwargs):
            output, output_grad = fwd_bwd_func(params, input, **forward_kwargs)
    jax.block_until_ready()
    end = time.time()

    print(f"Mean time: {(end - start)/timing_iters} ms")


class DotProductAttention(nn.Module):
    """Attention operation in Transformer layer

    Built with plain Flax modules.

    """
    num_attention_heads: int
    kv_channels: int
    attention_dropout: float = 0.1

    def setup(self):
        self.projection_size = self.kv_channels * self.num_attention_heads
        self.hidden_size_per_attention_head = self.kv_channels
        self.norm_factor = jnp.sqrt(self.hidden_size_per_attention_head)
        self.dropout = nn.Dropout(self.attention_dropout, rng_collection='attention')

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        b = query.size(1)
        np = query.size(2)
        sq = query.size(0)
        sk = key.size(0)
        hn = value.size(3)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query = query.view(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(sk, b * np, -1)

        bmm1 = jax.lax.batch_matmul(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) / self.norm_factor

        # change view to [b, np, sq, sk]
        attention_scores = bmm1.view(b, np, sq, sk)

        attention_probs = jax.nn.softmax(attention_scores, where=attention_mask)

        attention_probs = self.dropout(attention_probs)

        # change view [sk, b * np, hn]
        value = value.view(sk, b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = jax.lax.batch_matmul(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = jnp.transpose(context, axes=(2, 0, 1, 3))

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.view(sq, b, self.projection_size)

        return context


class BasicMLP(nn.Module):
    """Feed-forward network in Transformer layer

    Built with plain Flax modules.

    """
    hidden_size : int
    ffn_hidden_size : int
    
    def setup(self):
        super().__init__()
        self.linear1 = nn.DenseGeneral(self.ffn_hidden_size, use_bias=True)
        self.linear2 = nn.DenseGeneral(self.hidden_size, use_bias=True)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = jax.nn.gelu(x, approximate=True)
        x = self.linear2(x)
        return x
