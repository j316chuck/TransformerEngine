# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import jax
from jax import numpy as jnp
from flax import linen as nn
from typing import Optional
import transformer_engine.jax as te


class DotProductAttention(nn.Module):
    """Attention operation in Transformer layer

    Built with plain Flax modules.

    """
    num_attention_heads: int
    kv_channels: int
    dropout_rate: Optional[float] = 0.1
    dropout_rng: Optional[str] = 'dropout'
    dtype: Optional[type] = jnp.float32

    def setup(self):
        self.projection_size = self.kv_channels * self.num_attention_heads
        self.hidden_size_per_attention_head = self.kv_channels
        self.norm_factor = jnp.sqrt(self.hidden_size_per_attention_head)
        self.dropout = nn.Dropout(self.dropout_rate, rng_collection=self.dropout_rng)

    def __call__(
            self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None,
            train : Optional[bool] = False
    ) -> jnp.ndarray:
        sq = query.shape[0]  # query sequence length
        sk = key.shape[0]    # key sequence length
        b = query.shape[1]   # batch size
        np = query.shape[2]  # number of attention heads
        hn = value.shape[3]  # attention head size

        # Query * Key Mat-Mul
        query = jnp.reshape(query, (sq, b*np, hn))             # [sq, b, np, hn]-->[b*np, sq, hn]
        key = jnp.reshape(key, (sk, b*np, hn))                 # [sk, b, np, hn]-->[b*np, hn, sk]
        bmm1 = jax.lax.batch_matmul(               # [sq, hn]*[hn, sk]=[sq, sk] batched over b*np
            jnp.transpose(query, axes=(1, 0, 2)),
            jnp.transpose(key, axes=(1, 2, 0))
        )
        scores = jnp.reshape(bmm1, (b, np, sq, sk))            # [b*np, sq, sk]-->[b, np, sq, sk]
        
        # Softmax + Dropout
        probs = jax.nn.softmax(scores, where=attention_mask)
        probs = self.dropout(probs, deterministic=(not train))

        # Probabilities * Values Mat-Mul
        value = jnp.reshape(value, (sk, b*np, hn))             # [sk, b, np, hn]-->[sk, b*np, hn]
        probs = jnp.reshape(probs, (b*np, sq, sk))             # [b, np, sq, sk]-->[b*np, sq, sk]
        context = jax.lax.batch_matmul(
            probs,                                 # [sq, sk]*[sk, hn]=[sq, hn] batched over b*np
            jnp.transpose(value, axes=(1,0,2))
        )

        context = jnp.reshape(context, (b, np, sq, hn))        # [b*np, sq, hn]-->[b, np, sq, hn]
        context = jnp.transpose(context, (2, 0, 1, 3))         # [b, np, sq, hn]-->[sq, b, np, hn]
        context = jnp.reshape(context, 
                              (sq, b, self.projection_size))   # [sq, b, np, hn]-->[sq, b, np*hn]
        return context.astype(self.dtype)


class BasicMLP(nn.Module):
    """Feed-forward network in Transformer layer

    Built with plain Flax modules.

    """
    hidden_size : int
    ffn_hidden_size : int
    dtype : Optional[type] = jnp.float32
    
    def setup(self):
        self.linear1 = nn.DenseGeneral(self.ffn_hidden_size, dtype=self.dtype)
        self.linear2 = nn.DenseGeneral(self.hidden_size, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.linear1(x)
        x = jax.nn.gelu(x, approximate=True)
        x = self.linear2(x)
        return x


def inspect_params(data, prefix='', show_all=False):
    at_top_level = 'params' in data.keys()
    for idx, key in enumerate(data.keys()):
        if show_all or not at_top_level or key == 'params':
            if isinstance(data[key], dict):
                print(f"{prefix + key}")
                base = prefix[:-3]
                if len(prefix) > 0:
                    base += '   ' if idx+1 == len(data.keys()) else '|  '
                inspect_params(data[key], prefix=base+'|__')
            else:
                try:
                    info = data[key].shape
                except AttributeError:
                    info = data[key]
                print(f"{prefix + key}: {info}")

def share_params(te_params, flax_params):
    for key in te_params.keys():
        if key in flax_params.keys():
            if isinstance(te_params[key], dict):
                te_params[key] = share_params(te_params[key], flax_params[key])
            elif 'kernel' in key:
                te_params[key] = flax_params['kernel']
            else:
                te_params[key] = flax_params[key]
    return te_params

