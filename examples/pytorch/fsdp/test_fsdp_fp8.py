import os
import argparse
import torch

def lowercase(s):
    return str(s).lower()

def torch_dtype(d):
    typemap = {
        'fp32' : torch.float32,
        'float32' : torch.float32,
        'fp16' : torch.float16,
        'float16' : torch.float16,
        'bf16' : torch.bfloat16,
        'bfloat16' : torch.bfloat16
    }
    if lowercase(d) not in typemap.keys():
        raise TypeError
    return typemap[lowercase(d)]

parser = argparse.ArgumentParser(description="Run Transformer Engine modules with the " +
                                    "torch.distributed.fsdp.FullyShardedDataParallel strategy.")
parser.add_argument('-m', "--module", type=lowercase, default='linear',
                    choices=['linear', 'layernormlinear', 'layernormmlp', 'transformerlayer'],
                    help="Transformer Engine module that will be wrapped in FSDP.")
parser.add_argument("--no-fix", action="store_true", default=False,
                    help="Disables the cast+transpose fix for FP8 FSDP OOM bug.")
parser.add_argument("--no-fp8", action="store_true", default=False,
                    help="Disables the te.fp8_autocast() context.")
parser.add_argument('-i', "--num-iters", type=int, default=3,
                    help="Number of fake training iterations.")
parser.add_argument('-b', "--batch-size", type=int, default=32,
                    help="Input batch size.")
parser.add_argument('-s', "--seq-length", type=int, default=2048,
                    help="Input sequence length.")
parser.add_argument('-n', "--num-heads", type=int, default=64,
                    help="Number of attention heads.")
parser.add_argument('-d', "--head-dim", type=int, default=512,
                    help="Dimension of each attention head (number of KV channels).")
parser.add_argument('-l', "--num-layers", type=int, default=1,
                    help="Number of Transformer Engine modules chained together with nn.Sequential.")
parser.add_argument("--seed", type=int, default=1234,
                    help="PyTorch RNG seed.")
parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16,
                    help="Data type for input tensor and Transformer Engine module parameters.")
args = parser.parse_args()

import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy, transformer_auto_wrap_policy

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])

dist.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
if local_rank == 0:
    print(f"[GPU-0] WORLD_SIZE = {world_size}")

dtype = args.dtype
seq_len = args.seq_length
batch_size = args.batch_size
num_heads = args.num_heads
head_dim = args.head_dim
hidden_size = num_heads * head_dim
ffn_hidden_size = 4 * hidden_size
torch.manual_seed(args.seed)

parallel_mode = 'fsdp'
if args.no_fix:
    parallel_mode = None
    if local_rank == 0:
        print("[GPU-0] FP8 FSDP OOM workaround DISABLED")
lowercase(args.module)

def get_layer(layer_type, layer_number=None):
    if layer_type == 'linear':
        return te.Linear(
            hidden_size, ffn_hidden_size,
            bias=True,
            params_dtype=dtype,
            parallel_mode=parallel_mode
        )
    elif layer_type == 'layernormlinear':
        return te.LayerNormLinear(
            hidden_size, ffn_hidden_size,
            bias=True,
            params_dtype=dtype,
            parallel_mode=parallel_mode
        )
    elif layer_type == 'layernormmlp':
        return te.LayerNormMLP(
            hidden_size, ffn_hidden_size,
            bias=True,
            seq_length=seq_len,
            params_dtype=dtype,
            is_fsdp=not args.no_fix
        )
    elif layer_type == 'transformerlayer':
        return te.TransformerLayer(
            hidden_size, ffn_hidden_size, num_heads,
            layer_number=layer_number,
            seq_length=seq_len,
            fuse_qkv_params=True,
            qkv_weight_interleaved=True,
            params_dtype=dtype,
            is_fsdp=not args.no_fix
        )
    else:
        raise NotImplementedError

layer_type = lowercase(args.module)
if args.num_layers > 1:
    if layer_type in ['linear', 'layernormlinear']:
        # input and output features need to have the same size in order to
        # chain these layers back to back in nn.Sequential
        ffn_hidden_size = hidden_size
    te_layer_list = []
    for i in range(args.num_layers):
        new_layer = get_layer(layer_type, layer_number=i+1)
        te_layer_list.append(new_layer.cuda())
    te_model = nn.Sequential(*te_layer_list)
else:
    te_model = get_layer(layer_type).cuda()

if local_rank == 0:
    print(f"[GPU-0] {te_model}\n", end='')

wrap_policy = always_wrap_policy
if layer_type == 'transformerlayer':
    wrap_policy = transformer_auto_wrap_policy
te_model = FullyShardedDataParallel(te_model,
                                    use_orig_params=True,
                                    mixed_precision=MixedPrecision(
                                        param_dtype=dtype,
                                        reduce_dtype=torch.float32,
                                    ),
                                    sync_module_states=True,
                                    auto_wrap_policy=wrap_policy)

fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

optim = torch.optim.Adam(te_model.parameters(), lr=0.0001)
torch.cuda.synchronize()
start.record()
for i in range(args.num_iters):
    x = torch.rand(seq_len//world_size, batch_size, hidden_size).to(dtype=dtype).cuda()
    with te.fp8_autocast(enabled=not args.no_fp8, fp8_recipe=fp8_recipe):
        y = te_model(x)
        loss = y.sum()
    loss.backward()
    optim.step()
    del x
    if local_rank == 0:
        print(f"[GPU-0] Iter. {i+1}\n", end='')
end.record()
torch.cuda.synchronize()

train_time = start.elapsed_time(end)/1000.
max_memory_alloc = int(torch.cuda.max_memory_allocated() * 1e-6)
print(f"\n[GPU-{local_rank}] Training Time: {train_time}s\n" +
        f"[GPU-{local_rank}] Avg. Iter. Time: {train_time /args.num_iters}s\n" +
        f"[GPU-{local_rank}] Max memory allocated = {max_memory_alloc}MiB\n\n", end='')
