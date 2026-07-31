# mini-deepspeed

`mini-deepspeed` is an independent, pure-PyTorch teaching project for the
state-partitioning core of DeepSpeed ZeRO. It deliberately has a different
boundary from [`mini-megatron`](../mini-megatron): Megatron-style projects are
about model parallelism and large-model execution, while this repository asks
which training states must be retained by each *data-parallel* replica.

The first version implements a compact AdamW training engine for ZeRO Stages
0, 1, and 2. It has no dependency on DeepSpeed or on `mini-megatron`.

## What each stage owns

| Mode | Parameters per rank | Gradients per rank | Adam `m` + `v` per rank | Update synchronization |
| --- | --- | --- | --- | --- |
| Stage 0 | full | full | full | all-reduce gradients |
| ZeRO-1 | full | full | one equal shard | all-reduce gradients, then all-gather updated parameter shards |
| ZeRO-2 | full | one equal shard | one equal shard | reduce-scatter gradients, then all-gather updated parameter shards |

For `P` parameters and `N` data-parallel ranks, this gives the persistent
state model below (ignoring activations, temporary buffers, and communication
padding). The runtime report uses the equal-sized shard length, so a model
whose parameter count is not divisible by `N` includes at most one shard's
padding in its partitioned counters:

| Mode | Elements retained per rank |
| --- | --- |
| Stage 0 | `P + P + 2P = 4P` |
| ZeRO-1 | `P + P + 2P/N` |
| ZeRO-2 | `P + P/N + 2P/N` |

See [the design note](docs/design.md) for why these states differ and for the
important memory-accounting boundary of this educational implementation.

## Small API

```python
import mini_deepspeed as mds

engine = mds.initialize(model, {"zero_stage": 2, "lr": 1e-3})
loss = loss_fn(engine(inputs), targets)
engine.backward(loss)
engine.step()
engine.zero_grad()

print(engine.report())
```

`initialize` accepts `zero_stage` 0, 1, or 2, plus `lr`, `betas`, `eps`, and
`weight_decay`. The engine intentionally exposes only the lifecycle needed to
make the ownership and collectives easy to inspect. In a distributed launch it
also broadcasts rank 0's initial parameters once, matching DDP's replica
initialization invariant.

## Run locally

Requires Python 3.10+ and PyTorch 2.1+.

```bash
cd /path/to/mini-deepspeed
python3 -m pip install -e '.[dev]'
python3 -m pytest -q

# Two CPU/Gloo ranks
torchrun --standalone --nproc_per_node=2 examples/train_toy.py --zero-stage 2 --device cpu

# Check that stages 0, 1, and 2 end on identical parameter vectors.
torchrun --standalone --nproc_per_node=2 examples/validate_equivalence.py --device cpu --steps 4
```

On a Gloo build that does not provide `reduce_scatter`, ZeRO-2 takes a clearly
labelled all-reduce-and-slice correctness fallback. NCCL uses PyTorch's native
`reduce_scatter_tensor` path.

## L20 two-GPU run

The `experiment` experiment container requires an explicit loopback
rendezvous rather than `torchrun --standalone`. Its current reliable settings
are `NCCL_SHM_DISABLE=1` and `CUDA_DEVICE_MAX_CONNECTIONS=1`.

```bash
export NCCL_SHM_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29662 \
  examples/train_toy.py --zero-stage 2 --device cuda --steps 8

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29700 \
  examples/validate_equivalence.py --device cuda --steps 4
```

An L20/NCCL two-rank run completed eight toy-training steps at every stage
with the same reported losses at steps 1 and 8. For the 21,768-parameter toy
model, the reported per-rank retained model-state elements were:

| Stage | Parameters | Gradients | Adam states | Total |
| --- | ---: | ---: | ---: | ---: |
| 0 | 21,768 | 21,768 | 43,536 | 87,072 |
| 1 | 21,768 | 21,768 | 21,768 | 65,304 |
| 2 | 21,768 | 10,884 | 21,768 | 54,420 |

## Verification included

`tests/test_single_process.py` compares every stage with `torch.optim.AdamW`
in one process. `tests/test_distributed.py` launches two Gloo ranks with
different data on each rank, verifies that each stage keeps replicas equal,
and verifies ZeRO-1/2 against the Stage-0 baseline. The standalone
`examples/validate_equivalence.py` repeats the same comparison for a real
NCCL two-GPU launch.

## Scope and next work

This is a teaching engine, not a drop-in DeepSpeed replacement. It excludes
configuration compatibility, tensor parallelism, pipeline parallelism,
checkpoint sharding, mixed precision, gradient accumulation, CPU/NVMe
offload, and ZeRO-3 parameter sharding.

Most importantly, Stage 2 is correct about *logical state ownership*, but
autograd still materializes a full `param.grad` before the implementation
flattens it. Its report therefore accounts for retained state after
synchronization, not CUDA's transient backward peak. A real physical
peak-memory Stage-2 implementation needs bucketed gradient hooks that
reduce-scatter and release each bucket as soon as it is ready. ZeRO-3 would
then shard parameters too, all-gathering a layer only for its forward/backward
use. Those are the intended next milestones.
