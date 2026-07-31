# mini-deepspeed

`mini-deepspeed` is an independent, pure-PyTorch teaching project for the
state-partitioning core of DeepSpeed ZeRO. It deliberately has a different
boundary from [`mini-megatron`](../mini-megatron): Megatron-style projects are
about model parallelism and large-model execution, while this repository asks
which training states must be retained by each *data-parallel* replica.

The first version implements a compact AdamW training engine for ZeRO Stages
0 through 3. It has no dependency on DeepSpeed or on `mini-megatron`.

## What each stage owns

| Mode | Parameters per rank | Gradients per rank | Adam `m` + `v` per rank | Update synchronization |
| --- | --- | --- | --- | --- |
| Stage 0 | full | full | full | all-reduce gradients |
| ZeRO-1 | full | full | one equal shard | all-reduce gradients, then all-gather updated parameter shards |
| ZeRO-2 | full | one equal shard | one equal shard | reduce-scatter gradients, then all-gather updated parameter shards |
| ZeRO-3 | one equal shard at rest | one equal shard | one equal shard | all-gather parameters for `engine.forward()`, reduce-scatter gradients, release full parameters |

For `P` parameters and `N` data-parallel ranks, this gives the persistent
state model below (ignoring activations, temporary buffers, and communication
padding). These are logical retained-state counts, not CUDA allocator peak
measurements. Stage 2 reports the sum of equal shard lengths for its gradient
buckets, so every bucket can contribute final-shard padding when its size is
not divisible by `N`:

| Mode | Elements retained per rank |
| --- | --- |
| Stage 0 | `P + P + 2P = 4P` |
| ZeRO-1 | `P + P + 2P/N` |
| ZeRO-2 | `P + P/N + 2P/N` |
| ZeRO-3 | `P/N + P/N + 2P/N = 4P/N` |

See [the design note](docs/design.md) for why these states differ and for the
important memory-accounting boundary of this educational implementation.

## Small API

```python
import mini_deepspeed as mds

engine = mds.initialize(model, {"zero_stage": 3, "lr": 1e-3})
loss = loss_fn(engine(inputs), targets)
engine.backward(loss)
engine.step()
engine.zero_grad()

print(engine.report())
```

`initialize` accepts `zero_stage` 0, 1, 2, or 3, plus `lr`, `betas`, `eps`,
`weight_decay`, and the Stage-2 `reduce_bucket_size`. Calling `backward`
multiple times before `step` accumulates the unscaled gradient sum. The engine
intentionally exposes only the lifecycle needed to make ownership and
collectives easy to inspect. In a distributed launch it also broadcasts rank
0's initial parameters once, matching DDP's replica-initialization invariant.

ZeRO-3 deliberately has a narrower lifecycle: one `engine.forward()` must be
followed by one `engine.backward(loss)`. It gathers the full model for that
pair, then releases complete parameter tensors. `engine.parameter_vector()`
temporarily gathers a detached full vector for testing or inspection. If a
forward result will not be backpropagated, call `engine.abort_forward()`. In a
distributed launch `abort_forward()` is a coordinated call: every rank must
call it together, otherwise the ranks that call it block while the peers
proceed.

In distributed Stage 3, an ordinary rank-local module-forward failure, a
backward with missing trainable gradients, or a backward that runs over a
graph already backpropagated outside the engine (pre-existing `.grad`
tensors) is detected before the next reduce-scatter. Every rank releases its
materialization and raises, then every rank must call `zero_grad()` before
resuming. This does not relax the normal distributed contract: all ranks must
still follow the same engine API and collective schedule; rank-divergent user
control flow can still deadlock any synchronous collective program.

Stage 3 intentionally has no checkpoint format yet. `engine.state_dict()`,
`engine.load_state_dict()`, and `state_dict()` / `load_state_dict()` on the
module or any of its submodules raise rather than silently serializing the
empty parameter placeholders held between iterations. `torch.save(module)`
(pickle) and `copy.deepcopy(module)` bypass that guard and would serialize
empty tensors, so they must not be used between iterations. Pickling the
engine itself is likewise unsupported: it may appear to restore a
single-process run, but there is no multi-rank, cross-world-size,
cross-version, or mid-window guarantee. Ordinary tied
weights (two module attributes referring to the *same* `Parameter`) are
supported because PyTorch deduplicates `module.parameters()`. Independently
constructed `Parameter` views or distinct parameters sharing storage are
rejected in every stage: the flat layout updates each parameter independently,
so a shared region would keep only the last write-back and silently lose one
gradient contribution, where `torch.optim.AdamW` compounds both in-place
updates; Stage 3 would additionally break such aliases outright when it
replaces parameter storage. Non-contiguous parameters that own their full
storage (for example `nn.Parameter(tensor.t())`) are accepted and updated
exactly like `torch.optim.AdamW`. A trainable `Parameter` sharing storage
with a *frozen* `Parameter` or a registered buffer (for example a frozen copy
kept as an alias) is accepted in Stages 0-2, where the frozen alias follows
the in-place updates exactly like `torch.optim.AdamW`; Stage 3 rejects it at
initialization, because replacing parameter storage would silently leave the
frozen tensor reading stale weights. The supported entry point is
`initialize(module, ...)`. Direct Stage-3 construction is also supported only
as `ZeroOptimizer(model, config)`, with the owning `nn.Module` rather than a
parameter iterable: it must enumerate all registered Parameters (including
frozen ones) and buffers to reject hidden aliases safely. Stages 0-2 continue
to accept a parameter iterable.

## Run locally

Requires Python 3.10+ and PyTorch 2.1+.

```bash
cd /path/to/mini-deepspeed
python3 -m pip install -e '.[dev]'
python3 -m pytest -q

# Two CPU/Gloo ranks
torchrun --standalone --nproc_per_node=2 examples/train_toy.py --zero-stage 3 --device cpu

# Check that stages 0, 1, 2, and 3 end on equivalent parameter vectors.
torchrun --standalone --nproc_per_node=2 examples/validate_equivalence.py --device cpu --steps 4
```

On Gloo, ZeRO-2 takes a clearly labelled all-reduce-and-slice correctness
fallback: every rank receives the full padded gradient before retaining its
slice, so it is more communication-heavy than native reduce-scatter. NCCL uses
PyTorch's native `reduce_scatter_tensor` path. Other distributed backends are
explicitly rejected.

## L20 multi-GPU run

The `experiment` experiment container requires an explicit loopback
rendezvous rather than `torchrun --standalone`. Its current reliable settings
are `NCCL_SHM_DISABLE=1`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, and loopback
network-interface selection for Gloo and NCCL.

```bash
export NCCL_SHM_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29662 \
  examples/train_toy.py --zero-stage 2 --device cuda --steps 8

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29700 \
  examples/validate_equivalence.py --device cuda --steps 4 --reduce-bucket-size 4096

# Four L20s: exercise non-divisible shards and all ZeRO stages.
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29741 \
  examples/validate_equivalence.py --device cuda --steps 4 --reduce-bucket-size 4096
```

An L20/NCCL four-rank run completed the equivalence check with native
`reduce_scatter_tensor`: ZeRO-1/2/3 all matched ZeRO-0 after four steps. For
the 21,768-parameter toy model, the reported logical per-rank retained
model-state elements were:

| Stage | Parameters | Gradients | Adam states | Total |
| --- | ---: | ---: | ---: | ---: |
| 0 | 21,768 | 21,768 | 43,536 | 87,072 |
| 1 | 21,768 | 21,768 | 21,768 | 65,304 |
| 2 | 21,768 | 10,884 | 21,768 | 54,420 |
| 3 | 5,442 | 5,442 | 10,884 | 21,768 |

## Verification included

`tests/test_single_process.py` compares every stage with `torch.optim.AdamW`
in one process. `tests/test_distributed.py` covers world sizes 1, 2, and 4
with a 65-parameter model (deliberately not divisible by 2 or 4), two
different microbatches per update, and four Stage-2 buckets. It verifies that
the hook clears complete `param.grad` tensors after each backward call, keeps
replicas equal, checks that ZeRO-3 releases complete parameters after each
backward, and matches ZeRO-1/2/3 against the Stage-0 baseline.
`examples/validate_equivalence.py` repeats the comparison under a real NCCL
launch and compares the parameter vector after *every* step. Its ZeRO-2/3
reduce-scatter default is a pure absolute threshold (`rtol=0`, `atol=3e-7`),
rather than a broad relative tolerance: both stages use reduce-scatter while
the Stage-0 baseline uses all-reduce. The current four-L20, four-step run
observed maximum absolute differences of `1.006e-07` (ZeRO-2) and
`2.431e-07` (ZeRO-3), from the different FP32 reduction trees. That default
is calibrated only to the
documented L20 configuration (`lr=1e-3`, four steps). On the same setup at
20 steps, the observed maxima rose to `6.557e-07` (ZeRO-2) and `1.032e-06`
(ZeRO-3). Other learning rates, step counts, GPUs, or NCCL versions likewise
change how rounding noise propagates through AdamW, so re-calibrate
`--reduce-scatter-atol` on the printed `max_abs_error_over_steps` before
relying on it. `--stage3-atol` remains a compatibility alias.

`examples/compare_deepspeed.py` is the external reference check. It runs the
same model, per-rank deterministic inputs, AdamW configuration (including
non-zero weight decay), and gradient-accumulation semantics against DeepSpeed
ZeRO-0/1/2/3. It asserts equal replicated initial parameters before training and
compares every post-update parameter vector element by element. The DeepSpeed
loop follows its public GAS protocol: it calls `engine.step()` after every
microbatch and asserts the documented accumulation boundary and `global_steps`
behavior. The L20 result passed for DeepSpeed 0.19.3 and PyTorch 2.10.0a0 on
both 2 and 4 GPUs; the four-GPU maximum absolute error was at most `1.490e-08`
for ZeRO-0/1/2 and `7.451e-09` for ZeRO-3. The validation script disables
DeepSpeed NVTX annotations only, to avoid an NVTX-domain API incompatibility in
the current container; this does not change model, collective, or optimizer
behavior. DeepSpeed is installed only in an isolated validation environment and
is not a runtime dependency of this project.

## Scope and next work

This is a teaching engine, not a drop-in DeepSpeed replacement. It excludes
configuration compatibility, tensor/pipeline parallelism, checkpoint sharding,
mixed precision, gradient clipping, CPU/NVMe offload, and ZeRO-3 layer-wise
prefetch, communication overlap, or layer-at-a-time parameter release.
In particular, Stage-3 checkpoint save/load is deliberately rejected until a
dedicated (probably sharded) format is implemented.

Stage 2 registers post-accumulate-gradient hooks. When every parameter in a
complete parameter bucket is ready, the hook flattens and pads its gradients,
uses native NCCL reduce-scatter (or the labelled Gloo correctness fallback),
retains only the local averaged shard, and clears the full `param.grad`
tensors. This proves the intended lifecycle, but is not a CUDA allocator
peak-memory measurement: autograd still creates individual gradients, the
implementation uses temporary flattened buffers and no communication overlap,
and a single parameter is never split across buckets. Allocator telemetry is
required before claiming a physical peak-memory reduction.

ZeRO-3 adds parameter sharding, but its scope is intentionally explicit: it
all-gathers the *entire* flat model immediately before one forward and releases
it after that backward. Therefore its report proves steady-state ownership
(`4P/N` logical model state), not a real layer-wise backward peak-memory
reduction. Production DeepSpeed gathers and releases smaller parameter groups,
prefetches ahead, and overlaps communication; those mechanisms are outside
this mini implementation.
