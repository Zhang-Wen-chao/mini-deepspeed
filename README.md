# mini-deepspeed

`mini-deepspeed` is an independent, pure-PyTorch teaching project for the
state-partitioning core of DeepSpeed ZeRO. It deliberately has a different
boundary from Megatron-style projects: those focus on model parallelism and
large-model execution, while this repository asks which training states must
be retained by each *data-parallel* replica.

The first version implements a compact AdamW training engine for ZeRO Stages
0 through 3. It has no dependency on DeepSpeed or on `mini-megatron`.

> **Scope.** This is a teaching implementation, not a production DeepSpeed
> ZeRO-3 replacement. In particular, its Stage 3 eagerly gathers and releases
> the entire model for each forward/backward pair; it does not implement
> layer-wise scheduling, prefetch, communication/compute overlap, mixed
> precision, offload, sharded checkpoints, general autograd support, or
> production-grade fault tolerance.

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

## 代码导读（Code Walkthrough）

上一节是 API 地图，这一节是从入口走到出口的那条线。所有锚点都是 `文件:行`（相对项目根）。

### 入口清单：跑哪个，走的是哪条路

| 命令 | 走的路径 | 入口锚点 |
|---|---|---|
| `python examples/train_toy.py --zero-stage 3` | 单进程/多进程训练一个 MLP，打印 `report()` | `mds.initialize(...)` → `train_toy.py:61`，训练循环 `train_toy.py:62` |
| `torchrun ... examples/validate_equivalence.py` | 真实分布式下 ZeRO-1/2/3 与 Stage-0 基线逐 step 对比 | `run_stage(...)` → `validate_equivalence.py:54`，`main()` → `validate_equivalence.py:87` |
| `python examples/compare_deepspeed.py` | 与官方 DeepSpeed 同模型同配置逐元素对比（需 CUDA + 隔离环境装 DeepSpeed） | `_run_mini(...)` → `compare_deepspeed.py:100`，`_run_deepspeed(...)` → `compare_deepspeed.py:162` |
| `pytest` | 单进程 AdamW 等价 + 分布式 Gloo 等价 | `test_single_process.py:16`，`test_distributed.py:184` |

一个贯穿全项目的设计点：**`engine.py` 整个文件只有 109 行**，`DeepSpeedEngine`
（`engine.py:14`）的 `forward`/`backward`/`step`/`zero_grad` 四个 API 全部转发给
`ZeroOptimizer`；真正的生命周期在 `zero.py`，分片数学全部收敛在 `layout.py`。
换 stage 不换引擎——README 开头「What each stage owns」那张表，就是 `report()`
（`zero.py:502`）按 stage 分支（`:504-518`）打印出来的数字。

### 一条训练 step 的生命周期：讲代码就讲这条线

```python
engine = mds.initialize(model, {"zero_stage": 3, "lr": 1e-3})   # engine.py:98 → zero.py:75
loss = loss_fn(engine(inputs), targets)                          # engine.py:30 → zero.py:178
engine.backward(loss)                                            # engine.py:40 → zero.py:160
engine.step()                                                    # engine.py:43 → zero.py:329
engine.zero_grad()                                               # engine.py:46 → zero.py:141
```

`initialize` 建 `FlatParameterLayout`（`layout.py:21`）并广播 rank 0 的初始参数
（`zero.py:220`）。之后每个 API 按 stage 分派，分派点都在 `zero.py`：

| # | 动作 | 位置 | 干什么 |
|---|---|---|---|
| 1 | `forward` | `engine.py:30` → `zero.py:178` | 只有 Stage 3 做事：all-gather 分片 + `materialize` |
| 2 | `backward` | `engine.py:40` → `zero.py:160` | 0/1 直接 `loss.backward()`；2 走 bucket hooks；3 走 `_backward_stage3` |
| 3 | `step` | `engine.py:43` → `zero.py:329` | 0 all-reduce 全量；1 all-reduce + 分片；2/3 本地 shard AdamW + all-gather 回写 |
| 4 | `zero_grad` | `engine.py:46` → `zero.py:141` | 清 `.grad`；2/3 额外重置分片累积窗口 |

第 2、3 步往下按 stage 分三支，参数/梯度/优化器状态的流转是这条线的核心：

```
Stage 0/1   backward  loss.backward()                          zero.py:163
            step      0: flatten → all-reduce → _adamw 全量    zero.py:330-339
                      1: pad → all-reduce → local_shard → _adamw 分片
                         → all-gather 回写                     zero.py:341-353
Stage 2     backward  post-accumulate hook 逐桶触发            zero.py:258 → :305
                      → pad → reduce-scatter → 留 shard → 清 param.grad
            step      逐桶 local_shard → _adamw → all-gather 回写  zero.py:361-393
Stage 3     forward   all-gather → materialize（参数变 view）   zero.py:188-189
            backward  loss.backward → pad → reduce-scatter → release  zero.py:420-450
            step      只更新本地 shard，无通信                  zero.py:462-467
```

三个落在这条线上的设计点，适合主动展开：

- **分片数学只有五个操作**（`layout.py`）：`pad`（`:136`）/`local_shard`（`:143`）/
  `assign`（`:148`）/`materialize`（`:162`）/`release`（`:171`）。padding 只用于
  通信，`assign` 丢弃 padding；`shard_numel = ceil(numel / world_size)`
  （`layout.py:53`）。所有 stage 的参数/梯度流转都只是这五个操作的排列组合。
- **Stage 2 的通信发生在 autograd 内部**：`register_post_accumulate_grad_hook`
  （`zero.py:258`）在桶内最后一个参数梯度就绪时立即 reduce-scatter 并清空完整
  `param.grad`（`zero.py:321-327`）——这是与"step 边界才通信"的第一版的关键区别，
  也是 README「Verification included」里"hook 清空完整梯度"断言的代码落点
  （`test_distributed.py:85-86`）。
- **Stage 3 的 materialize/release 边界**：参数在 forward 前 all-gather 成临时向量的
  view（`layout.py:162`），backward 后 release 成空 tensor（`layout.py:171`）。
  `parameter_vector()`（`zero.py:211`）是诊断用的临时 gather，不改变持久所有权。
  README「Small API」里"one forward must be followed by one backward"的约束就是
  `prepare_forward` 的 `_stage3_parameters_materialized` 检查（`zero.py:182`）。

### 可替换边界

- **优化器算法**：`_adamw`（`zero.py:531`）是唯一的优化器实现，四个 stage 共用。
- **通信原语**：`_reduce_scatter_mean`（`zero.py:564`）是唯一的梯度通信（NCCL
  native `reduce_scatter_tensor` / Gloo all-reduce+slice fallback，`zero.py:571-578`）；
  `_all_gather_shards`（`zero.py:588`）是唯一的参数通信。换通信后端只改这两处。
- **失败协调**：`_synchronize_stage3_failure`（`zero.py:490`）用 all-reduce MAX 把
  rank-local 失败变成协调失败，避免一卡 raise 而其他卡阻塞在后续 collective。
- **Stage 3 的 checkpoint 拒绝**：`engine.py:60-95` 在引擎、模块、每个子模块上注册
  `state_dict` pre-hook（`engine.py:26-28`），保证分片空占位符永远不会被序列化。

### 阅读顺序

```
1. examples/train_toy.py     先跑一遍，建立"谁在驱动谁"的整体印象
2. engine.py                 109 行薄壳，四个 API 全部转发
3. layout.py                 FlatParameterLayout：五个操作覆盖所有分片流转
4. zero.py:329 step()        按 stage 分派的主干
5. zero.py:305 / :395        Stage 2 bucket hook 与 Stage 3 backward 两条支线
6. tests/                    单进程 AdamW 等价 + 分布式 Gloo 等价
7. examples/validate_equivalence.py / compare_deepspeed.py   真实分布式与官方参考
```

## Run locally

Requires Python 3.10+ and PyTorch 2.1+.

```bash
git clone https://github.com/Zhang-Wen-chao/mini-deepspeed.git
cd mini-deepspeed
python -m pip install -e '.[dev]'
python -m pytest -q

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

## Multi-GPU / NCCL validation

The following is a reproducible single-node NCCL example. Use an explicit
loopback rendezvous if `torchrun --standalone` is unreliable in your container
or hostname setup. The environment variables below were required by the
recorded L20 validation environment; treat them as environment-specific
workarounds, not universal requirements.

```bash
export NCCL_SHM_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29662 \
  examples/train_toy.py --zero-stage 2 --device cuda --steps 8

torchrun --nnodes=1 --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29700 \
  examples/validate_equivalence.py --device cuda --steps 4 --reduce-bucket-size 4096

# Four GPUs: exercise non-divisible shards and all ZeRO stages.
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29741 \
  examples/validate_equivalence.py --device cuda --steps 4 --reduce-bucket-size 4096
```

On 2026-07-31, a four-L20, four-rank NCCL run completed the equivalence check
with native `reduce_scatter_tensor`: ZeRO-1/2/3 all matched ZeRO-0 after four
steps. This is an environment-bound record, not a performance or portability
claim. For the 21,768-parameter toy model, the reported logical per-rank
retained model-state elements were:

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
behavior. On 2026-07-31, the code committed as `b9de653` passed this strengthened
three-update check on four L20 GPUs with DeepSpeed 0.19.3, PyTorch
`2.10.0a0+a36e1d39eb.nv26.01.42222806`, and CUDA 13.1: the maximum absolute
error was `1.490e-08` for ZeRO-0/1/2 and `7.451e-09` for ZeRO-3. This is
environment-bound evidence, not a claim about other hardware, models, or longer
runs. The validation script disables DeepSpeed NVTX annotations only, to avoid
an NVTX-domain API incompatibility in that container; this does not change
model, collective, or optimizer behavior. DeepSpeed is installed only in an
isolated validation environment and is not a runtime dependency of this project.

## Scope and next work

This teaching engine excludes configuration compatibility, tensor/pipeline
parallelism, checkpoint sharding, mixed precision, gradient clipping,
CPU/NVMe offload, and ZeRO-3 layer-wise prefetch, communication overlap, or
layer-at-a-time parameter release.
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
