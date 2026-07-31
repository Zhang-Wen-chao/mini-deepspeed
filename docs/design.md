# Design: why the ZeRO stages differ

For `P` parameter elements trained by `N` data-parallel ranks, Adam retains
two optimizer tensors (`m` and `v`) per parameter. Ignoring activations and
temporary buffers, the persistent model-state accounting is:

| Mode | Per-rank persistent state |
| --- | --- |
| Stage 0 | `P` parameters + `P` gradients + `2P` Adam = `4P` |
| Stage 1 | `P` parameters + `P` gradients + `2P/N` Adam |
| Stage 2 | `P` parameters + `P/N` gradients + `2P/N` Adam |
| Stage 3 | `P/N` parameters + `P/N` gradients + `2P/N` Adam = `4P/N` |

The implementation pads the final flat shard when `P` is not divisible by
`N`. Stage 2 partitions complete parameter buckets, so its runtime
`ZeroReport` sums the concrete equal-shard allocation of each bucket; each
bucket can contribute final-shard padding. The table intentionally shows the
unpadded model for clarity.

`mini-deepspeed` keeps full parameters for Stages 0-2, because every rank
needs a complete forward model. At update time, each rank owns one flat
parameter shard. For Stage 1 the full average gradient is available. Only the
owner updates its parameter shard. For Stage 2 the owner receives only its
averaged gradient shard via reduce-scatter. Both stages all-gather updated
parameter shards after the local AdamW update. At engine construction, rank
0's parameter vector is broadcast once so that all data-parallel replicas
start from the same weights, matching DDP's initialization behavior.

This is separate from model parallelism. TP and PP change which computation
graph resides on a device. ZeRO changes which training states are stored there.

## Stage-3 parameter lifecycle

Stage 3 starts by broadcasting rank 0's complete vector once, then retains
only each rank's padded parameter shard plus matching Adam state. An
`engine.forward()` all-gathers those shards and makes every model parameter a
view into the temporary flat vector. `engine.backward(loss)` checks that every
trainable parameter participated, reduce-scatters the averaged gradient to its
owner, and releases all complete parameter and gradient tensors. `step()`
updates only the local parameter shard.

This makes `P/N` parameter ownership observable between iterations. It is not
production ZeRO-3 scheduling: the educational implementation gathers the
entire model for every forward/backward pair, performs no prefetch or overlap,
and does not release one layer before the next layer runs. Its API intentionally
requires exactly one engine forward before each engine backward. A diagnostic
`parameter_vector()` call gathers only a detached inspection copy; it does not
change persistent ownership.

During a normal engine-mediated forward, a module exception is synchronized so
that every rank releases its materialization and raises. Likewise, after a
backward each rank synchronizes whether it saw an exception, a missing
trainable gradient, or pre-existing `.grad` tensors left by a backward that
ran outside the engine, before any reduce-scatter begins. The resulting
invalidated window is recoverable only when **every** rank calls `zero_grad()`
and resumes the same schedule. This prevents the common case where one rank
raises locally while peers block in a later collective; it cannot make
arbitrary rank-divergent engine calls safe, which is a fundamental constraint
of synchronous collectives.

There is no Stage-3 checkpoint API. State-dict save/load is explicitly
rejected on the engine, on the module, and on every submodule while no sharded
format exists, instead of serializing empty placeholders between iterations.
That guard is registered as `state_dict` pre-hooks on all submodules; pickling
the module (`torch.save(module)`) or `copy.deepcopy(module)` does not go
through `state_dict` and would serialize the empty placeholders, so those
paths are documented as unsupported between iterations. The flat layout supports normal weight tying where PyTorch exposes one
`Parameter` object once through `module.parameters()`. It rejects distinct
`Parameter` objects that are views or share storage in every stage: the flat
vector owns and updates each parameter independently, so the shared region
would end up with only the last write-back, silently dropping the other
parameter's gradient contribution (`torch.optim.AdamW` compounds both in-place
updates), and Stage 3 would break the aliases outright by replacing parameter
storage. Non-contiguous parameters that own their full storage remain
supported because the flat vector stores logical row-major values.

## Stage-2 bucket lifecycle

Stage 2 partitions consecutive complete parameters into buckets. A
post-accumulate-gradient hook records each parameter's readiness during
`engine.backward(loss)`. When the last gradient in a bucket arrives, it:

1. flattens and pads the bucket gradients;
2. performs native NCCL `reduce_scatter_tensor` (or Gloo all-reduce plus slice
   as a labelled correctness fallback);
3. retains and averages only this rank's gradient shard; and
4. immediately sets every full `param.grad` in the bucket to `None`.

`step()` updates the matching local parameter shard with its local AdamW state,
then all-gathers the updated parameter shards so every rank still has a full
forward model. Multiple `engine.backward()` calls before one `step()` add raw
gradient sums, and every complete bucket must participate in every backward.
Calling `zero_grad()` discards any retained Stage-2 gradient shards and starts
a fresh accumulation window. A failed or incomplete backward invalidates that
window; the caller must call `zero_grad()` before another `backward()` or
`step()`.

## Scope boundary

This lifecycle is observable and is validated against real DeepSpeed, but it
is not a CUDA allocator peak-memory proof. Autograd still creates individual
parameter gradients; flattening introduces temporary buffers; there is no
communication overlap; and a parameter larger than the configured bucket size
remains a one-parameter bucket. The report measures retained state after the
hook lifecycle, not the maximum allocator watermark. Allocator telemetry and
more scheduling work are required before a physical peak-memory claim. ZeRO-3
does shard parameters here, but its full-model eager gather still prevents a
layer-wise peak-memory claim.
