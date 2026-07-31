# Design: why the three stages differ

For `P` parameter elements trained by `N` data-parallel ranks, Adam retains
two optimizer tensors (`m` and `v`) per parameter. Ignoring activations and
temporary buffers, the persistent model-state accounting is:

| Mode | Per-rank persistent state |
| --- | --- |
| Stage 0 | `P` parameters + `P` gradients + `2P` Adam = `4P` |
| Stage 1 | `P` parameters + `P` gradients + `2P/N` Adam |
| Stage 2 | `P` parameters + `P/N` gradients + `2P/N` Adam |

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
would additionally shard parameters and all-gather a layer only for its
forward/backward use.
