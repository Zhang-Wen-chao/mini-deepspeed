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
`N`. Its runtime `ZeroReport` includes that concrete equal-shard allocation;
the table intentionally shows the unpadded model for clarity.

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

## Scope boundary

The implementation prioritizes transparent semantics over production-grade
memory scheduling. Autograd still produces a per-parameter gradient before the
update boundary, and the demonstration flattens it before calling a collective.
Therefore the report measures the states retained after synchronization, not
CUDA's short-lived backward peak. A production ZeRO-2 engine uses gradient
buckets and autograd hooks to reduce-scatter each bucket as soon as it is ready.
ZeRO-3 adds layer-wise parameter all-gather and release.

Those lifecycle mechanisms are the next educational milestones. Claiming that
this first version reaches DeepSpeed's physical peak-memory efficiency would
be inaccurate.
