# Memory Subsystem Bottleneck Analysis

## Problem

Compute-bound GEMM kernels show 30% more cycles with real memory vs
perfect memory (828K vs 636K cycles). On real A100, async copy (LDGSTS)
fully hides memory latency, so the kernel should be compute-bound
regardless of memory model.

## Mem_fetch Journey: SM → L2 → SM

A complete round trip for one LDGSTS mem_fetch, with model parameters
and real hardware comparison:

```
SM issues LDGSTS mem_fetch
    │
    ▼
[1] SM → ICNT input buffer
    Model:  icnt_in_buffer_limit = 512 entries
    Real:   SM memory unit output queue
    Status: NOT a bottleneck
    │
    ▼
[2] ICNT transport (SM → L2 sub-partition)
    Model:  local_interconnect, 1 hop, ~1-2 cycles
    Real:   A100 crossbar, ~few cycles
    Status: NOT a bottleneck
    │
    ▼
[3] ROP delay queue  ← MAIN BOTTLENECK
    Model:  rop_latency = 160 cycles, applied to ALL requests (hits+misses)
            1 request popped per sub-partition per cycle
    Real:   SM→L2 total pipeline delay ~40-60 cycles (crossbar + L2 tag
            lookup). L2 is pipelined: first result at ~40 cycles, then
            1 result/cycle sustained throughput.
    Status: rop_not_ready = 62.2M (47% of cycles waiting for ROP timer)
    Issue:  160 cycles is 3-4x real hardware latency
    │
    ▼
[4] icnt_L2_queue → L2 cache access
    Model:  queue size = 64, 1 access per data port per cycle
    Real:   L2 multi-bank, parallel access
    Status: l2_port_busy = 388K (1%, negligible)
    │
    ├──[HIT]──────────────────────────────────┐
    │                                         ▼
    │                              [5a] L2_icnt_queue (response)
    │                                   Model:  queue size = 64
    │                                   Status: l2_output_full = 5.5M (13%)
    │                                   Issue:  Response queue fills up,
    │                                           blocks L2 from accepting
    │                                           new hits
    │                                         │
    ├──[MISS]─→ DRAM queue → DRAM ────────────┤
    │   Model:  dram_latency = 100            │
    │   + DRAM timing ~30-40 cycles           │
    │   Real:   HBM2e ~100-200ns              │
    │           ≈ 140-280 cycles @1.41GHz     │
    │   Note:   Only 5% of requests miss L2   │
    │                                         │
    │                                         ▼
    │                              [6] L2_icnt_queue → ICNT (response)
    │                                   Model:  icnt_out_buffer_limit = 512
    │                                   Status: NOT a bottleneck
    │                                         │
    │                                         ▼
    │                              [7] ICNT → Cluster ejection buffer
    │                                   Model:  n_simt_ejection_buffer_size = 32
    │                                   Status: eject_full = 0, NOT a bottleneck
    │                                         │
    │                                         ▼
    │                              [8] Cluster → SM ldst_unit response_fifo
    │                                   Model:  ldst_unit_response_queue_size = 2
    │                                   Status: sm_resp_full = 0, NOT a bottleneck
    │                                           (L2 side already rate-limits)
    └─────────────────────────────────────────┘
```

## Profiling Data (baseline GEMM, realmem, all sub-partitions aggregated)

### ROP Queue

| State | Count | Share | Meaning |
|-------|-------|-------|---------|
| rop_empty | 31.9M | 24% | No pending requests |
| **rop_not_ready** | **62.2M** | **47%** | Requests waiting for 160-cycle timer |
| rop_ready_and_popped | 35.7M | 27% | Successfully forwarded to L2 |
| rop_ready_but_full | 2.8M | 2% | Ready but downstream L2 queue full |

### L2 Access

| State | Count | Share |
|-------|-------|-------|
| l2_proceed | 35.7M | 86% |
| **l2_output_full** | **5.5M** | **13%** |
| l2_port_busy | 388K | 1% |

### Cluster / SM Side

| State | Count | Note |
|-------|-------|------|
| sm_resp_accepted | 35.7M | All accepted |
| sm_resp_full | **0** | Never blocked |
| eject_full | **0** | Never blocked |

## Root Cause

1. **`rop_latency = 160` is 3-4x too high** for compute kernels.
   Real A100 SM→L2 pipeline latency is ~40-60 cycles (inferred from
   micro2025 Table 2: Load Global L2 hit ~29-35 cycles from issue,
   minus SM-internal pipeline ~10-15 cycles).

2. **ROP latency applies to L2 HITS equally** — on real hardware,
   L2 hits return through a pipelined path. The first hit takes ~40
   cycles, subsequent hits sustain 1/cycle throughput per L2 slice.

3. **L2 response queue (64 entries) backs up** because ROP drains
   slowly (1/cycle/sub-partition), while L2 hits produce results
   faster than they can be forwarded.

## Parameter Sensitivity

Reducing latency parameters barely helps (< 2%):

| Config | Cycles | vs perfmem |
|--------|--------|-----------|
| perfmem | 636,538 | — |
| rop=160, dram=100 | 828,073 | +30.1% |
| rop=40, dram=100 | 831,353 | +30.6% |
| rop=160, dram=30 | 821,856 | +29.1% |
| rop=40, dram=30 | 818,886 | +28.6% |

This confirms the bottleneck is **throughput**, not latency. The ROP
queue pops only 1 request per sub-partition per cycle, creating a
throughput ceiling regardless of latency value.

## Perfmem Latency Ablation Study

To isolate latency vs throughput, we added a `gpgpu_perfect_mem_latency`
config option: perfmem mode with a fixed delay before responses arrive.
This gives perfect throughput (unlimited bandwidth) with configurable
latency.

| Config | Cycles | vs perfmem | Note |
|--------|--------|-----------|------|
| perfmem lat=0 (instant) | 636,538 | — | Baseline |
| perfmem lat=100 | 709,125 | +11.4% | 100-cycle round-trip |
| perfmem lat=200 | 713,960 | +12.2% | ~same as lat=100 |
| perfmem lat=400 | 672,087 | +5.6% | Longer delay hides better |
| **realmem** | **828,073** | **+30.1%** | Full memory hierarchy |

**Key findings:**
1. **lat=100 → lat=200: only +0.7%** — async copy (LDGSTS + DEPBAR)
   successfully hides latency beyond ~100 cycles. The compute phase
   (~400 cycles) provides sufficient overlap.
2. **lat=400 < lat=200** — longer delays change warp scheduling behavior,
   sometimes beneficially.
3. **perfmem+lat200 vs realmem: 714K vs 828K (+16%)** — this 16% gap
   is purely from **throughput bottlenecks** in the memory pipeline,
   not latency.

**Conclusion:** The real memory overhead decomposes into:
- ~12% from latency (hidden by async copy if ≤ compute window)
- ~18% from throughput (memory pipeline stages limit requests/cycle)

Usage:
```
-gpgpu_perfect_mem 1
-gpgpu_perfect_mem_latency 200  # cycles of fixed round-trip delay
```

## Recommended Fixes

### Option A: Increase ROP pop throughput (preferred)

Allow ROP queue to pop multiple requests per cycle when they are ready.
Real L2 is multi-banked and can service multiple requests in parallel.

### Option B: Reduce rop_latency

Set `l2_rop_latency = 50` to match real SM→L2 pipeline delay. This
helps latency but doesn't fix the throughput cap.

### Option C: Bypass ROP for L2 hits

L2 hits don't need ROP processing. Add a fast path that skips the
ROP delay for cache hits, sending responses directly to L2_icnt_queue.

### Option D: Increase L2 response queue

Increase `L2_icnt_queue` from 64 to 128+ entries to reduce the 13%
back-pressure from response queue being full.
