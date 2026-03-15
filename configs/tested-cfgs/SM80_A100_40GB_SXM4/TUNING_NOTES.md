# A100 40GB SXM4 DRAM/L2 Bandwidth Tuning Notes

## Goal

Align accel-sim's DRAM and L2 cache bandwidth with real A100 40GB GPU measurements.

## Reference Hardware Measurements

Microbenchmarks run on real A100 40GB SXM4, transferring 1024 MB data each:

| Benchmark     | Bandwidth (GB/s) | Cycles @ 1.41GHz |
|---------------|-------------------|-------------------|
| dram_read     | 1398              | 1,008,594         |
| dram_write    | 1440              | 978,981           |
| dram_copy     | 1262              | 1,117,272         |
| l2cache_bw    | 3212              | 471,392           |

A100 40GB theoretical peak DRAM BW = 40 channels * 16 B * 1215 MHz * 2 (DDR) = 1555.2 GB/s.

## Iteration History

### Original Config (accel-sim stock SM80_A100)

Timing from accel-sim's default SM80_A100 config, adjusted for 40GB clock (1215 MHz DRAM):

```
RCD=22  CL=22  RAS=50  RP=22  RC=72  CCDL=4  RRD=7  WL=4  CDLR=5  WR=19  RTPL=7
dram_latency=190  l2_rop_latency=200
Address mapping: RBBBCCCC.BCCSSSSS
Separate write queue: disabled
```

Results:

| Benchmark  | Sim Cycles  | Real Cycles | Ratio  | DRAM bwutil |
|------------|-------------|-------------|--------|-------------|
| dram_read  | 2,811,078   | 1,008,594   | 2.79x  | 0.346       |
| dram_write | 1,125,442   | 978,981     | 1.15x  | 0.831       |
| dram_copy  | 2,743,403   | 1,117,272   | 2.45x  | 0.347       |
| l2cache_bw | 513,216     | 471,392     | 1.09x  | —           |

**Diagnosis**: Read/copy bandwidth far too low. Per-channel DRAM stats showed:
- `RCDc_limit` (row activation delay) and `CCDLc_limit` (bank group delay) were
  the dominant bottlenecks, together causing ~50% of DRAM cycles to be idle NOPs.
- Write was much better because L2 writeback coalescing gave better row locality
  (7.2 writes/activation vs 2.9 reads/activation), and `tRCDWR = RCD - WL - 1`
  is smaller than `tRCD` for reads.

### Plan A: Conservative Timing Reduction

Reduced timing to HBM2e typical values (~14ns @ 1215 MHz = ~17 cycles):

```
RCD=14  CL=14  RAS=33  RP=14  RC=47  CCDL=2  RRD=4  WL=3  CDLR=3  WR=12  RTPL=4
dram_latency=140  l2_rop_latency=180
```

Results:

| Benchmark  | Sim Cycles  | Ratio  | bwutil |
|------------|-------------|--------|--------|
| dram_read  | 1,660,509   | 1.65x  | 0.586  |
| dram_write | 954,096     | 0.97x  | 0.979  |
| dram_copy  | 1,728,250   | 1.55x  | 0.551  |
| l2cache_bw | 513,272     | 1.09x  | —      |

**Observation**: Write nearly perfect. Read/copy improved but still far off.
`RCDc_limit` still dominant for reads; `RTWc_limit` (read-to-write turnaround)
emerged as the dominant bottleneck for copy.

### Plan B: Aggressive Timing + Separate Write Queue

Further reduced timing and enabled separate write queue to batch writes:

```
RCD=11  CL=11  RAS=26  RP=11  RC=37  CCDL=1  RRD=3  WL=2  CDLR=2  WR=10  RTPL=2
dram_latency=100  l2_rop_latency=160
Separate write queue: ENABLED (64:56:32)
```

Results:

| Benchmark  | Sim Cycles  | Ratio  | bwutil |
|------------|-------------|--------|--------|
| dram_read  | 1,301,105   | 1.29x  | 0.748  |
| dram_write | 954,476     | 0.97x  | 0.979  |
| dram_copy  | 1,172,631   | 1.05x  | 0.812  |
| l2cache_bw | 513,004     | 1.09x  | —      |

**Observation**: Copy now nearly aligned thanks to separate write queue reducing
RTW/WTR turnaround count by ~85%. Read still 1.29x off — remaining bottleneck
was `RCDc_limit` driven by poor row hit rate (2.75 reads per row activation).

### Plan C: Two Variants Tested

**C1 — Even lower RCD/CL (RCD=9, CL=9)**: Surprisingly made read *worse* (1.35x).
The more aggressive timing altered scheduler behavior, actually increasing row
activations (313K vs 305K) and decreasing row hit rate.

**C2 — Changed address mapping only** (timing identical to Plan B):

```
Address mapping: RBBBCCCB.CCCSSSSS  (interleaved bank bit)
```

Results:

| Benchmark  | Sim Cycles | Ratio      | bwutil |
|------------|------------|------------|--------|
| dram_read  | 987,014    | **0.98x**  | 0.986  |
| dram_write | 954,476    | **0.97x**  | 0.979  |
| dram_copy  | 1,172,631  | **1.05x**  | 0.812  |
| l2cache_bw | 513,004    | **1.09x**  | —      |

**The address mapping was the single most impactful change for read bandwidth.**
Row activations dropped from 305K to 192K per channel (-37%), and row hit rate
jumped from 2.75 to 4.37 reads per activation. `RCDc_limit` dropped from 656K
to 20K — essentially eliminated.

## Final Config = Plan B timing + C2 address mapping

This is what `gpgpusim.config` in this directory now contains.

## Key Parameter Explanations

### DRAM Timing Parameters

All timing values are in DRAM clock cycles (@ 1215 MHz for A100 40GB).

| Parameter | Value | Nanoseconds | Description |
|-----------|-------|-------------|-------------|
| nbk       | 16    | —           | Banks per channel |
| nbkgrp    | 4     | —           | Bank groups (4 banks per group) |
| CCD       | 1     | 0.8 ns      | Column-to-Column Delay (different bank group): min gap between column commands |
| CCDL      | 1     | 0.8 ns      | CCD Long (same bank group): min gap between column commands in same bank group |
| RCD       | 11    | 9.1 ns      | Row-to-Column Delay: latency from row activation to first column command |
| CL        | 11    | 9.1 ns      | CAS Latency: delay from read command to data on bus |
| WL        | 2     | 1.6 ns      | Write Latency: delay from write command to data expected on bus |
| RAS       | 26    | 21.4 ns     | Row Active Strobe: minimum time a row must stay active |
| RP        | 11    | 9.1 ns      | Row Precharge: time to close a row before opening a new one |
| RC        | 37    | 30.5 ns     | Row Cycle: full cycle time for a row (should be >= RAS + RP) |
| RRD       | 3     | 2.5 ns      | Row-to-Row Delay: min gap between activations to different banks |
| WR        | 10    | 8.2 ns      | Write Recovery: time after write completes before precharge |
| CDLR      | 2     | 1.6 ns      | Column Delay Read (to different row): affects write-to-read turnaround |
| RTPL      | 2     | 1.6 ns      | Read-To-Precharge Latency: min time from read to precharge |

### Derived Parameters (computed by simulator)

These are NOT set in config — they are calculated in `gpu-sim.h` from the above:

```c
tRTW   = CL + BL/data_command_freq_ratio + 2 - WL    // Read-to-Write turnaround
       = 11 + 2/2 + 2 - 2 = 12 cycles (9.9 ns)

tWTR   = WL + BL/data_command_freq_ratio + tCDLR      // Write-to-Read turnaround
       = 2 + 2/2 + 2 = 5 cycles (4.1 ns)

tRCDWR = tRCD - (WL + 1)                              // RCD for write operations
       = 11 - 3 = 8 cycles (6.6 ns)

tWTP   = WL + BL/data_command_freq_ratio + tWR         // Write-to-Precharge
       = 2 + 2/2 + 10 = 13 cycles (10.7 ns)
```

`tRTW` and `tWTR` are global per-channel penalties — when the bus switches
direction, ALL banks in that channel are blocked for that many cycles.

### Address Mapping

```
dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCB.CCCSSSSS
```

- `S` (5 bits): Sub-partition select (32-byte sector within 128B cache line)
- `C` (6 bits): Column bits — select column within an open row
- `B` (4 bits): Bank select — which of 16 banks
- `R` (remaining): Row address

The key insight is the **interleaved bank bit** (`RBBBCCCB.CCCSSSSS`): one bank
bit is placed between column bits rather than grouping all bank bits together.
This causes sequential addresses to spread across different banks while staying
in the same row, dramatically improving row buffer hit rate for streaming access
patterns (4.37 reads/activation vs 2.75 with contiguous bank bits).

### Separate Write Queue

```
-dram_seperate_write_queue_enable 1
-dram_write_queue_size 64:56:32    # total:high_watermark:low_watermark
```

When enabled, write requests accumulate in a dedicated queue. The scheduler
drains writes in batches once the queue fills past the high watermark (56),
continuing until it drops below the low watermark (32). This dramatically
reduces read-write bus turnaround events.

Impact on copy workload: `RTWc_limit` dropped from 1.96M to 299K (-85%).

### Latency Parameters

| Parameter         | Value | Unit         | Description |
|-------------------|-------|--------------|-------------|
| gpgpu_l2_rop_latency | 160 | core cycles | Pipeline latency from L2 to memory partition ROP stage |
| dram_latency      | 100   | core cycles  | Fixed latency added to all DRAM accesses |

These are in **core clock** cycles (@ 1410 MHz), not DRAM cycles.

## Why Timing Below Nominal HBM2e Spec is Justified

The final DRAM timing values (~9 ns) are about 35% below the nominal HBM2e
specification (~14 ns). This is intentional and necessary because accel-sim's
DRAM model lacks several optimizations present in real A100 hardware:

1. **No read prefetching**: Real GPU memory controllers speculatively prefetch
   sequential cache lines, effectively hiding row activation latency. The
   simulator issues reads purely on demand.

2. **Simpler scheduling**: The simulator uses FR-FCFS (First Ready, First Come
   First Served) with a 64-entry queue. Real HBM2e controllers likely have
   larger reorder buffers and more sophisticated scheduling algorithms that
   better exploit bank-level parallelism.

3. **No write coalescing at DRAM controller level**: While L2 writeback provides
   some coalescing for writes, reads have no equivalent mechanism. Real
   controllers may coalesce or reorder reads more aggressively.

4. **Simplified bank state machine**: The simulator models basic open/closed row
   states. Real HBM2e may have optimizations like speculative precharge,
   adaptive open-page policies, or per-bank prefetch buffers.

Using slightly aggressive timing parameters is a standard calibration technique
in architectural simulation — it compensates for missing microarchitectural
details to achieve correct aggregate behavior. The alternative (e.g., LUTensor's
approach of inflating bus width and channel count) distorts the hardware topology,
which can cause incorrect behavior for workloads sensitive to memory layout.

## Comparison with LUTensor Config (A100 80GB)

| Aspect | This Config | LUTensor |
|--------|-------------|----------|
| Memory channels | 40 (correct for 40GB) | 64 (incorrect) |
| Bus width | 16 B (correct) | 20 B (incorrect) |
| Burst length | 2 (correct for HBM2) | 4 (incorrect) |
| DRAM clock | 1215 MHz (correct for 40GB) | 1512 MHz (80GB value) |
| Theoretical peak BW | 1555 GB/s (matches spec) | 3871 GB/s (2.5x over spec) |
| Timing in ns | ~9 ns | ~6 ns |
| Address mapping | Interleaved bank bit | Interleaved bank bit (same) |
| Approach | Correct topology, calibrated timing | Distorted topology to brute-force BW |

Both configs independently arrived at the same address mapping, validating its
importance. However, LUTensor chose to inflate the hardware topology (more
channels, wider bus) to achieve target bandwidth, while this config keeps the
topology accurate and achieves alignment through timing calibration and
scheduling features (separate write queue).

## Files in This Directory

- `gpgpusim.config` — **Final tuned config** (use this)
- `gpgpusim.config.bak` — Backup of the original untuned config (if preserved)
- `gpgpusim-tuned.config` — Plan A: conservative timing adjustment
- `gpgpusim-tuned-B.config` — Plan B: aggressive timing + separate write queue
- `gpgpusim-tuned-C1.config` — Plan C1: even lower RCD/CL (worse, not recommended)
- `gpgpusim-tuned-C2.config` — Plan C2: Plan B + interleaved address mapping (= final)
- `TUNING_NOTES.md` — This file
