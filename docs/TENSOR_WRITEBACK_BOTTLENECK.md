# Tensor Core Scoreboard: Timed Release Fix

## Problem

Tensor core (IMMA/HMMA) instructions suffered from artificially high
scoreboard stalls in accel-sim, causing CIM speedup to be severely
underestimated (~5.9% sim vs 10-20% real A100).

## Root cause analysis

### Profiling data (CIM GEMM, perfmem=1, SM0 warp0, 3889 stall events)

| Stall cause | Count | Share |
|-------------|-------|-------|
| **C/D accumulator** (previous IMMA output not written back) | 2,343 | **60.2%** |
| **A operand** (LDSM shared memory load not written back) | 1,545 | **39.7%** |
| B operand | 0 | 0% |

### Why C/D stalls are artificial

The simulator issues IMMA into an operand collector (OC) pipeline before
the tensor FU.  Multiple IMMAs queue up in OC, causing issue-to-writeback
latency of **500+ cycles** (observed) vs the **12-cycle FU latency**.

```
Timeline (observed):
  cycle 12214: ISSUE dst=R93 (IMMA enters OC pipeline)
  cycle 12672: next IMMA wants R93 → STALL (458 cycles later, R93 still on scoreboard!)
  cycle 12700+: R93 WRITEBACK still hasn't happened (500+ cycles total)
```

On real hardware, this doesn't happen because:

1. **No operand collector** — modern NVIDIA GPUs (micro2025 paper, Huerta et al.)
   do not use OC units.  Dependencies are managed by compiler-set *Stall counters*.

2. **Result queue + bypass** — fixed-latency instructions (FP, INT, Tensor) have
   a result queue per sub-core.  Results are available via bypass after exactly
   `latency` cycles from issue.  The paper states: "when there is a conflict
   between two fixed latency instructions... none of them is delayed" and
   "consumers of these instructions are not delayed, which implies the use of
   bypassing to forward the results before being written in the register file."

3. **Predictable latency** — the compiler knows exactly when IMMA results are
   ready and encodes this via control bits.  No hardware scoreboard needed for
   fixed-latency → fixed-latency dependencies.

## Fix: Timed scoreboard release

For tensor instructions (IMMA/HMMA), the output (D) registers are released
from the scoreboard after a fixed number of cycles (`latency`), regardless
of when the actual RF writeback happens.  Input register dependencies
(e.g., LDSM → A operand) are NOT affected — they still use normal
scoreboard reserve/release.

### Implementation

**Scoreboard** (`scoreboard.h`, `scoreboard.cc`):

```cpp
// New method: reserve output regs with timed auto-release
void reserveRegistersTimedRelease(const warp_inst_t *inst,
                                  unsigned long long current_cycle,
                                  unsigned latency);

// Called once per cycle to process expired timers
void cycle(unsigned long long current_cycle);

// Internal: sorted queue of pending releases
std::deque<timed_release_t> m_timed_releases;
```

**Issue path** (`shader_core_ctx::issue_warp`):

```cpp
if (is_tensor_op(**pipe_reg)) {
    m_scoreboard->reserveRegistersTimedRelease(*pipe_reg, current_cycle,
                                               tensor_latency);
} else {
    m_scoreboard->reserveRegisters(*pipe_reg);  // normal path
}
```

**Cycle hook** (`shader_core_ctx::issue`, top):

```cpp
m_scoreboard->cycle(current_cycle);  // process timed releases
```

**Writeback compatibility**: `releaseRegister()` silently ignores
already-released registers, so the normal writeback path works unchanged.

### Key design choices

1. **Only output registers get timed release** — input dependencies (A from
   LDSM, B from LDSM/CIM) still use normal scoreboard, correctly modeling
   the producer-consumer dependency with variable-latency loads.

2. **Latency value** — uses the instruction's `latency` field (set by config:
   `-trace_opcode_latency_initiation_spec_op_3 "12,8"` → latency=12).
   This matches the real hardware behavior where the compiler sets Stall
   counters based on the known fixed latency.

3. **Sorted deque** — releases are appended in cycle order (since
   `current_cycle + latency` is monotonically increasing for sequential
   issues), so `cycle()` just pops from the front.  O(1) per release.

## Results

### perfmem=1 (isolates pipeline effects)

| Config | Baseline | CIM | CIM speedup |
|--------|----------|-----|-------------|
| Before fix (OC scoreboard) | 642,389 | 604,702 | **5.9%** |
| **After fix (timed release)** | 636,538 | 522,060 | **18.0%** |
| Real A100 | — | — | **10-20%** |

The fix brings CIM speedup from 5.9% to **18.0%**, falling within the
real A100 measurement range of 10-20%.

## Other changes in this commit series

1. **`shmem_port_arbiter`** — models LDGSTS async write contention with LDSM
   reads.  See `docs/SHMEM_PORT_ARBITER.md`.

2. **`specialized_unit::stallable() = true`** — tensor core uses independent
   writeback path, no need to pre-reserve SM-wide result bus.

3. **Tensor profiling** (`ENABLE_TENSOR_PROFILING`) — per-scheduler-cycle
   breakdown of tensor core idle reasons + A/B/C scoreboard stall analysis.
   Compile-time gated, default off.

## Future work

- [ ] Extend timed release to all fixed-latency instructions (FP32, INT32),
  not just tensor, to fully match the micro2025 result bypass model.
- [ ] Replace hardware scoreboard with compiler-managed Stall counters
  (micro2025 Section 4) for the most accurate modeling.
- [ ] Validate with non-perfmem (real memory) configurations.
- [ ] Test with other kernels (MHA, convolution, etc.).
