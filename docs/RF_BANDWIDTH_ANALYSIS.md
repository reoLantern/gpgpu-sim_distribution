# Register File Bandwidth Analysis

## Finding: Simulator RF bandwidth is 8x real A100

| | Real A100 | Simulator (default) |
|---|---|---|
| RF banks per subcore | 2 | 8 (32 total / 4) |
| Reads per bank per cycle | 1 | 1 per step × 2 steps |
| **Total reads/cycle/subcore** | **2** | **16** |
| **Bandwidth** | **256 B/cycle/subcore** | **2048 B/cycle/subcore** |

Config: `-gpgpu_num_reg_banks 32`, `-gpgpu_reg_file_port_throughput 2`

Each "read" = 1 register × 32 threads × 4 bytes = 128 bytes.

HMMA/IMMA needs 10 operand reads (A=4, B=2, C=4):
- Real A100: 10 / 2 = **5 cycles** for operand collection
- Simulator: 10 / 16 ≈ **1 cycle**

## Impact on CIM speedup

Tested with `-gpgpu_num_reg_banks 8 -gpgpu_reg_file_port_throughput 1`
(= 256 bytes/cycle/subcore, matching real A100):

### A3 MHA (perfmem)

| RF bandwidth | Baseline | CIM | CIM speedup |
|---|---|---|---|
| 2048 B/cyc/sub | 1,273,721 | 1,200,454 | 5.8% |
| **256 B/cyc/sub** | **1,380,037** | **1,259,369** | **8.7%** |
| Real A100 | 1,353,600 | 1,085,700 | 19.8% |

- RF=256 baseline (1.38M) close to real (1.35M) — calibrated well
- CIM speedup improved 5.8% → 8.7% but still far from 19.8%

### GEMM int8 (perfmem)

| RF bandwidth | Baseline | CIM | CIM speedup |
|---|---|---|---|
| 2048 B/cyc/sub | 636,538 | 522,060 | 18.0% |
| **256 B/cyc/sub** | **690,362** | **622,935** | **9.8%** |
| Real A100 | — | — | 10-20% |

- GEMM speedup DECREASED 18.0% → 9.8% with slower RF
- Slower OC bottlenecks BOTH baseline and CIM equally

### Realmem + burst=2

| Config | Baseline | CIM | CIM speedup |
|---|---|---|---|
| realmem+burst2 (RF=2048) | 1,349,524 | 1,234,304 | 8.5% |
| Real A100 | 1,353,600 | 1,085,700 | 19.8% |

Baseline almost perfectly calibrated (1,350K vs 1,354K).
CIM still 13.7% slower than real.

## Root Cause Analysis

RF bandwidth alone doesn't explain the gap. The fundamental issue is
that the simulator's OC pipeline (ID_OC → CU → OC_EX → FU) doesn't
exist on real hardware. Real A100 uses:

- **No operand collector** (micro2025 confirmed)
- **Allocate stage**: 2-3 fixed cycles for RF read, all instructions
- **No CU queuing**: instruction goes directly to FU

The OC creates artificial coupling between RF bandwidth and instruction
throughput. On real hardware, the RF read (Allocate) is a fixed-latency
pipeline stage, not a variable-latency queue.

## Why Tensor OC Bypass Would Help

### For GEMM

Current (RF=2048, OC): IMMA goes through OC in ~3-4 cycles (fast).
FU II=8, so OC < II → OC is NOT the bottleneck → 18% CIM speedup.

With RF=256: IMMA OC time ~7-8 cycles ≈ FU II=8 → OC becomes bottleneck
→ CIM speedup drops to 9.8%.

With tensor bypass: IMMA enters FU in ~1-2 cycles (always < II=8) →
OC never bottlenecks → CIM speedup back to ~18% regardless of RF config.

### For FA/MHA

Current: baseline has LDSM + HMMA + softmax interleaved. Every HMMA
going through OC (even specialized) blocks the ID_OC slot for its
subcore. With 2 warps/subcore both issuing HMMA, tc_idoc_full = 35%.

With tensor bypass:
1. HMMA issues → directly to FU (no ID_OC blocking)
2. Scheduler immediately available for next instruction
3. In baseline: LDSM still takes scheduler slots, still goes through
   shared LDST (4 cycles), still competes for shmem bandwidth
4. In CIM: no LDSM → scheduler slots freed → softmax runs faster →
   total non-HMMA time decreases significantly
5. Key difference: without OC blocking, every LDSM removed = 1+ pure
   scheduler cycle saved (not absorbed by OC queuing)

Expected: CIM speedup should increase significantly for FA because
the OC was masking the scheduler-slot-competition effect of LDSM.

## TODO

- [ ] Implement tensor OC bypass (most impactful)
- [ ] Or reduce OC to fixed 2-3 cycle pipeline for tensor (simpler)
- [ ] Consider extending to all fixed-latency instructions (matches
  micro2025 finding that real hardware has no OC)
