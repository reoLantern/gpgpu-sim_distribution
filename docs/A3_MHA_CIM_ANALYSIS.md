# A3 MHA CIM Analysis

## Kernel: A3 MHA (Flash Attention)

- Shape: B=1, H=32, sq=4096, skv=4096, d=128, fp16
- Block: bM=128, bN=128, ns=2, block_dim=256
- grid=(32,32,1), shmem=160KB, nregs=210
- Occupancy: 1 CTA × 8 warps = 8 warps/SM (12.5%)

## Real A100 Results

| | Time | Cycles (@1.41GHz) | TFLOPS |
|---|---|---|---|
| Baseline | 0.96 ms | 1,353,600 | 143 |
| CIM skip-b | 0.77 ms | 1,085,700 | 178 |
| **CIM speedup** | | | **19.8%** |

## Simulator Results (perfmem)

| Config | Baseline | CIM | CIM speedup |
|--------|----------|-----|-------------|
| Original (oc_ex=4, cu=16) | 1,273,721 | 1,200,454 | **5.8%** |
| wide_ocex (oc_ex=8) | 1,273,721 | 1,200,454 | **5.8%** |
| more_cu (cu=32) | 1,270,225 | 1,181,573 | **7.0%** |

## Sim vs Real Comparison

```
                  Real A100        Simulator      Ratio
Baseline:         1,353,600        1,273,721      0.94x (sim faster!)
CIM:              1,085,700        1,200,454      1.11x (sim slower)
CIM savings:        267,900           73,267      3.7x less savings
```

Key: simulator baseline is 6% FASTER than real (perfmem effect),
but simulator CIM is 11% SLOWER than real. The problem is that
CIM's LDSM savings are not translating into faster execution.

## Instruction Breakdown (warp 0, TB 0,0,0)

| Instruction | Baseline | CIM | Change |
|-------------|----------|-----|--------|
| HMMA.16816 | 256 | 256 | 0% |
| LDSM.M88.4 | 72 | 8 | -89% (Q only) |
| LDSM.MT88.4 | 64 | 0 | -100% (K/V removed) |
| **Total LDSM** | **136** | **8** | **-94%** |
| FMUL (softmax) | 130 | 130 | 0% |
| FMNMX (softmax) | 72 | 72 | 0% |
| MUFU (softmax) | 68 | 68 | 0% |
| FFMA (softmax) | 68 | 68 | 0% |
| FADD (softmax) | 68 | 68 | 0% |

CIM removes 94% of LDSM (K/V loads), keeps Q loads.
HMMA count identical. Softmax ops identical.

## LDSM-HMMA Interleaving Pattern

Baseline: heavily interleaved
```
5 LDSM → 1 HMMA → 1 LDSM → 5 HMMA → 1 LDSM → 1 HMMA → ...
```

CIM: long HMMA bursts
```
1 LDSM → 8 HMMA → 1 LDSM → 7 HMMA → ... → 173 HMMA burst!
```

## Tensor Core Profiling

| Metric | Baseline | CIM | Change |
|--------|----------|-----|--------|
| tc_issued | 34.6M (7.1%) | 34.6M (7.5%) | same count |
| tc_scoreboard | 92.3M (19.0%) | 15.0M (3.3%) | **-84%** |
| tc_idoc_full | 43.1M (8.9%) | 161.5M (35.2%) | **+275%** |
| tc_no_tensor_warp | 316.2M (65.0%) | 247.3M (53.9%) | -22% |

### Scoreboard Breakdown

| | Baseline | CIM |
|---|---|---|
| tc_sb_A (LDSM) | 9.2M | 22.7M |
| tc_sb_B | 1.1M | 0.15M |
| tc_sb_C (prev HMMA) | 1.9M | 0.14M |
| **tc_sb_other** | **104.8M** | **2.3M** |

## Root Cause

CIM eliminates scoreboard stalls (19% → 3.3%), but `tc_idoc_full`
explodes (8.9% → 35.2%). The freed scheduler cycles don't convert
to more HMMA throughput — they pile up at the OC entry.

The OC pipeline (ID_OC → CU → OC_EX → FU) has:
- ID_OC: 1 slot per subcore (locked to num_sched)
- CU: 4 per subcore (operand collection ~6-8 cycles)
- OC_EX: 1 per subcore
- FU: accepts 1 HMMA every 8 cycles (II=8)

With 2 warps per subcore both in HMMA phase (CIM removes LDSM gaps),
they compete for the single ID_OC slot → tc_idoc_full.

Increasing OC_EX width or CU count has minimal effect (5.8% → 7.0%).
The fundamental issue is that the OC pipeline adds latency that real
hardware (which has no OC, per micro2025) does not have.

## TODO

- [ ] Consider bypassing OC entirely for tensor instructions (most
  impactful, closest to real hardware behavior)
- [ ] Or reduce OC operand collection time for tensor (fewer RF reads
  needed since operands are large register blocks)
- [ ] Validate with realmem + burst=2 after OC fix
