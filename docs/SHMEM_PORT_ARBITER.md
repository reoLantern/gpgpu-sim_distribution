# Shared Memory Port Arbiter (`shmem_port_arbiter`)

## Motivation

On real NVIDIA GPUs, shared memory has limited per-bank bandwidth that is
shared between multiple access sources:

1. **Pipeline reads/writes** -- LDSM, LDS, STS instructions issued by warps
   through the normal ldst_unit pipeline.
2. **Async writes** -- LDGSTS (cp.async) data returning from global memory and
   being written into shared memory banks.
3. **(Future) TMA loads** -- Tensor Memory Accelerator transfers that write
   bulk data from global/L2 into shared memory.
4. **(Future) WGMMA reads** -- Warp-group MMA instructions that read operands
   directly from shared memory (Hopper+).

The original accel-sim models LDGSTS as a pure global memory load: when the
data returns from the memory hierarchy, it is "completed" by decrementing a
counter.  The shared memory write is **not modeled** -- there is no bank
occupancy, no bandwidth consumption, and no contention with concurrent LDSM
reads.

This means CIM kernels that eliminate half of their LDSM instructions show
almost no speedup in the simulator (~2%), whereas real A100 hardware measures
10-20% improvement.  The missing shared memory contention is a major
contributor to this gap.

## Design

`shmem_port_arbiter` is a lightweight per-SM component (member of `ldst_unit`)
that tracks shared memory bandwidth consumed by async write sources.

### Interface

```cpp
class shmem_port_arbiter {
    void init(unsigned num_banks, unsigned bank_width_bytes);
    void new_cycle();                      // drain 1 cycle of write work
    void async_write(unsigned num_bytes);  // register arriving async data
    bool is_write_active() const;          // pipeline contention query
};
```

### Two access modes (designed for extensibility)

| Mode | When to use | Example |
|------|-------------|---------|
| `async_write(bytes)` | Async data arrives, exact destination banks unknown | LDGSTS writeback, future TMA writeback |
| (future) `access_banks(bank_ids)` | Exact bank IDs known | Future WGMMA shmem reads, or if LDGSTS destination addresses become available |

Currently only `async_write` is implemented.  When destination bank
information becomes available (e.g. through extended traces or register
tracking), the arbiter can be extended with bank-level arbitration.

## Integration points

### 1. Initialization (`ldst_unit::init`)

```cpp
m_shmem_arbiter.init(m_config->num_shmem_bank, 4 /* bytes/bank/cycle */);
```

### 2. Cycle tick (`ldst_unit::cycle`, top)

```cpp
m_shmem_arbiter.new_cycle();  // drains one cycle of async write occupancy
```

### 3. LDGSTS writeback (`ldst_unit::writeback`)

When a LDGSTS `mem_fetch` returns from global memory and transitions into
`m_next_wb_ldgsts`:

```cpp
unsigned wb_bytes = m_next_global->get_data_size();
m_shmem_arbiter.async_write(wb_bytes);
```

Each LDGSTS.128 instruction generates ~8 mem_fetches (coalesced to 64-byte
segments).  Each 64-byte return adds `ceil(64 / 128) = 1` cycle of shared
memory write occupancy.

### 4. Pipeline read contention (`ldst_unit::shared_cycle`)

After `dispatch_delay()` completes (bank-conflict cycles consumed), check
for async write contention:

```cpp
if (!stall && m_shmem_arbiter.is_write_active()) {
    stall = true;   // extra stall while async writes occupy shmem
}
```

This stall persists every cycle until `m_async_write_cycles` reaches 0.

## Adding new async sources (TMA, WGMMA, etc.)

To add a new source of shared memory contention:

1. **Find the data arrival point** -- where the data transitions from the
   memory hierarchy into the SM and needs to be written to shared memory.

2. **Call `m_shmem_arbiter.async_write(bytes)`** at that point with the
   number of bytes being written.

3. **No other changes needed** -- the arbiter automatically creates
   contention with pipeline LDSM/STS instructions.

For sources that read from shared memory (e.g. WGMMA), extend the arbiter
with an `access_banks()` method that checks per-bank occupancy from both
pipeline and async sources.

## Known limitations

- Async write contention is modeled at bandwidth level (bytes/cycle), not
  at individual bank level, because LDGSTS destination addresses are not
  recorded in traces.
- The EX_WB (writeback) stage is currently SM-wide shared.  The micro2025
  paper (Huerta et al.) shows each sub-core has an independent result queue
  and RF write port.  This causes artificial scoreboard pressure on tensor
  core instructions.  See TODO below.

## TODO

- [ ] Per-subcore writeback: split `m_pipeline_reg[EX_WB]` into per-subcore
  result queues to match real hardware (micro2025 Figure 3).
- [ ] Bank-level arbitration when destination addresses are available.
- [ ] TMA integration for Hopper+ architectures.
- [ ] WGMMA shared memory read contention for Hopper+.
