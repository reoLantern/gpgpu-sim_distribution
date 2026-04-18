// v2 stub for MICRO 2025 fusedMemory/coalescingStats.h.
//
// MICRO 2025's InterWarpCoalescingUnit records rich per-SM and across-SM
// coalescing statistics.  Stage 1 of our port does NOT engage that path —
// is_SM_remodeling_enabled defaults off, so these stubs are never executed,
// only compiled.  When Stage 2 needs the real stats system, replace this
// file with a verbatim port of MICRO 2025's 5-class header.

#pragma once

#include "../../../abstract_hardware_model.h"   // _memory_space_t
#include "../../stubs.h"                         // coalescingStatsPerSm/AcrossSms stubs

class SM;

class coalescingWarpStats {};
class coalescingCycleHistory {};

// Real MICRO 2025 class owns per-SM + per-warp + cycle history.
// Stubbed to just satisfy calls from ldst_unit_sm.cc (constructor +
// getStats() + resetHistory() + registerInst()).
class warp_inst_t;
class coalescingAddressStats {
 public:
  coalescingAddressStats(SM * /*sm*/, const char * /*name*/,
                         _memory_space_t /*space*/) {}
  coalescingStatsPerSm *getStats() { return &m_stats; }
  void resetHistory() {}
  void registerInst(unsigned long long /*cycle*/, warp_inst_t * /*inst*/) {}

 private:
  coalescingStatsPerSm m_stats;
};
