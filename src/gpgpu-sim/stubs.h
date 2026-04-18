// v2 stubs for MICRO 2025 port classes that live outside the paper's
// "remodeling" scope.  These let shader_core_wrapper.h and the ported
// remodeling/ code compile; runtime calls into these stubs never execute
// in Stage 1/2 (they sit behind is_micro2025_arch_enabled+is_loog_enabled
// toggles that default off).
//
// When Stage 3 ships, each stub is either:
//   - deleted (if we formally drop the feature), or
//   - replaced with the real port from MICRO 2025 / fusedMemory /
//     Iliakis-2022 LOOG paper.

#pragma once

// LOOG (Huerta 2023 "Simple Out-of-Order Core for GPGPUs" + RRS rename
// register storage).  Explicitly incompatible with remodeling per
// MICRO 2025 sm.cc:1025; get_loog_rrs() returns nullptr in v2.
class RRS {};

// MICRO 2025 fusedMemory inter-warp coalescing stats.  Not ported in
// Stage 1b; gather_*_stats call sites receive these as by-reference
// parameters but never dereference them along the default path.  Methods
// below are no-op stubs (addStats / registerInst) sufficient for compile.
class warp_inst_t;
class coalescingStatsPerSm {};
class coalescingStatsAcrossSms {
 public:
  void addStats(coalescingStatsPerSm * /*per_sm*/) {}
};
