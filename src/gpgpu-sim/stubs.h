// v2 stubs for MICRO 2025 port classes that are intentionally dormant.
// After Stage 1e-B1/B2 this file contains only the RRS stub (LOOG); the
// former coalescingStats stubs have been replaced with the real classes
// from remodeling/fusedMemory/coalescingStats.h.

#pragma once

// LOOG (Huerta 2023 "Simple Out-of-Order Core for GPGPUs" + RRS rename
// register storage).  Explicitly incompatible with remodeling per
// MICRO 2025 sm.cc:1025; get_loog_rrs() returns nullptr in v2.
class RRS {};

// Stage 1e-B1: coalescingStatsPerSm / coalescingStatsAcrossSms are now the
// real classes ported from MICRO 2025 remodeling/fusedMemory/.  Forward-
// declared here so headers that only need pointers/references (e.g.
// shader_core_wrapper.h) keep compiling lightly; full definition is in
// remodeling/fusedMemory/coalescingStats.h.
class coalescingStatsPerSm;
class coalescingStatsAcrossSms;
