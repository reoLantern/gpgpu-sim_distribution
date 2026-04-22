# Exclude power_interface.cc if no power model
list(APPEND gpgpusim_SRC addrdec.cc
    dram.cc
    dram_sched.cc
    gpu-cache.cc
    gpu-misc.cc
    gpu-sim.cc
    hashing.cc
    histogram.cc
    icnt_wrapper.cc
    l2cache.cc
    local_interconnect.cc
    mem_fetch.cc
    mem_latency_stat.cc
    power_interface.cc
    power_stat.cc
    result_bus.cc   # Stage 1h pivot B1: ResultBusses extracted to own file
    scoreboard.cc
    scoreboard_reads.cc   # MICRO 2025 port: WAR scoreboard (Stage 1c.4.4)
    shader.cc
    stack.cc
    stat-tool.cc
    traffic_breakdown.cc
    visualizer.cc
    # MICRO 2025 port: Stage 1d.4+5 — traces_enhanced data classes are now
    # compiled as part of the trace-parser static lib (outer CMakeLists).
    # No per-sim-lib copy needed; eliminates duplicate-symbol risk.
    # MICRO 2025 port: remodeling/ (Stage 1c.4).  Compiled but not yet
    # invoked at runtime (is_micro2025_arch_enabled dispatch arrives in 1e).
    remodeling/first_level_instruction_cache.cc
    remodeling/functional_unit.cc
    remodeling/ibuffer_remodeled.cc
    remodeling/l0_icnt.cc
    remodeling/ldst_unit_sm.cc
    remodeling/register_file.cc
    remodeling/sm.cc
    remodeling/stream_buffer.cc
    remodeling/subcore.cc
    remodeling/warp_dependency_state.cc
    # Stage 1e-B1: real coalescingStats port (replaces stubs.h stub classes).
    # MICRO 2025 uses these for inter-/intra-warp coalescing histograms.
    remodeling/fusedMemory/coalescingStats.cc
    # Stage 1e-B2: dormant-but-complete GMMU + PageTableWalker classes.
    # MICRO 2025 never instantiates them; we carry the code for future
    # multi-GPU TLB / Hopper TMA address-translation work.
    remodeling/gmmu.cc
    remodeling/page_table_walker.cc)
if(NOT GPGPUSIM_USE_POWER_MODEL)
    list(REMOVE_ITEM ${gpgpusim_SRC} power_interface.cc)
endif()

# Create libgpgpusim.a
add_library(gpgpusim STATIC ${gpgpusim_SRC})
target_include_directories(gpgpusim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gpgpusim PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
# MICRO 2025 port: traced_instruction.cc transitively includes
# ISA_Def/trace_opcode.h, which itself does bare `#include "abstract_hardware_model.h"`.
# Add src/ so that unqualified include resolves.
target_include_directories(gpgpusim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Enable OpenMP
find_package(OpenMP REQUIRED)
target_compile_options(gpgpusim PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(gpgpusim PUBLIC OpenMP::OpenMP_CXX)

# Stage 1h pivot: accelwattch headers always needed (gpu-sim.h declares
# gpgpu_sim_wrapper* members unconditionally; power_interface.h includes
# gpgpu_sim_wrapper.h unconditionally).  Compile-time define still gated.
target_include_directories(gpgpusim PRIVATE ${GPGPUSIM_POWER_MODEL})
if(GPGPUSIM_USE_POWER_MODEL)
target_compile_definitions(gpgpusim PRIVATE GPGPUSIM_POWER_MODEL)
endif()

