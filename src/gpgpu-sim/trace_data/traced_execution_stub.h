// v2 stub for traced_execution — MICRO 2025's protobuf-trace-execution host.
//
// We deliberately did NOT port their `traced_execution.{h,cc}` (which
// requires protobuf to parse dynamic_trace/*.proto).  Stage 1d will wire
// a NVBit-v1.8-text→traced_instruction adapter; until then, any code
// that calls `gpgpu_sim::get_extra_trace_info()` (inside remodeling SM::*
// methods) is dead under is_SM_remodeling_enabled=0.  This stub keeps
// those call sites compiling.
//
// When Stage 1d lands, replace this header with a real port of MICRO 2025
// traced_execution.{h,cc} (minus the protobuf reader, plus our text adapter).

#pragma once

#include "../../abstract_hardware_model.h"
#include <string>
#include <vector>

class traced_kernel;

// Field names match MICRO 2025 (m_* prefix on the lookup-result fields).
struct search_func_addr_result {
  bool m_has_been_traced = false;
  unsigned int m_unique_function_id = 0;
  address_type function_addr = 0;
};

struct traced_execution_kernel_stub {
  address_type get_function_addr() const { return 0; }
};

class traced_execution {
 public:
  traced_execution() = default;

  search_func_addr_result search_function_addr(address_type /*call_addr*/) {
    return {};
  }

  unsigned int get_unique_function_id(const std::string & /*kernel_name*/) {
    return 0;
  }

  traced_execution_kernel_stub get_kernel_by_unique_function_id(
      unsigned int /*unique_function_id*/) {
    return {};
  }
};
