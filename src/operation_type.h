// v2 shim: MICRO 2025 port code includes "operation_type.h" for the
// uarch_op_t / op_type enum.  In our v2 layout the enum lives in
// abstract_hardware_model.h (extended with the MICRO 2025 ops); this
// header forwards there.  Include guards prevent double-definition.

#pragma once
#include "abstract_hardware_model.h"
