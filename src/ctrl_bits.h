// Per-PC control bits from NVBit v1.8 getSassBinary().
// Shared between trace parser and simulator.
//
// Packed as 23-bit value (instruction bits[105:127], stall in LSB):
//   [3:0]   stall   (4 bits, 0-15)
//   [4]     yield   (1 bit, hw: 1=no-yield, 0=yield)
//   [7:5]   r-bar   (3 bits, 7=none)
//   [10:8]  w-bar   (3 bits, 7=none)
//   [16:11] b-mask  (6 bits, wait-barrier bitmask)
//   [20:17] reuse   (4 bits, RFC reuse flags)

#ifndef CTRL_BITS_H
#define CTRL_BITS_H

#include <stdint.h>

struct ctrl_bits_t {
  uint32_t raw;
  ctrl_bits_t() : raw(0) {}
  explicit ctrl_bits_t(uint32_t v) : raw(v) {}
  unsigned stall() const { return raw & 0x0Fu; }
  unsigned yield_raw() const { return (raw >> 4) & 0x01u; }
  bool is_yield() const { return !yield_raw(); }
  unsigned r_bar() const { return (raw >> 5) & 0x07u; }
  unsigned w_bar() const { return (raw >> 8) & 0x07u; }
  unsigned b_mask() const { return (raw >> 11) & 0x3Fu; }
  unsigned reuse() const { return (raw >> 17) & 0x0Fu; }
  bool has_r_bar() const { return r_bar() != 7; }
  bool has_w_bar() const { return w_bar() != 7; }
  bool valid() const { return raw != 0; }
};

#endif  // CTRL_BITS_H
