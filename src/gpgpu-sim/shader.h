// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Andrew Turner,
// Ali Bakhoda, Vijay Kandiah, Nikos Hardavellas, 
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern 
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef SHADER_H
#define SHADER_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <memory>

//#include "../cuda-sim/ptx.tab.h"

#include "../abstract_hardware_model.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "scoreboard_reads.h" // MOD. Fix WAR at baseline.
#include "remodeling/ibuffer_remodeled.h" // MOD. Remodeling
#include "remodeling/warp_dependency_state.h" // MOD. Remodeling
#include "remodeling/l0_icnt.h" // MOD. Added L0I
#include <stack>
#include "stats.h"
#include "traffic_breakdown.h"

#include "shader_core_wrapper.h"
# include <omp.h>

#define NO_OP_FLAG 0xFF

/* READ_PACKET_SIZE:
   bytes: 6 address (flit can specify chanel so this gives up to ~2GB/channel,
   so good for now), 2 bytes   [shaderid + mshrid](14 bits) + req_size(0-2 bits
   if req_size variable) - so up to 2^14 = 16384 mshr total
 */

#define READ_PACKET_SIZE 8

// WRITE_PACKET_SIZE: bytes: 6 address, 2 miscelaneous.
#define WRITE_PACKET_SIZE 8

#define WRITE_MASK_SIZE 8

class gpgpu_context;
class ldst_unit_remake; // MOD. Fixed LDST_Unit model
class coalescingStatsAcrossSms;
class Subcore;

void check_kernel_launch_limitation(
    const kernel_info_t &k, const shader_core_config *shader_config,
    shader_core_stats *stats);

class thread_ctx_t {
 public:
  unsigned m_cta_id;  // hardware CTA this thread belongs

  // per thread stats (ac stands for accumulative).
  unsigned n_insn;
  unsigned n_insn_ac;
  unsigned n_l1_mis_ac;
  unsigned n_l1_mrghit_ac;
  unsigned n_l1_access_ac;

  bool m_active;
};

struct function_call_entry_info {
  function_call_entry_info() {
    unique_function_id = 0;
    active_mask.reset();
  }
  unsigned int unique_function_id;
  active_mask_t active_mask;
};

class shd_warp_t {
 public:
  shd_warp_t(class shader_core_ctx_wrapper *shader, unsigned warp_size, shader_core_stats *stats) 
      : m_shader(shader), m_warp_size(warp_size) {
    m_stores_outstanding = 0;
    m_inst_in_pipeline = 0;
    m_IBuffer_remodeled = new IBuffer_Remodeled(shader->get_config(), this, stats); // MOD. Remodeling
    m_dependency_state = new Dependency_State(shader->get_config(), stats); // MOD. Remodeling
    m_last_unique_inst_id = 0;
    m_kernel_id = 0;
    m_gridbar = false;
    reset();
  }

  virtual ~shd_warp_t() {
    delete m_IBuffer_remodeled; // MOD. Remodeling
    delete m_dependency_state; // MOD. Remodeling
  }

  void reset() {
    assert(m_stores_outstanding == 0);
    assert(m_inst_in_pipeline == 0);
    m_imiss_pending = false;
    m_warp_id = (unsigned)-1;
    m_dynamic_warp_id = (unsigned)-1;
    n_completed = m_warp_size;
    m_n_atomic = 0;
    m_membar = false;
    m_done_exit = true;
    m_last_fetch = 0;
    m_next = 0;
    m_last_unique_inst_id = 0;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;
    while(!m_function_call_stack.empty()) {
      m_function_call_stack.pop();
    }
  }
  void init(address_type start_pc, unsigned cta_id, unsigned wid,
            const std::bitset<MAX_WARP_SIZE> &active,
            unsigned dynamic_warp_id, int shader_id) {
    m_cta_id = cta_id;
    m_warp_id = wid;
    m_dynamic_warp_id = dynamic_warp_id;
    m_next_pc = start_pc;
    assert(n_completed >= active.count());
    assert(n_completed <= m_warp_size);
    n_completed -= active.count();  // active threads are not yet completed
    m_active_threads = active;
    m_done_exit = false;

    // Jin: cdp support
    m_cdp_latency = 0;
    m_cdp_dummy = false;

    m_last_unique_inst_id = 1;

    m_is_pending_store = false; // MOD. Fix load after stores
    m_is_pending_load = false; // MOD. Fix load after stores
  }

  const active_mask_t& get_active_mask() {
    return m_active_threads;
  }

  void push_function_call(unsigned int unique_function_id, active_mask_t active_mask) {
    if(active_mask.any()) {
      function_call_entry_info entry_info;
      entry_info.unique_function_id = unique_function_id;
      entry_info.active_mask = active_mask;
      m_function_call_stack.push(entry_info);
    }
  }

  void pop_function_call(active_mask_t active_mask) {
    assert(!m_function_call_stack.empty());
    m_function_call_stack.top().active_mask ^= active_mask;
    if(m_function_call_stack.top().active_mask.none()) {
      m_function_call_stack.pop();
    }
  }

  unsigned int get_current_unique_function_id_call() {
    assert(!m_function_call_stack.empty());
    return m_function_call_stack.top().unique_function_id;
  }

  bool functional_done() const;
  bool waiting();  // not const due to membar
  bool hardware_done() const;

  bool done_exit() const { return m_done_exit; }

  void set_done_exit() { 
    pop_function_call(m_active_threads);
    m_done_exit = true;
  }

  void print(FILE *fout) const;
  void print_ibuffer(FILE *fout) const;

  void set_scheduler(scheduler_unit* scheduler) { m_scheduler = scheduler; } // MOD. Added L0I
  scheduler_unit* get_scheduler() { return m_scheduler; } // MOD. Added L0I
  bool get_is_pending_store() { return m_is_pending_store; } // MOD. Fix load after stores
  void set_is_pending_store(bool pending) { m_is_pending_store = pending; } // MOD. Fix load after stores
  bool get_is_pending_load() { return m_is_pending_load; } // MOD. Fix load after stores
  void set_is_pending_load(bool pending) { m_is_pending_load = pending; } // MOD. Fix load after stores

  IBuffer_Remodeled* get_IBuffer_remodeled(){ return m_IBuffer_remodeled; } // MOD. Remodeling
  Dependency_State* get_dependency_state(){ return m_dependency_state; } // MOD. Remodeling
  unsigned get_n_completed() const { return n_completed; }
  void set_completed(unsigned lane) {
    assert(m_active_threads.test(lane));
    m_active_threads.reset(lane);
    n_completed++;
  }

  void set_last_fetch(unsigned long long sim_cycle) {
    m_last_fetch = sim_cycle;
  }

  unsigned get_n_atomic() const { return m_n_atomic; }
  void inc_n_atomic() { m_n_atomic++; }
  void dec_n_atomic(unsigned n) { m_n_atomic -= n; }

  bool is_atomic_pending() const { return m_n_atomic > 0; }

  void set_membar() { m_membar = true; }
  void clear_membar() { m_membar = false; }
  bool get_membar() const { return m_membar; }
  void set_gridbar() { m_gridbar = true; }
  void clear_gridbar() { m_gridbar = false; }
  bool get_gridbar() const { return m_gridbar; }
  virtual address_type get_pc() const { return m_next_pc; }
  virtual kernel_info_t* get_kernel_info() const;
  void set_next_pc(address_type pc) { 
    m_next_pc = pc; 
  }

  void store_info_of_last_inst_at_barrier(const warp_inst_t *pI) {
    m_inst_at_barrier = *pI;
  }
  warp_inst_t *restore_info_of_last_inst_at_barrier() {
    return &m_inst_at_barrier;
  }

  void ibuffer_fill(unsigned slot, const warp_inst_t *pI) {
    assert(slot < IBUFFER_SIZE);
    m_ibuffer[slot].m_inst = pI;
    m_ibuffer[slot].m_valid = true;
    m_next = 0;
  }
  bool ibuffer_empty() const {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++)
      if (m_ibuffer[i].m_valid) return false;
    return true;
  }
  void ibuffer_flush() {
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) dec_inst_in_pipeline();
      m_ibuffer[i].m_inst = NULL;
      m_ibuffer[i].m_valid = false;
    }
  }
  const warp_inst_t *ibuffer_next_inst() { return m_ibuffer[m_next].m_inst; }
  bool ibuffer_next_valid() { return m_ibuffer[m_next].m_valid; }
  void ibuffer_free() {
    m_ibuffer[m_next].m_inst = NULL;
    m_ibuffer[m_next].m_valid = false;
  }
  void ibuffer_step() { m_next = (m_next + 1) % IBUFFER_SIZE; }

  bool imiss_pending() const { return m_imiss_pending; }
  void set_imiss_pending() { m_imiss_pending = true; }
  void clear_imiss_pending() { m_imiss_pending = false; }

  bool stores_done() const { return m_stores_outstanding == 0; }
  void inc_store_req() { m_stores_outstanding++; }
  void dec_store_req() {
    assert(m_stores_outstanding > 0);
    m_stores_outstanding--;
  }

  unsigned num_inst_in_buffer() const {
    unsigned count = 0;
    for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
      if (m_ibuffer[i].m_valid) count++;
    }
    return count;
  }
  unsigned num_inst_in_pipeline() const { return m_inst_in_pipeline; }
  unsigned num_issued_inst_in_pipeline() const {
    return (num_inst_in_pipeline() - num_inst_in_buffer());
  }
  bool inst_in_pipeline() const { return m_inst_in_pipeline > 0; }
  void inc_inst_in_pipeline() { m_inst_in_pipeline++; }
  void dec_inst_in_pipeline() {
    assert(m_inst_in_pipeline > 0);
    m_inst_in_pipeline--;
  }

  unsigned get_cta_id() const { return m_cta_id; }

  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  unsigned get_warp_id() const { return m_warp_id; }

  // MOD. Begin IBuffer_ooo debug
  std::map<unsigned,unsigned> pc_incs;
  std::map<unsigned,unsigned> pc_decs;

  void add_inc_pc(unsigned pc)
  {
    std::map<unsigned,unsigned>::const_iterator it = pc_incs.find(pc);
    if(it == pc_incs.end())
    {
      pc_incs[pc] = 1;
    }else {
      unsigned current_incs = it->second;
      pc_incs[pc] = current_incs + 1;
    }
  }

  void add_dec_pc(unsigned pc)
  {
    std::map<unsigned,unsigned>::const_iterator it = pc_decs.find(pc);
    if(it == pc_decs.end())
    {
      pc_decs[pc] = 1;
    }else {
      unsigned current_decs = it->second;
      pc_decs[pc] = current_decs + 1;
    }
  }
  void print_inc_decs() 
  {
    std::cout << "Size pc_incs: " << pc_incs.size() << ", size pc_decs: " << pc_decs.size() << ". PC comp:" <<std::endl;
    std::map<unsigned,unsigned>::const_iterator it, it2;
    unsigned aux_pc, pc_incs_val, pc_decs_val;
    for(it = pc_incs.begin(); it != pc_incs.end(); it++)
    {
      aux_pc = it->first;
      pc_incs_val = it -> second;
      it2 = pc_decs.find(aux_pc);
      if(it2 == pc_decs.end())
      {
        std::cout << "ERROR, pc not found in decs: " << std::hex << aux_pc << std::dec << std::endl;
      }else {
        pc_decs_val = it2->second;
        std::cout << "Inc_Dec. PC: " << std::hex << aux_pc << std::dec << ", incs: " << pc_incs_val << ", decs: " << pc_decs_val << std::endl;
      }
    }
  }
  // MOD. end IBuffer_ooo debug

  class shader_core_ctx_wrapper *get_shader() {
    return m_shader;
  }

 private:
  static const unsigned IBUFFER_SIZE = 2;
  class shader_core_ctx_wrapper *m_shader;
  unsigned m_cta_id;
  unsigned m_warp_id;
  unsigned m_warp_size;
  unsigned m_dynamic_warp_id;

  address_type m_next_pc;
  unsigned n_completed;  // number of threads in warp completed
  std::bitset<MAX_WARP_SIZE> m_active_threads;

  bool m_imiss_pending;

  struct ibuffer_entry {
    ibuffer_entry() {
      m_valid = false;
      m_inst = NULL;
    }
    const warp_inst_t *m_inst;
    bool m_valid;
  };

  warp_inst_t m_inst_at_barrier;
  ibuffer_entry m_ibuffer[IBUFFER_SIZE];
  unsigned m_next;

  unsigned m_n_atomic;  // number of outstanding atomic operations
  bool m_membar;        // if true, warp is waiting at memory barrier
  bool m_gridbar;      // if true, warp is waiting at grid barrier

  bool m_done_exit;  // true once thread exit has been registered for threads in
                     // this warp

  unsigned long long m_last_fetch;

  unsigned m_stores_outstanding;  // number of store requests sent but not yet
                                  // acknowledged
  unsigned m_inst_in_pipeline;

  scheduler_unit *m_scheduler; // MOD. Added L0I
  int m_is_pending_store; // MOD. Fix loads after store
  int m_is_pending_load; // MOD. Fix loads after store
  IBuffer_Remodeled *m_IBuffer_remodeled; // MOD. Remodeling
  Dependency_State *m_dependency_state; // MOD. Remodeling
  // MOD. End. VPREG
  // Jin: cdp support
 public:
  unsigned int m_cdp_latency;
  bool m_cdp_dummy;
  std::stack<function_call_entry_info> m_function_call_stack;
  unsigned long long m_last_unique_inst_id;
  unsigned int m_kernel_id;
  Subcore *m_subcore;
};

inline unsigned hw_tid_from_wid(unsigned wid, unsigned warp_size, unsigned i) {
  return wid * warp_size + i;
};
inline unsigned wid_from_hw_tid(unsigned tid, unsigned warp_size) {
  return tid / warp_size;
};


int register_bank(int regnum, int wid, unsigned num_banks,
                  unsigned bank_warp_shift, bool sub_core_model,
                  int banks_per_sched, unsigned sched_id);

class shader_core_ctx;
class shader_core_config;
class shader_core_stats;

enum scheduler_prioritization_type {
  SCHEDULER_PRIORITIZATION_LRR = 0,   // Loose Round Robin
  SCHEDULER_PRIORITIZATION_SRR,       // Strict Round Robin
  SCHEDULER_PRIORITIZATION_GTO,       // Greedy Then Oldest
  SCHEDULER_PRIORITIZATION_GTLRR,     // Greedy Then Loose Round Robin
  SCHEDULER_PRIORITIZATION_GTY,       // Greedy Then Youngest
  SCHEDULER_PRIORITIZATION_OLDEST,    // Oldest First
  SCHEDULER_PRIORITIZATION_YOUNGEST,  // Youngest First
};

// Each of these corresponds to a string value in the gpgpsim.config file
// For example - to specify the LRR scheudler the config must contain lrr
enum concrete_scheduler {
  CONCRETE_SCHEDULER_LRR = 0,
  CONCRETE_SCHEDULER_GTO,
  CONCRETE_SCHEDULER_TWO_LEVEL_ACTIVE,
  CONCRETE_SCHEDULER_RRR,
  CONCRETE_SCHEDULER_WARP_LIMITING,
  CONCRETE_SCHEDULER_OLDEST_FIRST,
  NUM_CONCRETE_SCHEDULERS
};







// Static Warp Limiting Scheduler


class barrier_set_t {
 public:
  barrier_set_t(shader_core_ctx_wrapper *shader, unsigned max_warps_per_core,
                unsigned max_cta_per_core, unsigned max_barriers_per_cta,
                unsigned warp_size);

  // during cta allocation
  void allocate_barrier(unsigned cta_id, warp_set_t warps);

  // during cta deallocation
  void deallocate_barrier(unsigned cta_id);

  typedef std::map<unsigned, warp_set_t> cta_to_warp_t;
  typedef std::map<unsigned, warp_set_t>
      bar_id_to_warp_t; /*set of warps reached a specific barrier id*/

  // individual warp hits barrier
  void warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                            warp_inst_t *inst);

  // warp reaches exit
  void warp_exit(unsigned warp_id);

  // assertions
  bool warp_waiting_at_barrier(unsigned warp_id) const;

  // debug
  void dump();

 private:
  unsigned m_max_cta_per_core;
  unsigned m_max_warps_per_core;
  unsigned m_max_barriers_per_cta;
  unsigned m_warp_size;
  cta_to_warp_t m_cta_to_warps;
  bar_id_to_warp_t m_bar_id_to_warps;
  warp_set_t m_warp_active;
  warp_set_t m_warp_at_barrier;
  shader_core_ctx_wrapper *m_shader;
};

struct insn_latency_info {
  unsigned pc;
  unsigned long latency;
};

struct ifetch_buffer_t {
  ifetch_buffer_t() { m_valid = false; }

  ifetch_buffer_t(address_type pc, unsigned nbytes, unsigned warp_id) {
    m_valid = true;
    m_pc = pc;
    m_nbytes = nbytes;
    m_warp_id = warp_id;
    only_read_I2 = false; // MOD. VPREG
  }

  bool m_valid;
  address_type m_pc;
  unsigned m_nbytes;
  unsigned m_warp_id;
  bool only_read_I2; // MOD. VPREG
};

class shader_core_config;









class simt_core_cluster;
class shader_memory_interface;
class shader_core_mem_fetch_allocator;
class cache_t;


enum pipeline_stage_name_t {
  ID_OC_SP = 0,
  ID_OC_DP,
  ID_OC_INT,
  ID_OC_SFU,
  ID_OC_MEM,
  OC_EX_SP,
  OC_EX_DP,
  OC_EX_INT,
  OC_EX_SFU,
  OC_EX_MEM,
  EX_WB,
  ID_OC_TENSOR_CORE,
  OC_EX_TENSOR_CORE,
  N_PIPELINE_STAGES
};

const char *const pipeline_stage_name_decode[] = {
    "ID_OC_SP",          "ID_OC_DP",         "ID_OC_INT", "ID_OC_SFU",
    "ID_OC_MEM",         "OC_EX_SP",         "OC_EX_DP",  "OC_EX_INT",
    "OC_EX_SFU",         "OC_EX_MEM",        "EX_WB",     "ID_OC_TENSOR_CORE",
    "OC_EX_TENSOR_CORE", "N_PIPELINE_STAGES"};

struct specialized_unit_params {
  unsigned latency;
  unsigned num_units;
  unsigned id_oc_spec_reg_width;
  unsigned oc_ex_spec_reg_width;
  char name[20];
  unsigned ID_OC_SPEC_ID;
  unsigned OC_EX_SPEC_ID;
};

class shader_core_config : public core_config {
 public:
  shader_core_config(gpgpu_context *ctx) : core_config(ctx) {
    pipeline_widths_string = NULL;
    gpgpu_ctx = ctx;
  }

  void init() {
    int ntok = sscanf(gpgpu_shader_core_pipeline_opt, "%d:%d",
                      &n_thread_per_shader, &warp_size);
    if (ntok != 2) {
      printf(
          "GPGPU-Sim uArch: error while parsing configuration string "
          "gpgpu_shader_core_pipeline_opt\n");
      abort();
    }

    char *toks = new char[100];
    char *tokd = toks;
    strcpy(toks, pipeline_widths_string);

    toks = strtok(toks, ",");

    /*	Removing the tensorcore pipeline while reading the config files if the
       tensor core is not available. If we won't remove it, old regression will
       be broken. So to support the legacy config files it's best to handle in
       this way.
     */
    int num_config_to_read = N_PIPELINE_STAGES - 2 * (!gpgpu_tensor_core_avail);

    for (int i = 0; i < num_config_to_read; i++) {
      assert(toks);
      ntok = sscanf(toks, "%d", &pipe_widths[i]);
      assert(ntok == 1);
      toks = strtok(NULL, ",");
    }

    delete[] tokd;

    if (n_thread_per_shader > MAX_THREAD_PER_SM) {
      printf(
          "GPGPU-Sim uArch: Error ** increase MAX_THREAD_PER_SM in "
          "abstract_hardware_model.h from %u to %u\n",
          MAX_THREAD_PER_SM, n_thread_per_shader);
      abort();
    }
    max_warps_per_shader = n_thread_per_shader / warp_size;
    assert(!(n_thread_per_shader % warp_size));

    set_pipeline_latency();

    m_L0I_config.init(m_L0I_config.m_config_string, FuncCachePreferNone); // MOD. Added L0I
    m_L1I_L1_half_C_cache_config.init(m_L1I_L1_half_C_cache_config.m_config_string, FuncCachePreferNone);
    m_L1T_config.init(m_L1T_config.m_config_string, FuncCachePreferNone);
    m_L1C_config.init(m_L1C_config.m_config_string, FuncCachePreferNone);
    m_L0C_config.init(m_L0C_config.m_config_string, FuncCachePreferNone);
    m_L1D_config.init(m_L1D_config.m_config_string, FuncCachePreferNone);
    gpgpu_cache_texl1_linesize = m_L1T_config.get_line_sz();
    gpgpu_cache_constl1_linesize = m_L1C_config.get_line_sz();
    m_valid = true;

    m_specialized_unit_num = 0;
    // parse the specialized units
    for (unsigned i = 0; i < SPECIALIZED_UNIT_NUM; ++i) {
      unsigned enabled;
      specialized_unit_params sparam;
      sscanf(specialized_unit_string[i], "%u,%u,%u,%u,%u,%s", &enabled,
             &sparam.num_units, &sparam.latency, &sparam.id_oc_spec_reg_width,
             &sparam.oc_ex_spec_reg_width, sparam.name);

      if (enabled) {
        m_specialized_unit.push_back(sparam);
        strncpy(m_specialized_unit.back().name, sparam.name,
                sizeof(m_specialized_unit.back().name));
        m_specialized_unit_num += sparam.num_units;
      } else
        break;  // we only accept continuous specialized_units, i.e., 1,2,3,4
    }

    // parse gpgpu_shmem_option for adpative cache config
    if (adaptive_cache_config) {
      std::stringstream ss(gpgpu_shmem_option);
      while (ss.good()) {
        std::string option;
        std::getline(ss, option, ',');
        shmem_opt_list.push_back((unsigned)std::stoi(option) * 1024);
      }
      std::sort(shmem_opt_list.begin(), shmem_opt_list.end());
    }
  }
  void reg_options(class OptionParser *opp);
  unsigned int max_cta(const kernel_info_t &k) const;
  unsigned int num_shader() const {
    return n_simt_clusters * n_simt_cores_per_cluster;
  }
  unsigned sid_to_cluster(unsigned sid) const {
    return sid / n_simt_cores_per_cluster;
  }
  unsigned sid_to_cid(unsigned sid) const {
    return sid % n_simt_cores_per_cluster;
  }
  unsigned cid_to_sid(unsigned cid, unsigned cluster_id) const {
    return cluster_id * n_simt_cores_per_cluster + cid;
  }
  void set_pipeline_latency();

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  // data
  char *gpgpu_shader_core_pipeline_opt;
  bool gpgpu_perfect_mem;
  bool gpgpu_clock_gated_reg_file;
  bool gpgpu_clock_gated_lanes;
  enum divergence_support_t model;
  unsigned int n_thread_per_shader;
  unsigned int n_regfile_gating_group;
  unsigned int max_warps_per_shader;
  unsigned
      max_cta_per_core;  // Limit on number of concurrent CTAs in shader core
  unsigned max_barriers_per_cta;
  char *gpgpu_scheduler_string;
  unsigned gpgpu_shmem_per_block;
  unsigned gpgpu_registers_per_block;
  char *pipeline_widths_string;
  int pipe_widths[N_PIPELINE_STAGES];

  mutable cache_config m_L0I_config; // MOD. Added L0I
  mutable cache_config m_L1I_L1_half_C_cache_config;
  mutable cache_config m_L1T_config;
  mutable cache_config m_L1C_config;
  mutable cache_config m_L0C_config;
  mutable l1d_cache_config m_L1D_config;

  bool gpgpu_dwf_reg_bankconflict;

  unsigned gpgpu_num_sched_per_core;
  int gpgpu_max_insn_issue_per_warp;
  bool gpgpu_dual_issue_diff_exec_units;

  // op collector
  bool enable_specialized_operand_collector;
  int gpgpu_operand_collector_num_units_sp;
  int gpgpu_operand_collector_num_units_dp;
  int gpgpu_operand_collector_num_units_sfu;
  int gpgpu_operand_collector_num_units_tensor_core;
  int gpgpu_operand_collector_num_units_mem;
  unsigned int gpgpu_operand_collector_num_units_gen;
  int gpgpu_operand_collector_num_units_int;

  unsigned int gpgpu_operand_collector_num_in_ports_sp;
  unsigned int gpgpu_operand_collector_num_in_ports_dp;
  unsigned int gpgpu_operand_collector_num_in_ports_sfu;
  unsigned int gpgpu_operand_collector_num_in_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_in_ports_mem;
  unsigned int gpgpu_operand_collector_num_in_ports_gen;
  unsigned int gpgpu_operand_collector_num_in_ports_int;

  unsigned int gpgpu_operand_collector_num_out_ports_sp;
  unsigned int gpgpu_operand_collector_num_out_ports_dp;
  unsigned int gpgpu_operand_collector_num_out_ports_sfu;
  unsigned int gpgpu_operand_collector_num_out_ports_tensor_core;
  unsigned int gpgpu_operand_collector_num_out_ports_mem;
  unsigned int gpgpu_operand_collector_num_out_ports_gen;
  unsigned int gpgpu_operand_collector_num_out_ports_int;

  unsigned int gpgpu_num_sp_units;
  unsigned int gpgpu_tensor_core_avail;
  unsigned int gpgpu_num_dp_units;
  unsigned int gpgpu_num_sfu_units;
  unsigned int gpgpu_num_tensor_core_units;
  unsigned int gpgpu_num_mem_units;
  unsigned int gpgpu_num_int_units;

  // Shader core resources
  unsigned gpgpu_shader_registers;
  int gpgpu_warpdistro_shader;
  int gpgpu_warp_issue_shader;
  unsigned gpgpu_num_reg_banks;
  bool gpgpu_reg_bank_use_warp_id;
  bool gpgpu_local_mem_map;
  bool gpgpu_ignore_resources_limitation;
  bool sub_core_model;

  unsigned max_sp_latency;
  unsigned max_int_latency;
  unsigned max_sfu_latency;
  unsigned max_dp_latency;
  unsigned max_tensor_core_latency;

  unsigned n_simt_cores_per_cluster;
  unsigned n_simt_clusters;
  unsigned n_simt_ejection_buffer_size;
  unsigned ldst_unit_response_queue_size;

  int simt_core_sim_order;

  unsigned smem_latency;

  unsigned mem2device(unsigned memid) const { return memid + n_simt_clusters; }

  // Jin: concurrent kernel on sm
  bool gpgpu_concurrent_kernel_sm;

  bool perfect_inst_const_cache;
  unsigned inst_fetch_throughput;
  unsigned reg_file_port_throughput;

  // specialized unit config strings
  char *specialized_unit_string[SPECIALIZED_UNIT_NUM];
  mutable std::vector<specialized_unit_params> m_specialized_unit;
  unsigned m_specialized_unit_num;

  bool is_trace_predication_enabled; // MOD. Predication
  // MOD. Begin. Fix WAR at baseline.
  char *scoreboard_war_mode; // Indicates the mode of use of the scoreboard_reads in order to fix the war hazards at the baseline with a string
  scoreboard_reads_mode scoreboard_war_reads_mode; // Indicates the mode of use of the scoreboard_reads in order to fix the war hazards at the baseline with an enum
  unsigned int scoreboard_war_max_uses_per_reg; // Maximum of concurrent uses per register in the scoreboard_reads
  double scoreboard_war_static_power;
  double scoreboard_war_dynamic_power;
  // MOD. End

  bool is_fix_memory_reordering_enabled_baseline;  // MOD. Fix loads after stores in the baseline.

  bool is_L0I_enabled; // MOD. Added L0I
  bool is_fix_instruction_fetch_misalignment; // MOD. Fix misaligned fetched instructions
  bool is_fix_different_kernels_pc_addresses; // MOD. Fix instruction addresses of different kernels to have a different address request in memory
  bool is_fix_not_decoding_not_contiguos_instructions; // MOD. Not decoding instructions that have separated PC.
  bool is_improved_ldst_unit_enabled; // MOD. Fixed LDST_Unit model.
  bool is_improved_result_bus; // MOD. Improved Result bus to take into account conflicts with RF banks.
  int max_request_allowed_to_L1I; // MOD. Added L0I
  int max_reply_allowed_from_L1I; // MOD. Added L0I
  int latency_L0_to_L1; // MOD. Added L0I
  int latency_L1_to_L0; // MOD. Added L0I
  bool is_fetch_and_decode_improved; // MOD. Improving fetch and decode
  bool is_opc_improved; // MOD. Improving OPC
  int cu_num_ports; // MOD. Improving OPC
  bool is_skip_rf_limit_enabled; // MOD. Skip RF limitation.
  bool is_relax_barriers_baseline; // MOD. Relax barriers in baseline

  concrete_scheduler warp_scheduling_mode;

  bool is_trace_mode; // MOD. General Config Helper
  unsigned int filter_first_kernel_id; // If it has a value of 1 or 0 it is disabled
  unsigned int filter_last_kernel_id; // If it has a value of 1 or 0 it is disabled


  // MOD. Begin. Extended IBuffer
  bool is_extended_ibuffer_enabled;
  int extended_ibuffer_size;
  int fetch_decode_width;
  double extended_ibuffer_static_power;
  double extended_ibuffer_dynamic_power;
  // MOD. End. Extended IBuffer

  // MOD. Begin. LOOG
  bool is_loog_enabled;
  int loog_frontend_size;
  int loog_rrs_size;
  int loog_memory_queues_size;
  // MOD. End. LOOG

  // MOD. Begin VPREG
  char *vpreg_mode_string;
  bool is_vpreg_enabled;
  bool is_vpreg_predicated_war_waw_dependencies_ignored;
  bool is_vpreg_predicated_dest_reg_dependencies_ignored;
  bool is_vpreg_balanced_banks_mode_enabled;
  int vpreg_num_virtual_regs_per_sm;
  int vpreg_num_physical_regs_per_sm;
  int vpreg_reissue_informed_socgpu_threshold;
  int vpreg_max_rollback_entries_done_in_a_cycle;

  double vpreg_merge_module_static_power;
  double vpreg_merge_module_dynamic_power;
  double vpreg_collector_unit_extra_static_power;
  double vpreg_collector_unit_extra_dynamic_power;
  // MOD. Begin VPREG
  // MOD. Begin. Remodeling
  bool is_SM_remodeling_enabled; 
  bool is_remodeling_scoreboarding_enabled; 
  int num_subcores_in_SM;
  bool is_ibuffer_remodeled_enabled;
  int ibuffer_remodeled_size;
  unsigned int num_wait_barriers_per_warp;
  int sfu_latency;
  int tensor_latency;
  int tensor_extra_latency_16816_fp32_1688_fp32;
  int tensor_rate_per_cycle;
  int branch_latency;
  int half_latency;
  int uniform_latency;
  unsigned int predicate_latency;
  int miscellaneous_queue_latency;
  int miscellaneous_no_queue_latency;
  int sfu_initiation;
  int tensor_initiation;
  int branch_initiation;
  int half_initiation;
  int uniform_initiation;
  int predicate_initiation;
  int miscellaneous_queue_initiation;
  int miscellaneous_no_queue_initiation;
  unsigned int memory_intermidiate_stages_subcore_unit;
  unsigned int dp_shared_intermidiate_stages;
  unsigned int miscellaneous_queue_size;
  unsigned int memory_subcore_queue_size;
  unsigned int memory_sm_prt_size;
  unsigned int num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_mem_inst;
  unsigned int num_cycles_to_wait_to_dispatch_another_inst_from_subcore_to_sm_shared_pipeline_when_is_dp_inst;
  unsigned int memory_shared_memory_minimum_latency;
  unsigned int memory_shared_memory_extra_latency_ldsm_multiple_matrix;
  unsigned int memmory_max_concurrent_requests_shmem_per_sm;
  unsigned int memmory_max_concurrent_requests_standard_per_sm;
  unsigned int sm_memory_unit_l1c_access_queue_size;
  unsigned int sm_memory_unit_l1t_access_queue_size;
  unsigned int sm_memory_unit_l1d_access_queue_size;
  unsigned int sm_memory_unit_shmem_access_queue_size;
  unsigned int sm_memory_unit_bypass_l1d_directly_go_to_l2_access_queue_size;
  unsigned int sm_memory_unit_miscellaneous_access_queue_size;
  unsigned int constant_cache_latency_at_sm_structure;
  unsigned int constant_cache_miss_latency_at_subcore_to_access_upper_level;
  unsigned int memory_l1d_minimum_latency;
  unsigned int memory_global_shared_latency_for_ldgsts;
  unsigned int memory_l1d_max_lookups_per_cycle_per_bank;
  unsigned int memory_maximum_coalescing_cycles;
  unsigned int memory_subcore_extra_latency_load_shared_mem;
  unsigned int memory_num_scalar_units_per_subcore;
  unsigned int cycles_needed_for_address_calculation;
  unsigned int memory_subcore_link_to_sm_byte_size;
  unsigned int maximum_l1d_latency_at_sm_structure;
  unsigned int maximum_shared_memory_latency_at_sm_structure;
  unsigned int dp_subcore_queue_size;
  unsigned int dp_subcore_max_latency;
  unsigned int dp_sm_shared_queue_size;
  bool is_dp_pipeline_shared_for_subcores;
  bool is_load_half_bandwidth_in_the_subcore_link_to_sm_enabled;
  bool is_store_half_bandwidth_in_the_subcore_link_to_sm_enabled;
  bool is_fp32ops_allowed_in_int_pipeline;
  bool is_fp32_and_int_unified_pipeline;
  bool is_const_cache_accessed_blocks_tracking_enabled;
  bool is_global_memory_accesses_blocks_tracking_enabled;
  bool is_num_virtual_pages_tracking_enabled;
  unsigned int virtual_page_size_in_bytes;
  unsigned int num_const_cache_cycle_misses_before_switch_to_other_warp;
  unsigned int num_cycles_issue_port_busy_after_imadwide;
  unsigned int num_stall_cycles_wait_after_bits_stall_0_and_yield;
  unsigned int num_cycles_to_stall_SM_at_gpu_memory_barrier;
  unsigned int num_cycles_to_stall_SM_at_system_memory_barrier;
  unsigned int num_cycles_to_stall_SM_at_cta_memory_barrier;
  
  int offset_latency_firts_stage_memory_subcore;

  bool invalidate_instruction_caches_at_kernel_end;
  bool ibuffer_coalescing;
  bool perfect_instruction_cache;
  bool perfect_constant_cache;
  bool is_instruction_prefetching_enabled;
  unsigned int prefetch_per_stream_buffer_size;
  unsigned int prefetch_num_stream_buffers;
  unsigned int num_instruction_prefetches_per_cycle;

  bool is_rf_cache_enabled;
  int max_operands_regular_register_file; 
  int max_latency_regular_register_file_latency; 
  int num_regular_register_file_read_ports_per_bank;
  int num_regular_register_file_write_ports_per_bank;
  int max_size_register_file_write_queue_for_fixed_latency_instructions;
  int max_pops_per_cycle_register_file_write_queue_for_fixed_latency_instructions;
  int num_threads_granularity_read_regular_register_file_dp_inst;
  int num_threads_granularity_read_regular_register_file_mem_inst;
  int num_threads_granularity_read_regular_register_file_sfu_inst;
  int num_threads_granularity_read_regular_register_file_other_inst;
  int num_cycles_needed_to_write_a_reg_from_sm_struct_to_subcore;
  // MOD. End. Remodeling

  // MOD. Begin. Parallelism
  bool is_custom_omp_scheduler_enabled;
  float custom_omp_scheduler_ratio_to_dynamic;
  // MOD. End. Parallelism

  // MOD. Begin. InterWarp coalescing
  bool measure_coalescing_potential_stats;
  bool is_interwarp_coalescing_enabled;
  unsigned int num_interwarp_coalescing_tables;
  unsigned int max_size_interwarp_coalescing_per_table;
  unsigned int interwarp_coalescing_quanta;
  double interwarp_coalescing_quanta_warppool_policy_miss_ratio_threshold;
  unsigned int number_of_coalescers;
  unsigned int number_of_clusters_for_prt_selection;
  char* interwarp_coalescing_selection_policy_string;
  char* prt_selection_policy_string;
  InterWarpCoalescingSelectionPolicies interwarp_coalescing_selection_policy;
  PRTSelectionPolicies prt_selection_policy;
  // MOD. End. InterWarp coalescing
};

struct shader_core_stats_pod {
  void *
      shader_core_stats_pod_start[0];  // DO NOT MOVE FROM THE TOP - spaceless
                                       // pointer to the start of this structure
  unsigned long long *shader_cycles;

  // MOD. Begin. Custom Stats
  //First dimension is the number of kernel. Second dimension is the number of SM of the GPU
  std::vector<std::vector<unsigned long long>> m_num_sim_winsn_per_shader_per_kernel;
  std::vector<std::vector<unsigned long long>> shader_active_warps_per_kernel;
  std::vector<std::vector<unsigned long long>> shader_maximum_theoretical_warps_per_kernel;
  std::vector<std::vector<unsigned long long>> shader_cycles_per_kernel;
  std::vector<std::vector<double>> shader_warp_ipc_per_kernel;
  std::vector<std::vector<double>> shader_occupancy_per_kernel;
  std::vector<unsigned long long> gpu_cycles_per_kernel;
  std::vector<double> weighted_average_shader_warp_ipc_per_kernel;
  std::vector<double> weighted_average_shader_occupancy_per_kernel;
  std::vector<double> average_num_shader_active_per_kernel;
  std::vector<unsigned long long> number_of_warps_per_kernel;
  std::vector<unsigned long long> m_num_sim_winsn_per_shader;
  std::vector<double> shader_warp_ipc_per_shader;

  double total_weighted_average_warp_ipc_between_shaders;
  double total_weighted_average_shader_warp_ipc_with_kernels;
  double total_weighted_average_shader_occupancy;
  double total_weighted_average_num_shader_active;

  long double total_weighted_average_warps_per_kernel;

  unsigned long long number_of_total_warps;

  unsigned m_last_kernel_id;
  unsigned m_current_kernel_pos;
  unsigned numEffectiveIncompleteWarps;
  unsigned numberOfTotalWarps;

  unsigned long long tot_scheduler_cycles;
  unsigned long long tot_scheduler_issues;

  unsigned long long tot_num_expected_wb; // MOD. Custom stats
  unsigned long long tot_num_allocated_wb; // MOD. Custom stats

  unsigned long long tot_fetch_instruction_misalignments; // MOD. Fix misaligned fetched instructions
  unsigned long long tot_fetch_requests; // MOD. Fix misaligned fetched instructions

  unsigned num_scoreboard_reads_check_collision;// MOD. Scoreboard_reads
  unsigned num_scoreboard_reads_collision_due_to_max_uses_per_reg;// MOD. Scoreboard_reads
  unsigned num_scheduler_stall_cycle_due_to_war_scoreboard; // MOD. Scoreboard_reads
  unsigned num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard; // MOD. Scoreboard_reads


  // MOD. Begin. IBuffer_ooo stats
  // First dimension is kernel, second is shader id, third dimension is warp
   
  std::vector<std::vector<std::vector<unsigned long long>>> ins_issued_per_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> ins_released_wb_per_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> ins_released_opc_per_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> num_flushes_kernel_per_sid_per_warp;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_ibooo_empty;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_ibooo_empty_evaluated;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_ibooo_full;
  std::vector<std::vector<std::vector<unsigned long long>>> num_times_fetch_ibooo_tried;

  unsigned long long total_ins_issued_per_kernel_per_sid_per_warp;
  unsigned long long total_ins_released_wb_per_kernel_per_sid_per_warp;
  unsigned long long total_ins_released_opc_per_kernel_per_sid_per_warp;
  unsigned long long total_num_flushes_kernel_per_sid_per_warp;
  unsigned long long total_num_barriers;
  unsigned long long total_num_returns;
  unsigned long long total_num_branches;
  unsigned long long total_num_jumps;
  unsigned long long total_num_warpsyncs;
  unsigned long long total_num_bsyncs;
  unsigned long long total_num_rpcmovs;
  unsigned long long total_num_yields;
  unsigned long long total_num_barriers_and_controlflows;
  unsigned long long total_num_times_ibooo_empty;
  unsigned long long total_num_times_ibooo_empty_evaluated;
  unsigned long long total_num_times_ibooo_full;
  unsigned long long total_num_times_fetch_ibooo_tried;
  unsigned long long total_ibooo_num_entries_valid_and_not_issued;
  unsigned long long total_ibooo_num_entries_valid_not_issued_and_ready;
  unsigned long long total_ibooo_num_entries;
  unsigned long long total_ibooo_num_times_without_any_candidate;
  unsigned long long total_ibooo_num_times_without_any_ready_candidate;
  unsigned long long total_ibooo_evaluations_compute_selection_stats;
  double total_percentage_ibooo_full;
  double total_percentage_ibooo_empty;
  unsigned long long total_instructions_inserted_in_ibooo;
  unsigned long long total_war_waw_dependencies;
  unsigned long long total_raw_dependencies;
  unsigned long long total_stop_point_dependencies;
  unsigned long long total_memory_reordering_dependencies;

  unsigned long long last_ins_issued_per_kernel_per_sid_per_warp;
  unsigned long long last_ins_released_wb_per_kernel_per_sid_per_warp;
  unsigned long long last_ins_released_opc_per_kernel_per_sid_per_warp;
  unsigned long long last_num_flushes_kernel_per_sid_per_warp;
  unsigned long long last_num_times_ibooo_empty;
  unsigned long long last_num_times_ibooo_empty_evaluated;
  unsigned long long last_num_times_ibooo_full;
  unsigned long long last_num_times_fetch_ibooo_tried;
  double last_percentage_ibooo_full;
  double last_percentage_ibooo_empty;
  // MOD. End. IBuffer_ooo stats

  // MOD. Begin. VPREG
  unsigned long long total_vpreg_predication_dependencies;
  unsigned long long total_vpreg_merges;
  unsigned long long total_vpreg_extra_rf_reads;
  unsigned long long total_rf_reads;
  unsigned int total_number_of_kernels_limited_by_regs;
  unsigned int total_number_of_kernels_limited_by_ctas;
  unsigned int total_number_of_kernels_limited_by_threads;
  unsigned int total_number_of_kernels_limited_by_shared_memory;
  unsigned int total_number_of_vpreg_decode_rollbacks;
  unsigned int total_number_of_vpreg_reissues;
  unsigned int total_number_of_vpreg_not_enough_virtual_at_decode;
  int max_vpreg_virtual_regs_used_in_subcore;
  int max_vpreg_physical_regs_used_in_subcore;
  int max_vpreg_physical_freepool_usage_in_bank;
  int max_vpreg_number_of_consumers;
  // MOD. End. VPREG

  // MOD. Begin. OPC custom stats
  unsigned long long total_number_of_opc_conflicts;
  unsigned long long total_number_of_opc_requests;
  
  unsigned long long num_times_cu_subcore_custom_stats_evaluated;
  unsigned long long num_times_no_cu_dispatched;
  unsigned long long num_times_no_cu_allocated_and_nothing_to_allocate;
  unsigned long long num_times_no_cu_allocated;
  unsigned long long num_times_no_cu_allocated_due_to_cus_are_full;
  unsigned long long num_times_no_cu_dispatched_due_to_dispatch_reg_full;
  unsigned long long num_times_no_cu_dispatched_due_to_no_ready_operands;
  unsigned long long num_times_no_cu_dispatched_due_to_all_cus_empty;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready;
  unsigned long long num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled;
  unsigned long long num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled;
  unsigned long long total_num_try_ldst_unit_dispatches;
  unsigned long long total_num_ldst_unit_dispatches_failed_due_to_not_empty_dispatch_reg;
  // MOD. End. OPC custom stats

  // MOD. Begin. Memory stats
  std::vector<std::vector<unsigned long long>> l1d_accesses_per_sid_per_bank;
  std::vector<std::vector<unsigned long long>> l1d_evals_per_sid_per_bank;
  long double total_avg_usage_l1d_bank;
  double max_avg_usage_l1d_bank;
  unsigned long long total_shared_mem_evals;
  unsigned long long total_shared_mem_accesses;
  unsigned long long total_num_dp_instructions;
  unsigned long long total_num_ldst_unit_instructions;
  unsigned long long total_num_warp_instructions;
  unsigned long long total_l1d_instructions;
  unsigned long long total_accesses_l1d_instructions;
  unsigned long long total_avg_cycles_to_schedule_accesses;
  unsigned long long total_shared_instructions;
  unsigned long long total_conflicts_shared_instructions;
  unsigned long long total_cycles_instructions_in_ldst_unit_dispatch_reg;
  unsigned long long total_cycles_instructions_in_ldst_unit_arbiter_latch; // MOD. Fixed LDST_Unit model.
  // MOD. End. Memory stats

  unsigned long long total_cycles_instructions_in_cu; // MOD. CU stats

  // First dimension SM, second warp ID
  std::vector<std::vector<unsigned long long>> warp_issues_from_last_power_sample; // MOD. Custom powermodel stats
  std::vector<std::vector<unsigned long long>> bank_wb_from_last_power_sample; // MOD. Custom powermodel stats 
  std::vector<std::vector<unsigned long long>> collector_unit_allocations_from_last_power_sample; // MOD. Custom powermodel stats 
  // MOD. End. custom Stats


  unsigned *m_num_sim_insn;   // number of scalar thread instructions committed
                              // by this shader core
  unsigned *m_num_sim_winsn;  // number of warp instructions committed by this
                              // shader core
  unsigned *m_last_num_sim_insn;
  unsigned *m_last_num_sim_winsn;
  unsigned *
      m_num_decoded_insn;  // number of instructions decoded by this shader core
  float *m_pipeline_duty_cycle;
  unsigned *m_num_FPdecoded_insn;
  unsigned *m_num_INTdecoded_insn;
  unsigned *m_num_storequeued_insn;
  unsigned *m_num_loadqueued_insn;
  unsigned *m_num_tex_inst;
  double *m_num_ialu_acesses;
  double *m_num_fp_acesses;
  double *m_num_imul_acesses;
  double *m_num_fpmul_acesses;
  double *m_num_idiv_acesses;
  double *m_num_fpdiv_acesses;
  double *m_num_sp_acesses;
  double *m_num_sfu_acesses;
  double *m_num_tensor_core_acesses;
  double *m_num_tex_acesses;
  double *m_num_const_acesses;
  double *m_num_dp_acesses;
  double *m_num_dpmul_acesses;
  double *m_num_dpdiv_acesses;
  double *m_num_sqrt_acesses;
  double *m_num_log_acesses;
  double *m_num_sin_acesses;
  double *m_num_exp_acesses;
  double *m_num_mem_acesses;
  unsigned *m_num_sp_committed;
  unsigned *m_num_tlb_hits;
  unsigned *m_num_tlb_accesses;
  unsigned *m_num_sfu_committed;
  unsigned *m_num_tensor_core_committed;
  unsigned *m_num_mem_committed;
  unsigned *m_read_regfile_acesses;
  unsigned *m_write_regfile_acesses;
  unsigned *m_non_rf_operands;
  double *m_num_imul24_acesses;
  double *m_num_imul32_acesses;
  unsigned *m_active_sp_lanes;
  unsigned *m_active_sfu_lanes;
  unsigned *m_active_tensor_core_lanes;
  unsigned *m_active_fu_lanes;
  unsigned *m_active_fu_mem_lanes;
  double *m_active_exu_threads; //For power model
  double *m_active_exu_warps; //For power model
  unsigned *m_n_diverge;  // number of divergence occurring in this shader
  unsigned long long gpgpu_n_load_insn;
  unsigned long long gpgpu_n_store_insn;
  unsigned long long gpgpu_n_shmem_insn;
  unsigned long long gpgpu_n_sstarr_insn;
  unsigned long long gpgpu_n_tex_insn;
  unsigned long long gpgpu_n_const_insn;
  unsigned long long gpgpu_n_param_insn;
  unsigned long long gpgpu_n_shmem_bkconflict;
  unsigned long long gpgpu_n_l1cache_bkconflict;
  unsigned long long gpgpu_n_l1cache_coalescing_conflicts;
  int gpgpu_n_intrawarp_mshr_merge;
  unsigned gpgpu_n_cmem_portconflict;
  unsigned gpgpu_n_cmem_coalescing_conflicts;
  unsigned gpu_stall_shd_mem_breakdown[N_MEM_STAGE_ACCESS_TYPE]
                                      [N_MEM_STAGE_STALL_TYPE];
  unsigned gpu_reg_bank_conflict_stalls;
  unsigned *shader_cycle_distro;
  unsigned *last_shader_cycle_distro;
  unsigned *num_warps_issuable;
  unsigned gpgpu_n_stall_dispatch_to_subpipeline_mem;
  unsigned *single_issue_nums;
  unsigned *dual_issue_nums;

  unsigned ctas_completed;
  // memory access classification
  int gpgpu_n_mem_read_local;
  int gpgpu_n_mem_write_local;
  int gpgpu_n_mem_texture;
  int gpgpu_n_mem_const;
  int gpgpu_n_mem_read_global;
  int gpgpu_n_mem_write_global;
  int gpgpu_n_mem_read_inst;

  int gpgpu_n_mem_l2_writeback;
  int gpgpu_n_mem_l1_write_allocate;
  int gpgpu_n_mem_l2_write_allocate;

  unsigned made_write_mfs;
  unsigned made_read_mfs;

  unsigned *gpgpu_n_shmem_bank_access;
  long *n_simt_to_mem;  // Interconnect power stats
  long *n_mem_to_simt;

  // MOD. Begin. Remodeling
  unsigned long long total_num_register_file_cache_hits;
  unsigned long long total_num_register_file_cache_allocations;
  unsigned long long total_num_regular_regfile_reads;
  unsigned long long total_num_regular_regfile_writes;
  unsigned long long total_num_uniform_regfile_reads;
  unsigned long long total_num_uniform_regfile_writes;
  unsigned long long total_num_predicate_regfile_reads;
  unsigned long long total_num_predicate_regfile_writes;
  unsigned long long total_num_uniform_predicate_regfile_reads;
  unsigned long long total_num_uniform_predicate_regfile_writes;
  unsigned long long total_num_constant_cache_reads;
  std::set<new_addr_type> all_const_cache_accessed_blocks;
  std::set<new_addr_type> all_global_memory_accessed_blocks;
  std::set<unsigned int> all_virtual_pages_accessed;

  unsigned long long total_num_times_wb_evaluated;
  unsigned long long total_num_times_wb_port_conflict;

  unsigned long long total_num_cycles_issue_stage_evaluated;
  unsigned long long total_num_cycles_issue_stage_issuing;
  unsigned long long total_num_cycles_issue_stage_stall_issue_port_busy;
  unsigned long long total_num_cycles_issue_stage_stall_no_valid_instruction;
  unsigned long long total_num_cycles_issue_stage_stall_no_warps_ready;

  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count;
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c;

  unsigned int num_kernel_not_in_binary;
  // MOD. End. Remodeling

};

class shader_core_stats : public shader_core_stats_pod {
 public:
  shader_core_stats(const shader_core_config *config, gpgpu_sim *gpu) {
    m_config = config;
    shader_core_stats_pod *pod = reinterpret_cast<shader_core_stats_pod *>(
        this->shader_core_stats_pod_start);
    memset(reinterpret_cast<void *>(pod), 0, sizeof(shader_core_stats_pod));
    shader_cycles = (unsigned long long *)calloc(config->num_shader(),
                                                 sizeof(unsigned long long));
    m_num_sim_insn = (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    // MOD.. Begin. Custom Stats
    m_last_kernel_id = 0;
    total_weighted_average_warp_ipc_between_shaders = 0;
    total_weighted_average_shader_occupancy = 0;
    total_weighted_average_shader_warp_ipc_with_kernels = 0;
    total_weighted_average_num_shader_active = 0;
    m_num_sim_winsn_per_shader.resize(m_config->num_shader());
    shader_warp_ipc_per_shader.resize(m_config->num_shader());

    tot_scheduler_cycles = 0;
    tot_scheduler_issues = 0;

    tot_num_expected_wb = 0; // MOD. Custom stats
    tot_num_allocated_wb = 0; // MOD. Custom stats

    tot_fetch_instruction_misalignments = 0; // MOD. Fix misaligned fetched instructions
    tot_fetch_requests = 0; // MOD. Fix misaligned fetched instructions

    numEffectiveIncompleteWarps = 0;

    num_scoreboard_reads_check_collision = 0;// MOD. Scoreboard_reads
    num_scoreboard_reads_collision_due_to_max_uses_per_reg = 0;// MOD. Scoreboard_reads
    num_scheduler_stall_cycle_due_to_war_scoreboard = 0;// MOD. Scoreboard_reads
    num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard = 0;// MOD. Scoreboard_reads

    // MOD. Begin. IBuffer_ooo stats
    total_ins_issued_per_kernel_per_sid_per_warp = 0;
    total_ins_released_wb_per_kernel_per_sid_per_warp = 0;
    total_ins_released_opc_per_kernel_per_sid_per_warp = 0;
    total_num_flushes_kernel_per_sid_per_warp = 0;
    total_num_barriers = 0;
    total_num_returns = 0;
    total_num_branches = 0;
    total_num_barriers_and_controlflows = 0;
    total_num_times_ibooo_empty = 0;
    total_num_times_ibooo_empty_evaluated = 0;
    total_num_times_ibooo_full = 0;
    total_num_times_fetch_ibooo_tried = 0;
    total_percentage_ibooo_full = 0;
    total_percentage_ibooo_empty = 0;
    total_instructions_inserted_in_ibooo = 0;
    total_war_waw_dependencies = 0;
    total_raw_dependencies = 0;
    total_stop_point_dependencies = 0;
    total_memory_reordering_dependencies = 0;
    // MOD. End. IBuffer_ooo stats

    // MOD. Begin. VPREG stats
    total_vpreg_merges = 0;
    total_vpreg_extra_rf_reads = 0;
    total_rf_reads = 0;
    total_number_of_kernels_limited_by_regs = 0;
    total_number_of_vpreg_decode_rollbacks = 0;
    total_number_of_vpreg_reissues = 0;
    total_number_of_vpreg_not_enough_virtual_at_decode = 0;
    max_vpreg_virtual_regs_used_in_subcore = 0;
    max_vpreg_physical_regs_used_in_subcore = 0;
    max_vpreg_physical_freepool_usage_in_bank = 0;
    max_vpreg_number_of_consumers = 0;
    total_vpreg_predication_dependencies = 0;
    // MOD. End. VPREG stats

    // MOD. Begin. OPC custom stats
    total_number_of_opc_conflicts = 0;
    total_number_of_opc_requests = 0;
    num_times_cu_subcore_custom_stats_evaluated = 0;
    num_times_no_cu_dispatched = 0;
    num_times_no_cu_allocated_and_nothing_to_allocate = 0;
    num_times_no_cu_allocated = 0;
    num_times_no_cu_allocated_due_to_cus_are_full = 0;
    num_times_no_cu_dispatched_due_to_dispatch_reg_full = 0;
    num_times_no_cu_dispatched_due_to_no_ready_operands = 0;
    num_times_no_cu_dispatched_due_to_all_cus_empty = 0;
    num_times_no_cu_dispatched_and_all_cus_full = 0;
    num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready = 0;
    num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready = 0;
    num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled = 0;
    num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled = 0;
    total_num_try_ldst_unit_dispatches = 0;
    total_num_ldst_unit_dispatches_failed_due_to_not_empty_dispatch_reg = 0;
    total_cycles_instructions_in_cu = 0; // MOD. CU stats
    // MOD. End. OPC custom stats

    // MOD. Begin. Memory stats
    l1d_accesses_per_sid_per_bank.resize(m_config->num_shader());
    l1d_evals_per_sid_per_bank.resize(m_config->num_shader());
    max_avg_usage_l1d_bank = 0;
    total_shared_mem_evals = 0;
    total_shared_mem_accesses = 0;
    total_num_ldst_unit_instructions = 0;
    total_num_dp_instructions = 0;
    total_num_warp_instructions = 0;
    total_l1d_instructions = 0;
    total_accesses_l1d_instructions = 0;
    total_avg_cycles_to_schedule_accesses = 0;
    total_cycles_instructions_in_ldst_unit_dispatch_reg = 0;
    total_cycles_instructions_in_ldst_unit_arbiter_latch = 0; // MOD. Fixed LDST_Unit model.
    total_shared_instructions = 0;
    total_conflicts_shared_instructions = 0;
    // MOD. End. Memory stats

    // MOD. Begin. Custom powermodel stats
    warp_issues_from_last_power_sample.resize(m_config->num_shader());
    bank_wb_from_last_power_sample.resize(m_config->num_shader());
    collector_unit_allocations_from_last_power_sample.resize(m_config->num_shader());

    unsigned int max_num_j_iters = std::max(std::max(m_config->max_warps_per_shader, m_config->gpgpu_num_reg_banks), (unsigned int)m_config->gpgpu_operand_collector_num_units_gen);
    for(unsigned int i = 0; i < m_config->num_shader(); i++) {
        warp_issues_from_last_power_sample[i].resize(m_config->max_warps_per_shader);
        bank_wb_from_last_power_sample[i].resize(m_config->gpgpu_num_reg_banks);
        collector_unit_allocations_from_last_power_sample[i].resize(m_config->gpgpu_operand_collector_num_units_gen);
        
        for(unsigned int j = 0; j < max_num_j_iters; j++) {
          if(j < m_config->max_warps_per_shader) {
            warp_issues_from_last_power_sample[i][j] = 0;
          }
          if(j < m_config->gpgpu_num_reg_banks) {
            bank_wb_from_last_power_sample[i][j] = 0;
          }
          if(j < m_config->gpgpu_operand_collector_num_units_gen) {
            collector_unit_allocations_from_last_power_sample[i][j] = 0;
          }
        }
        l1d_accesses_per_sid_per_bank[i].resize( m_config->m_L1D_config.l1_banks); // MOD. Memory stats
        l1d_evals_per_sid_per_bank[i].resize( m_config->m_L1D_config.l1_banks); // MOD. Memory stats
    }
    // MOD. End. Custom powermodel stats

    // MOD. Begin. Remodeling
    total_num_register_file_cache_hits = 0;
    total_num_register_file_cache_allocations = 0;
    total_num_regular_regfile_reads = 0;
    total_num_regular_regfile_writes = 0;
    total_num_uniform_regfile_reads = 0;
    total_num_uniform_regfile_writes = 0;
    total_num_predicate_regfile_reads = 0;
    total_num_predicate_regfile_writes = 0;
    total_num_uniform_predicate_regfile_reads = 0;
    total_num_uniform_predicate_regfile_writes = 0;
    total_num_constant_cache_reads = 0;
    all_const_cache_accessed_blocks.clear();
    all_global_memory_accessed_blocks.clear();
    all_virtual_pages_accessed.clear();

    total_num_times_wb_evaluated = 0;
    total_num_times_wb_port_conflict = 0;

    total_num_cycles_issue_stage_evaluated = 0;
    total_num_cycles_issue_stage_issuing = 0;
    total_num_cycles_issue_stage_stall_issue_port_busy = 0;
    total_num_cycles_issue_stage_stall_no_valid_instruction = 0;
    total_num_cycles_issue_stage_stall_no_warps_ready = 0;

    total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count = 0;
    total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c = 0;

    num_kernel_not_in_binary = 0;
    // MOD. End. Remodeling

    // Mod. End. Custom Stats

    m_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_winsn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_last_num_sim_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_pipeline_duty_cycle =
        (float *)calloc(config->num_shader(), sizeof(float));
    m_num_decoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_FPdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_storequeued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_loadqueued_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tex_inst = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_INTdecoded_insn =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_ialu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul24_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_imul32_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpmul_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_idiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_fpdiv_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_dp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpmul_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_dpdiv_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sp_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sfu_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tensor_core_acesses = 
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_const_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_tex_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sqrt_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_log_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_sin_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_exp_acesses = 
        (double*) calloc(config->num_shader(),sizeof(double));
    m_num_mem_acesses =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_num_sp_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_hits = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tlb_accesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sp_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_sfu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_tensor_core_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_fu_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_active_exu_threads =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_exu_warps =
        (double *)calloc(config->num_shader(), sizeof(double));
    m_active_fu_mem_lanes =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_sfu_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_tensor_core_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_num_mem_committed =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_read_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_write_regfile_acesses =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_non_rf_operands =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    m_n_diverge = 
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));
    shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    last_shader_cycle_distro =
        (unsigned *)calloc(config->warp_size + 3, sizeof(unsigned));
    single_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));
    dual_issue_nums =
        (unsigned *)calloc(config->gpgpu_num_sched_per_core, sizeof(unsigned));

    ctas_completed = 0;
    n_simt_to_mem = (long *)calloc(config->num_shader(), sizeof(long));
    n_mem_to_simt = (long *)calloc(config->num_shader(), sizeof(long));

    m_outgoing_traffic_stats = new traffic_breakdown("coretomem");
    m_incoming_traffic_stats = new traffic_breakdown("memtocore");

    gpgpu_n_shmem_bank_access =
        (unsigned *)calloc(config->num_shader(), sizeof(unsigned));

    m_shader_dynamic_warp_issue_distro.resize(config->num_shader());
    m_shader_warp_slot_issue_distro.resize(config->num_shader());
    m_gpu = gpu;
  }

  ~shader_core_stats() {
    delete m_outgoing_traffic_stats;
    delete m_incoming_traffic_stats;
    free(m_num_sim_insn);
    free(m_num_sim_winsn);
    free(m_pipeline_duty_cycle);
    free(m_num_decoded_insn);
    free(m_num_FPdecoded_insn);
    free(m_num_INTdecoded_insn);
    free(m_num_storequeued_insn);
    free(m_num_loadqueued_insn);
    free(m_num_ialu_acesses);
    free(m_num_fp_acesses);
    free(m_num_imul_acesses);
    free(m_num_tex_inst);
    free(m_num_fpmul_acesses);
    free(m_num_idiv_acesses);
    free(m_num_fpdiv_acesses);
    free(m_num_sp_acesses);
    free(m_num_sfu_acesses);
    free(m_num_tensor_core_acesses);
    free(m_num_tex_acesses);
    free(m_num_const_acesses);
    free(m_num_dp_acesses);
    free(m_num_dpmul_acesses);
    free(m_num_dpdiv_acesses);
    free(m_num_sqrt_acesses);
    free(m_num_log_acesses);
    free(m_num_sin_acesses);
    free(m_num_exp_acesses);
    free(m_num_mem_acesses);
    free(m_num_sp_committed);
    free(m_num_tlb_hits);
    free(m_num_tlb_accesses);
    free(m_num_sfu_committed);
    free(m_num_tensor_core_committed);
    free(m_num_mem_committed);
    free(m_read_regfile_acesses);
    free(m_write_regfile_acesses);
    free(m_non_rf_operands);
    free(m_num_imul24_acesses);
    free(m_num_imul32_acesses);
    free(m_active_sp_lanes);
    free(m_active_sfu_lanes);
    free(m_active_tensor_core_lanes);
    free(m_active_fu_lanes);
    free(m_active_exu_threads);
    free(m_active_exu_warps);
    free(m_active_fu_mem_lanes);
    free(m_n_diverge);
    free(shader_cycle_distro);
    free(last_shader_cycle_distro);
    free(n_simt_to_mem);
    free(n_mem_to_simt);
    free(shader_cycles);
    free(m_last_num_sim_insn);
    free(single_issue_nums);
    free(dual_issue_nums);

    free(gpgpu_n_shmem_bank_access);
    free(m_last_num_sim_winsn);
  }

  // MOD. Begin. Custom Stats
  void allocate_for_a_new_kernel() {
    m_last_kernel_id +=1;
    m_current_kernel_pos = m_last_kernel_id - 1;

    gpu_cycles_per_kernel.resize(m_last_kernel_id);
    weighted_average_shader_warp_ipc_per_kernel.resize(m_last_kernel_id);
    weighted_average_shader_occupancy_per_kernel.resize(m_last_kernel_id);
    average_num_shader_active_per_kernel.resize(m_last_kernel_id);
    number_of_warps_per_kernel.resize(m_last_kernel_id);

    m_num_sim_winsn_per_shader_per_kernel.resize(m_last_kernel_id);
    shader_active_warps_per_kernel.resize(m_last_kernel_id);
    shader_maximum_theoretical_warps_per_kernel.resize(m_last_kernel_id);
    shader_cycles_per_kernel.resize(m_last_kernel_id);
    shader_warp_ipc_per_kernel.resize(m_last_kernel_id);
    shader_occupancy_per_kernel.resize(m_last_kernel_id);

    m_num_sim_winsn_per_shader_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_active_warps_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_maximum_theoretical_warps_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_cycles_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_warp_ipc_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    shader_occupancy_per_kernel[m_current_kernel_pos].resize(m_config->num_shader());
    // MOD. IBuffer_ooo. Begin stats
    ins_issued_per_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    ins_released_opc_per_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    ins_released_wb_per_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    num_flushes_kernel_per_sid_per_warp.resize(m_last_kernel_id);
    num_times_ibooo_empty.resize(m_last_kernel_id);
    num_times_ibooo_empty_evaluated.resize(m_last_kernel_id);
    num_times_ibooo_full.resize(m_last_kernel_id);
    num_times_fetch_ibooo_tried.resize(m_last_kernel_id);
    ins_issued_per_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    ins_released_opc_per_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    ins_released_wb_per_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    num_flushes_kernel_per_sid_per_warp[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_ibooo_empty[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_ibooo_empty_evaluated[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_ibooo_full[m_current_kernel_pos].resize(m_config->num_shader());
    num_times_fetch_ibooo_tried[m_current_kernel_pos].resize(m_config->num_shader());
    for(unsigned int i = 0; i < m_config->num_shader(); i++)
    {
      ins_issued_per_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      ins_released_opc_per_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      ins_released_wb_per_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_flushes_kernel_per_sid_per_warp[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_ibooo_empty[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_ibooo_empty_evaluated[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_ibooo_full[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      num_times_fetch_ibooo_tried[m_current_kernel_pos][i].resize(m_config->max_warps_per_shader);
      for(unsigned int j = 0; j < m_config->max_warps_per_shader; j++)
      {
        ins_issued_per_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        ins_released_opc_per_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        ins_released_wb_per_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        num_flushes_kernel_per_sid_per_warp[m_current_kernel_pos][i][j] = 0;
        num_times_ibooo_empty[m_current_kernel_pos][i][j] = 0;
        num_times_ibooo_empty_evaluated[m_current_kernel_pos][i][j] = 0;
        num_times_ibooo_full[m_current_kernel_pos][i][j] = 0;
        num_times_fetch_ibooo_tried[m_current_kernel_pos][i][j] = 0;
      }
    }
    // MOD. IBuffer_ooo. Begin stats
  }

  void print_custom_shader_stats(FILE *fout) const; // MOD.
  void print_remodeling_stats(FILE *fout); // MOD. Remodeling
  void print_coalescing_stats(FILE *fout); 
  void compute_ibuffer_ooo_stats(); // MOD. IBuffer_ooo
  void print_ibuffer_ooo_stats(FILE *fout) const; // MOD. IBuffer_ooo
  void print_vpreg_stats(FILE *fout) const; // MOD. VPREG
  void print_single_custom_shader_stat_long(FILE *fout, std::string stat_name, std::vector<std::vector<unsigned long long>> vector_stat) const;
  void print_single_custom_shader_stat_double(FILE *fout, std::string stat_name, std::vector<std::vector<double>> vector_stat) const;
  void compute_derived_custom_stats();

  // MOD. End. Custom Stats

  void new_grid() {}

  void event_warp_issued(unsigned s_id, unsigned warp_id, unsigned num_issued,
                         unsigned dynamic_warp_id);

  void visualizer_print(gzFile visualizer_file);

  void print(FILE *fout);

  const std::vector<std::vector<unsigned>> &get_dynamic_warp_issue() const {
    return m_shader_dynamic_warp_issue_distro;
  }

  const std::vector<std::vector<unsigned>> &get_warp_slot_issue() const {
    return m_shader_warp_slot_issue_distro;
  }

  traffic_breakdown *m_outgoing_traffic_stats;  // core to memory partitions
  traffic_breakdown *m_incoming_traffic_stats;  // memory partition to core

 private:
  const shader_core_config *my_custom_config; // MOD. Declared attribute to prevent crashing due to segFault because of adding to many stats doesn't like it
  const shader_core_config *m_config;


  // Counts the instructions issued for each dynamic warp.
  std::vector<std::vector<unsigned>> m_shader_dynamic_warp_issue_distro;
  std::vector<unsigned> m_last_shader_dynamic_warp_issue_distro;
  std::vector<std::vector<unsigned>> m_shader_warp_slot_issue_distro;
  std::vector<unsigned> m_last_shader_warp_slot_issue_distro;

  gpgpu_sim *m_gpu;

  friend class power_stat_t;
  friend class shader_core_ctx;
  friend class ldst_unit;
  friend class simt_core_cluster;
  friend class scheduler_unit;
  friend class TwoLevelScheduler;
  friend class LooseRoundRobbinScheduler;
};

class memory_config;
class shader_core_mem_fetch_allocator : public mem_fetch_allocator {
 public:
  shader_core_mem_fetch_allocator(unsigned core_id, unsigned cluster_id,
                                  const memory_config *config) {
    m_core_id = core_id;
    m_cluster_id = cluster_id;
    m_memory_config = config;
  }
  mem_fetch *alloc(new_addr_type addr, mem_access_type type, unsigned size,
                   bool wr, unsigned long long cycle) const;
  mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                   const active_mask_t &active_mask,
                   const mem_access_byte_mask_t &byte_mask,
                   const mem_access_sector_mask_t &sector_mask, unsigned size,
                   bool wr, unsigned long long cycle, unsigned wid,
                   unsigned sid, unsigned tpc, mem_fetch *original_mf) const;
  mem_fetch *alloc(const warp_inst_t &inst, const mem_access_t &access,
                   unsigned long long cycle) const {
    warp_inst_t inst_copy = inst;
    mem_fetch *mf = new mem_fetch(
        access, &inst_copy,
        access.is_write() ? WRITE_PACKET_SIZE : READ_PACKET_SIZE,
        inst.warp_id(), m_core_id, m_cluster_id, m_memory_config, cycle);
    return mf;
  }

 private:
  unsigned m_core_id;
  unsigned m_cluster_id;
  const memory_config *m_memory_config;
};


class simt_core_cluster {
 public:
  simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                    const shader_core_config *config,
                    const memory_config *mem_config, shader_core_stats *stats,
                    memory_stats_t *mstats);
  virtual ~simt_core_cluster() {
    for(unsigned int i = 0; i < m_core.size(); i++) {
      delete m_core[i];
    }
  }

  void reset_cycless_access_history() {
    for(unsigned i = 0; i < m_core.size(); i++) {
      m_core[i]->reset_cycless_access_history();
    }
  }

  void gather_stats(Element_stats &all_stats, coalescingStatsAcrossSms& coal_stats_l1d, coalescingStatsAcrossSms& coal_stats_const, coalescingStatsAcrossSms& coal_stats_sharedmem) {
    for(unsigned i = 0; i < m_core.size(); i++) {
      m_core[i]->gather_gpu_per_sm_stats(all_stats, coal_stats_l1d, coal_stats_const, coal_stats_sharedmem);
    }
  }

  void gather_single_stat(Element_stats &all_stats, std::string stat_name) {
    for(unsigned i = 0; i < m_core.size(); i++) {
      m_core[i]->gather_gpu_per_sm_single_stat(all_stats, stat_name);
    }
  }

  traffic_breakdown& get_incomming_traffic_stats() { return m_incoming_traffic_stats; }
  traffic_breakdown& get_outgoing_traffic_stats() { return m_outgoing_traffic_stats; }

  void core_cycle();
  void icnt_cycle();

  void reinit();
  unsigned issue_block2core();
  void cache_flush();
  void cache_invalidate();
  bool icnt_injection_buffer_full(unsigned size, bool write);
  void icnt_inject_request_packet(class mem_fetch *mf);

  // for perfect memory interface
  bool response_queue_full() {
    return (m_response_fifo.size() >= m_config->n_simt_ejection_buffer_size);
  }
  void push_response_fifo(class mem_fetch *mf) {
    m_response_fifo.push_back(mf);
  }

  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc) const;
  unsigned max_cta(const kernel_info_t &kernel);
  unsigned get_not_completed() const;
  void print_not_completed(FILE *fp) const;
  unsigned get_n_active_cta() const;
  unsigned get_n_active_sms() const;
  gpgpu_sim *get_gpu() { return m_gpu; }

  void display_pipeline(unsigned sid, FILE *fout, int print_mem, int mask);
  void print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                         unsigned &dl1_misses) const;

  void get_cache_stats(cache_stats &cs) const;
  void get_L1I_sub_stats(struct cache_sub_stats &css) const;
  void get_L0I_sub_stats(struct cache_sub_stats &css) const; // MOD. L0I
  void get_L1D_sub_stats(struct cache_sub_stats &css) const;
  void get_L1C_sub_stats(struct cache_sub_stats &css) const;
  void get_L1T_sub_stats(struct cache_sub_stats &css) const;

  void get_icnt_stats(long &n_simt_to_mem, long &n_mem_to_simt) const;
  float get_current_occupancy(unsigned long long &active,
                              unsigned long long &total) const;
  virtual void create_shader_core_ctx() = 0;

  void create_gpu_per_cluster_stats(Element_stats &all_stats);

 protected:
  unsigned m_cluster_id;
  gpgpu_sim *m_gpu;
  const shader_core_config *m_config;
  shader_core_stats *m_stats;
  memory_stats_t *m_memory_stats;
  std::vector<shader_core_ctx_wrapper *> m_core;
  const memory_config *m_mem_config;

  unsigned m_cta_issue_next_core;
  std::list<unsigned> m_core_sim_order;
  std::list<mem_fetch *> m_response_fifo;

  Element_stats m_cluster_stats;
  traffic_breakdown m_outgoing_traffic_stats;//("coretomem");  // core to memory partitions
  traffic_breakdown m_incoming_traffic_stats;//("memtocore");  // memory partition to core
};

class shader_memory_interface : public mem_fetch_interface {
 public:
  shader_memory_interface(shader_core_ctx_wrapper *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  ~shader_memory_interface() override {}
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->icnt_injection_buffer_full(size, write);
  }
  virtual void push(mem_fetch *mf) {
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->icnt_inject_request_packet(mf);
  }

  virtual void flush() {}

 private:
  shader_core_ctx_wrapper *m_core;
  simt_core_cluster *m_cluster;
};

class perfect_memory_interface : public mem_fetch_interface {
 public:
  perfect_memory_interface(shader_core_ctx_wrapper *core, simt_core_cluster *cluster) {
    m_core = core;
    m_cluster = cluster;
  }
  ~perfect_memory_interface() override {}
  virtual bool full(unsigned size, bool write) const {
    return m_cluster->response_queue_full();
  }
  virtual void push(mem_fetch *mf) {
    if (mf && mf->isatomic())
      mf->do_atomic();  // execute atomic inside the "memory subsystem"
    m_core->inc_simt_to_mem(mf->get_num_flits(true));
    m_cluster->push_response_fifo(mf);
  }

  virtual void flush() {}

 private:
  shader_core_ctx_wrapper *m_core;
  simt_core_cluster *m_cluster;
};


#endif /* SHADER_H */
