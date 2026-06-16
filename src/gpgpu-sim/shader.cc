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

// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Ali Bakhoda,
// George L. Yuan, Andrew Turner, Inderpreet Singh, Vijay Kandiah, Nikos Hardavellas, 
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

#include "shader.h"
#include <float.h>
#include <limits.h>
#include <string.h>
#include <memory>
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_sim.h"
#include "../statwrapper.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"
#include "../constants.h"

#include "remodeling/sm.h"
#include "remodeling/new_stats.h"

#define PRIORITIZE_MSHR_OVER_WB 1
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

mem_fetch *shader_core_mem_fetch_allocator::alloc(
    new_addr_type addr, mem_access_type type, unsigned size, bool wr,
    unsigned long long cycle) const {
  mem_access_t access(type, addr, size, wr, m_memory_config->gpgpu_ctx);
  mem_fetch *mf =
      new mem_fetch(access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, -1,
                    m_core_id, m_cluster_id, m_memory_config, cycle);
  return mf;
}

mem_fetch *shader_core_mem_fetch_allocator::alloc(
    new_addr_type addr, mem_access_type type, const active_mask_t &active_mask,
    const mem_access_byte_mask_t &byte_mask,
    const mem_access_sector_mask_t &sector_mask, unsigned size, bool wr,
    unsigned long long cycle, unsigned wid, unsigned sid, unsigned tpc,
    mem_fetch *original_mf) const {
  mem_access_t access(type, addr, size, wr, active_mask, byte_mask, sector_mask,
                      m_memory_config->gpgpu_ctx);
  mem_fetch *mf = new mem_fetch(
      access, NULL, wr ? WRITE_PACKET_SIZE : READ_PACKET_SIZE, wid, m_core_id,
      m_cluster_id, m_memory_config, cycle, original_mf);
  return mf;
}
/////////////////////////////////////////////////////////////////////////////

void check_kernel_launch_limitation(
    const kernel_info_t &k, const shader_core_config *shader_config,
    shader_core_stats *stats) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % shader_config->warp_size)
    padded_cta_size = ((padded_cta_size / shader_config->warp_size) + 1) *
                      (shader_config->warp_size);

  // Limit by n_threads/shader
  unsigned int result_thread =
      shader_config->n_thread_per_shader / padded_cta_size;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);
  unsigned kernel_id = stats->m_last_kernel_id;

  // Limit by shmem/shader
  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    result_shmem = shader_config->gpgpu_shmem_size / kernel_info->smem;

  // Limit by register count, rounded up to multiple of 4.
  unsigned int result_regs = (unsigned)-1;
  unsigned int num_configured_regs = shader_config->gpgpu_shader_registers;
  if(shader_config->is_vpreg_enabled) {
    num_configured_regs = shader_config->vpreg_num_physical_regs_per_sm * 32; // Translate from warp registers to thread registers
  }
  if (kernel_info->regs > 0)
    result_regs = num_configured_regs /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));

  // Limit by CTA
  unsigned int result_cta = shader_config->max_cta_per_core;

  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  result = gs_min2(result, result_regs);
  result = gs_min2(result, result_cta);

  static unsigned last_kernel_id = std::numeric_limits<unsigned>::max();
  
  // Important added lines for stats
  if (last_kernel_id != kernel_id) {  // Only tries to increment the counter if
                                    // kernel_info struct changes
    last_kernel_id = kernel_id;
    if (result == result_regs) stats->total_number_of_kernels_limited_by_regs++;
    if (result == result_shmem) stats->total_number_of_kernels_limited_by_shared_memory++;
    if (result == result_cta) stats->total_number_of_kernels_limited_by_ctas++;
    if (result == result_thread) stats->total_number_of_kernels_limited_by_threads++;
  }
}

// MOD. Begin.
// MOD. End.

// MOD. Begin. Fix WAR at baseline
// MOD. End

// return the next pc of a thread

void gpgpu_sim::get_pdom_stack_top_info(unsigned sid, unsigned tid,
                                        unsigned *pc, unsigned *rpc) {
  unsigned cluster_id = m_shader_config->sid_to_cluster(sid);
  m_cluster[cluster_id]->get_pdom_stack_top_info(sid, tid, pc, rpc);
}

// MOD. Begin. Custom Stats
void shader_core_stats::compute_derived_custom_stats()
{
  long double occupancy_numerator = 0;
  long double weighted_warp_ipc_numerator = 0;
  unsigned long long total_num_sim_winsn_per_kernel = 0;
  unsigned long long sum_shader_cycles = 0;
  unsigned long long sum_sym_cycles = 0;
  unsigned long long sum_shader_cycles_no_separating_kernels = 0;

  total_weighted_average_warp_ipc_between_shaders = 0;
  
  // Memory
  total_avg_usage_l1d_bank = 0;
  int memory_numerator = 0;

  for(unsigned int i = 0; i < m_config->num_shader(); i++)
  {
    shader_occupancy_per_kernel[m_current_kernel_pos][i] = (shader_maximum_theoretical_warps_per_kernel[m_current_kernel_pos][i] ) ?
        ( ((double)shader_active_warps_per_kernel[m_current_kernel_pos][i])/shader_maximum_theoretical_warps_per_kernel[m_current_kernel_pos][i] ) : 0;
    occupancy_numerator += shader_occupancy_per_kernel[m_current_kernel_pos][i] * shader_cycles_per_kernel[m_current_kernel_pos][i];
    
    shader_warp_ipc_per_kernel[m_current_kernel_pos][i] = (shader_cycles_per_kernel[m_current_kernel_pos][i]) ?
        ( ((double)m_num_sim_winsn_per_shader_per_kernel[m_current_kernel_pos][i])/shader_cycles_per_kernel[m_current_kernel_pos][i] ) : 0;
    weighted_warp_ipc_numerator += shader_warp_ipc_per_kernel[m_current_kernel_pos][i] * shader_cycles_per_kernel[m_current_kernel_pos][i];
    total_num_sim_winsn_per_kernel += shader_warp_ipc_per_kernel[m_current_kernel_pos][i];

    shader_warp_ipc_per_shader[i] = (shader_cycles[i]) ? ( ( (double)m_num_sim_winsn_per_shader[i])/shader_cycles[i] ) : 0;
    total_weighted_average_warp_ipc_between_shaders += shader_warp_ipc_per_shader[i] * shader_cycles[i];

    sum_shader_cycles += shader_cycles_per_kernel[m_current_kernel_pos][i];
    sum_shader_cycles_no_separating_kernels += shader_cycles[i];

    // Memory
    for(unsigned int j = 0; j < m_config->m_L1D_config.l1_banks; j++) {
      if (l1d_accesses_per_sid_per_bank[i][j] > 0) {
        memory_numerator++;
        double aux_l1d_avg_usage_per_sid_per_bank = ((double)l1d_accesses_per_sid_per_bank[i][j]) / l1d_evals_per_sid_per_bank[i][j];
        total_avg_usage_l1d_bank += aux_l1d_avg_usage_per_sid_per_bank;
        if (aux_l1d_avg_usage_per_sid_per_bank > max_avg_usage_l1d_bank) {
          max_avg_usage_l1d_bank = aux_l1d_avg_usage_per_sid_per_bank;
        }
      }
    }
  }
  if(memory_numerator > 0) {
    total_avg_usage_l1d_bank = total_avg_usage_l1d_bank / memory_numerator;
  }else {
    total_avg_usage_l1d_bank = 0;
  }

  total_weighted_average_warp_ipc_between_shaders = total_weighted_average_warp_ipc_between_shaders / sum_shader_cycles_no_separating_kernels;

  average_num_shader_active_per_kernel[m_current_kernel_pos] = ((double)sum_shader_cycles) / gpu_cycles_per_kernel[m_current_kernel_pos];
  weighted_average_shader_occupancy_per_kernel[m_current_kernel_pos] = occupancy_numerator / sum_shader_cycles;
  weighted_average_shader_warp_ipc_per_kernel[m_current_kernel_pos] = weighted_warp_ipc_numerator / sum_shader_cycles;

  total_weighted_average_shader_occupancy = 0;
  total_weighted_average_shader_warp_ipc_with_kernels = 0;
  total_weighted_average_num_shader_active = 0;
  total_weighted_average_warps_per_kernel = 0;
  number_of_total_warps = 0;

  for(unsigned int i = 0; i < m_last_kernel_id; i++)
  {
    total_weighted_average_shader_occupancy += weighted_average_shader_occupancy_per_kernel[i] * gpu_cycles_per_kernel[i];
    total_weighted_average_shader_warp_ipc_with_kernels += weighted_average_shader_warp_ipc_per_kernel[i] * gpu_cycles_per_kernel[i];
    total_weighted_average_num_shader_active += average_num_shader_active_per_kernel[i] * gpu_cycles_per_kernel[i];
    total_weighted_average_warps_per_kernel += number_of_warps_per_kernel[i] * gpu_cycles_per_kernel[i];
    number_of_total_warps += number_of_warps_per_kernel[i];

    sum_sym_cycles += gpu_cycles_per_kernel[i];
  }

  total_weighted_average_warps_per_kernel = ((double)total_weighted_average_warps_per_kernel) / sum_sym_cycles;
  total_weighted_average_shader_occupancy = ((double)total_weighted_average_shader_occupancy) / sum_sym_cycles;
  total_weighted_average_shader_warp_ipc_with_kernels = ((double)total_weighted_average_shader_warp_ipc_with_kernels) / sum_sym_cycles;
  total_weighted_average_num_shader_active = ((double)total_weighted_average_num_shader_active) / sum_sym_cycles;
}

void shader_core_stats::print_single_custom_shader_stat_long(FILE *fout, std::string stat_name, std::vector<std::vector<unsigned long long>> vector_stat) const {
  auto it_max = std::max_element(std::begin(vector_stat[m_current_kernel_pos]),std::end(vector_stat[m_current_kernel_pos]));
  auto it_min = std::min_element(std::begin(vector_stat[m_current_kernel_pos]),std::end(vector_stat[m_current_kernel_pos]));
  int pos_max = it_max - vector_stat[m_current_kernel_pos].begin();
  int pos_min = it_min - vector_stat[m_current_kernel_pos].begin();
  int number_of_0s = std::count(std::begin(vector_stat[m_current_kernel_pos]),std::end(vector_stat[m_current_kernel_pos]),0);
  fprintf(fout, "%s summary\n", stat_name.c_str());
  fprintf(fout, "%s_max_pos:%d\t%s_max_val:%lld\n", stat_name.c_str(),pos_max,stat_name.c_str(),*it_max);
  fprintf(fout, "%s_min_pos:%d\t%s_min_val:%lld\n", stat_name.c_str(),pos_min,stat_name.c_str(),*it_min);
  fprintf(fout, "%s_number_of_0s:%d\n", stat_name.c_str(),number_of_0s);
  fprintf(fout, "%s detailed\n", stat_name.c_str());
  for(unsigned int i = 0; i < m_config->num_shader(); i++)
  {
    fprintf(fout,"%s_SM[%d]:%lld\t", stat_name.c_str(), i, vector_stat[m_current_kernel_pos][i]);
  }
  fprintf(fout, "\n");
}

void shader_core_stats::print_single_custom_shader_stat_double(FILE *fout, std::string stat_name, std::vector<std::vector<double>> vector_stat) const {
  auto it_max = std::max_element(std::begin(vector_stat[m_current_kernel_pos]),std::end(vector_stat[m_current_kernel_pos]));
  auto it_min = std::min_element(std::begin(vector_stat[m_current_kernel_pos]),std::end(vector_stat[m_current_kernel_pos]));
  int pos_max = it_max - vector_stat[m_current_kernel_pos].begin();
  int pos_min = it_min - vector_stat[m_current_kernel_pos].begin();
  int number_of_0s = std::count(std::begin(vector_stat[m_current_kernel_pos]),std::end(vector_stat[m_current_kernel_pos]),0);
  fprintf(fout, "%s summary\n", stat_name.c_str());
  fprintf(fout, "%s_max_pos:%d\t%s_max_val:%.4lf\n", stat_name.c_str(),pos_max,stat_name.c_str(),*it_max);
  fprintf(fout, "%s_min_pos:%d\t%s_min_val:%.4lf\n", stat_name.c_str(),pos_min,stat_name.c_str(),*it_min);
  fprintf(fout, "%s_number_of_0s:%d\n", stat_name.c_str(),number_of_0s);
  fprintf(fout, "%s detailed\n", stat_name.c_str());
  for(unsigned int i = 0; i < m_config->num_shader(); i++)
  {
    fprintf(fout,"%s_SM[%d]:%.4lf\t", stat_name.c_str(), i, vector_stat[m_current_kernel_pos][i]);
  }
  fprintf(fout, "\n");
}

void shader_core_stats::print_custom_shader_stats(FILE *fout) const {
  fprintf(fout, "Custom shader stats\n");

  print_single_custom_shader_stat_long(fout,"shader_maximum_theoretical_warps_per_kernel", shader_maximum_theoretical_warps_per_kernel);
  print_single_custom_shader_stat_long(fout,"shader_active_warps_per_kernel", shader_active_warps_per_kernel);
  print_single_custom_shader_stat_long(fout,"shader_cycles_per_kernel", shader_cycles_per_kernel);
  print_single_custom_shader_stat_long(fout,"m_num_sim_winsn_per_shader_per_kernel", m_num_sim_winsn_per_shader_per_kernel);
  print_single_custom_shader_stat_double(fout,"shader_occupancy_per_kernel", shader_occupancy_per_kernel);
  print_single_custom_shader_stat_double(fout,"shader_warp_ipc_per_kernel", shader_warp_ipc_per_kernel);
  fprintf(fout,"number_of_warps_last_kernel = %lld\n",number_of_warps_per_kernel[m_current_kernel_pos]);

  fprintf(fout, "weighted_average_shader_occupancy_per_kernel = %.4lf\n",weighted_average_shader_occupancy_per_kernel[m_current_kernel_pos]);
  fprintf(fout, "weighted_average_shader_warp_ipc_per_kernel = %.4lf\n",weighted_average_shader_warp_ipc_per_kernel[m_current_kernel_pos]);
  fprintf(fout, "average_num_shader_active_per_kernel = %.4lf\n",average_num_shader_active_per_kernel[m_current_kernel_pos]);
  fprintf(fout, "total_weighted_average_shader_occupancy = %.4lf\n",total_weighted_average_shader_occupancy);
  fprintf(fout, "total_weighted_average_shader_warp_ipc_with_kernels = %.4lf\n",total_weighted_average_shader_warp_ipc_with_kernels);
  fprintf(fout, "total_weighted_average_warp_ipc_between_shaders = %.4lf\n",total_weighted_average_warp_ipc_between_shaders);
  fprintf(fout, "total_average_num_shader_active = %.4lf\n",total_weighted_average_num_shader_active);
  fprintf(fout, "total_weighted_average_warps_per_kernel = %.4LF\n",total_weighted_average_warps_per_kernel);
  fprintf(fout, "number_of_total_warps = %lld\n",number_of_total_warps);

  fprintf(fout, "tot_scheduler_cycles = %lld\n", tot_scheduler_cycles);
  fprintf(fout, "tot_scheduler_issues = %lld\n", tot_scheduler_issues);

  double per_cyc_sched_issued = ( ((double) tot_scheduler_issues)/ tot_scheduler_cycles) * 100;
  double per_cyc_sched_stall_idle = ( ((double) shader_cycle_distro[0])/ tot_scheduler_cycles) * 100;
  double per_cyc_sched_stall_dependencies = ( ((double) shader_cycle_distro[1])/ tot_scheduler_cycles) * 100;
  double per_cyc_sche_stall_pipeline = ( ((double) shader_cycle_distro[2])/ tot_scheduler_cycles) * 100;
  double per_cyc_sche_stall_war_scoreboard_dependencies = ( ((double) num_scheduler_stall_cycle_due_to_war_scoreboard)/ tot_scheduler_cycles) * 100;
  double per_cyc_sche_stall_dependencies_other_reasons_not_war_scoreboard = ( ((double) num_scheduler_stall_cycle_dependencies_other_reasons_not_war_scoreboard)/ tot_scheduler_cycles) * 100;
  fprintf(fout, "per_cyc_sched_issued = %.4lf\n", per_cyc_sched_issued);
  fprintf(fout, "per_cyc_sched_stall_idle = %.4lf\n", per_cyc_sched_stall_idle);
  fprintf(fout, "per_cyc_sched_stall_dependencies = %.4lf\n", per_cyc_sched_stall_dependencies);
  fprintf(fout, "per_cyc_sche_stall_pipeline = %.4lf\n", per_cyc_sche_stall_pipeline);
  fprintf(fout, "per_cyc_sche_stall_war_scoreboard_dependencies = %.4lf\n", per_cyc_sche_stall_war_scoreboard_dependencies);
  fprintf(fout, "per_cyc_sche_stall_dependencies_other_reasons_not_war_scoreboard = %.4lf\n", per_cyc_sche_stall_dependencies_other_reasons_not_war_scoreboard);

  fprintf(fout, "tot_num_expected_wb = %lld\n", tot_num_expected_wb);
  fprintf(fout, "tot_num_allocated_wb = %lld\n", tot_num_allocated_wb);
  double percentage_allocated_wb_respect_expected = ( ((double) tot_num_allocated_wb)/ tot_num_expected_wb) * 100;
  fprintf(fout, "percentage_allocated_wb_respect_expected = %.4lf\n", percentage_allocated_wb_respect_expected);

  // MOD. Begin. Fix misaligned fetched instructions
  double per_fetch_instruction_misalignments = tot_fetch_instruction_misalignments ? ( ( ((double) tot_fetch_instruction_misalignments)/ tot_fetch_requests) * 100 ) : 0; // Avoid NaN because there is not any fetch instruction misalignment
  fprintf(fout, "tot_fetch_instruction_misalignments = %lld\n", tot_fetch_instruction_misalignments);
  fprintf(fout, "tot_fetch_requests = %lld\n", tot_fetch_requests);
  fprintf(fout, "per_fetch_instruction_misalignments = %.4lf\n", per_fetch_instruction_misalignments);
  double scoreboard_reads_max_usage_collision = num_scoreboard_reads_collision_due_to_max_uses_per_reg ? ( ( ((double) num_scoreboard_reads_collision_due_to_max_uses_per_reg)/ num_scoreboard_reads_check_collision) * 100 ) : 0; // Avoid NaN because there is not any fetch instruction misalignment
  fprintf(fout, "total_num_scoreboard_reads_check_collision = %u\n", num_scoreboard_reads_check_collision);
  fprintf(fout, "total_num_scoreboard_reads_collision_due_to_max_uses_per_reg = %u\n", num_scoreboard_reads_collision_due_to_max_uses_per_reg);
  fprintf(fout, "total_scoreboard_reads_max_usage_collision = %.4lf\n", scoreboard_reads_max_usage_collision);
  // MOD. End. Fix misaligned fetched instructions

  // MOD. Begin. Memory stats
  fprintf(fout, "total_percentage_avg_usage_l1d_banks = %.4Lf\n", total_avg_usage_l1d_bank * 100) ;
  fprintf(fout, "total_percentage_max_avg_usage_of_l1d_bank = %.4lf\n", max_avg_usage_l1d_bank * 100);
  long double total_avg_usage_shared_mem = total_shared_mem_accesses ? ( ( ((double) total_shared_mem_accesses)/ total_shared_mem_evals) * 100 ) : 0; // Avoid NaN
  fprintf(fout, "total_percentage_avg_usage_shared_mem = %.4Lf\n", total_avg_usage_shared_mem);
  long double total_percentage_ldst_unit_instructions = total_num_ldst_unit_instructions ? ( ( ((double) total_num_ldst_unit_instructions)/ total_num_warp_instructions) * 100 ) : 0; // Avoid NaN
  fprintf(fout, "total_percentage_ldst_unit_instructions = %.4Lf\n", total_percentage_ldst_unit_instructions);
  long double total_percentage_dp_instructions = total_num_dp_instructions ? ( ( ((double) total_num_dp_instructions)/ total_num_warp_instructions) * 100 ) : 0; // Avoid NaN
  fprintf(fout, "total_percentage_dp_instructions = %.4Lf\n", total_percentage_dp_instructions);
  long double total_accesses_per_l1d_instruction = total_l1d_instructions ? ( ( ((double) total_accesses_l1d_instructions)/ total_l1d_instructions)) : 0; // Avoid NaN
  fprintf(fout, "total_accesses_per_l1d_instruction = %.4Lf\n", total_accesses_per_l1d_instruction);
  long double total_avg_cycles_to_schedule_accesses_per_l1d_instruction = total_l1d_instructions ? ( ( ((double) total_avg_cycles_to_schedule_accesses)/ total_l1d_instructions)) : 0; // Avoid NaN
  fprintf(fout, "total_avg_cycles_to_schedule_accesses_per_l1d_instruction = %.4Lf\n", total_avg_cycles_to_schedule_accesses_per_l1d_instruction);
  long double total_conflicts_per_shared_instruction = total_shared_instructions ? ( ( ((double) total_conflicts_shared_instructions)/ total_shared_instructions)) : 0; // Avoid NaN
  fprintf(fout, "total_conflicts_per_shared_instruction = %.4Lf\n", total_conflicts_per_shared_instruction);
  long double total_cyles_in_ldst_unit_dispatch_reg_per_ldst_unit_instruction = total_num_ldst_unit_instructions ? ( ( ((double) total_cycles_instructions_in_ldst_unit_dispatch_reg)/ total_num_ldst_unit_instructions)) : 0; // Avoid NaN
  fprintf(fout, "total_cyles_in_ldst_unit_dispatch_reg_per_ldst_unit_instruction = %.4Lf\n", total_cyles_in_ldst_unit_dispatch_reg_per_ldst_unit_instruction);
  long double total_cyles_in_ldst_unit_arbiter_latch_per_ldst_unit_instruction = total_num_ldst_unit_instructions ? ( ( ((double) total_cycles_instructions_in_ldst_unit_arbiter_latch)/ total_num_ldst_unit_instructions)) : 0; // Avoid NaN
  fprintf(fout, "total_cyles_in_ldst_unit_arbiter_latch_per_ldst_unit_instruction = %.4Lf\n", total_cyles_in_ldst_unit_arbiter_latch_per_ldst_unit_instruction); // MOD. Fixed LDST_Unit model
  // MOD. End. Memory stats
}

void shader_core_stats::print_coalescing_stats(FILE *out) {
  total_l1d_instructions = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_l1d_instructions"]->get_value();
  total_shared_instructions = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_shared_instructions"]->get_value();
  long double avg_accesses_per_l1d_instruction = total_l1d_instructions ? ( ( ((double) m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_accesses_l1d_instructions"]->get_value())/ total_l1d_instructions)) : 0; // Avoid NaN 
  fprintf(out, "total_accesses_per_l1d_instruction = %.4Lf\n", avg_accesses_per_l1d_instruction);
  long double avg_accesses_per_shared_instruction = total_shared_instructions ? ( ( ((double) m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_conflicts_shared_instructions"]->get_value())/ total_shared_instructions)) : 0; // Avoid NaN
  fprintf(out, "total_accesses_per_shared_instruction = %.4Lf\n", avg_accesses_per_shared_instruction);
  unsigned long long total_num_accesses_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_total_eval_accesses;
  unsigned long long total_num_coalesced_intrawarp_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing;
  unsigned long long total_num_coalesced_interwarp_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing;
  unsigned long long total_not_coalesced_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_not_coalesced;

  unsigned long long total_num_coalesced_intrawarp_less_equal_than_5_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_5_cyc;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_10_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_10_cyc + total_num_coalesced_intrawarp_less_equal_than_5_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_20_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_20_cyc + total_num_coalesced_intrawarp_less_equal_than_10_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_30_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_30_cyc + total_num_coalesced_intrawarp_less_equal_than_20_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_40_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_40_cyc + total_num_coalesced_intrawarp_less_equal_than_30_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_50_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_50_cyc + total_num_coalesced_intrawarp_less_equal_than_40_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_100_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_100_cyc + total_num_coalesced_intrawarp_less_equal_than_50_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_200_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_less_or_equal_than_200_cyc + total_num_coalesced_intrawarp_less_equal_than_100_cyc_l1d;
  unsigned long long total_num_coalesced_intrawarp_bigger_than_200_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_intrawarp_coalescing_bigger_than_200_cyc;

  unsigned long long total_num_coalesced_interwarp_less_equal_than_5_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_5_cyc;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_10_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_10_cyc + total_num_coalesced_interwarp_less_equal_than_5_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_20_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_20_cyc + total_num_coalesced_interwarp_less_equal_than_10_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_30_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_30_cyc + total_num_coalesced_interwarp_less_equal_than_20_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_40_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_40_cyc + total_num_coalesced_interwarp_less_equal_than_30_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_50_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_50_cyc + total_num_coalesced_interwarp_less_equal_than_40_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_100_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_100_cyc + total_num_coalesced_interwarp_less_equal_than_50_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_200_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_less_or_equal_than_200_cyc + total_num_coalesced_interwarp_less_equal_than_100_cyc_l1d;
  unsigned long long total_num_coalesced_interwarp_bigger_than_200_cyc_l1d = m_gpu->m_coalescing_stats_across_sms_l1d.m_num_interwarp_coalescing_bigger_than_200_cyc;

  long double total_percentage_coalesced_intrawarp_l1d = total_num_accesses_l1d ? ( ( ((double) total_num_coalesced_intrawarp_l1d)/ total_num_accesses_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_l1d = total_num_accesses_l1d ? ( ( ((double) total_num_coalesced_interwarp_l1d)/ total_num_accesses_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_not_coalesced_l1d = total_num_accesses_l1d ? ( ( ((double) total_not_coalesced_l1d)/ total_num_accesses_l1d) * 100 ) : 0; // Avoid NaN
  
  long double total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_5_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_10_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_20_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_30_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_40_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_50_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_100_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_200_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_bigger_than_200_cyc_l1d = total_num_coalesced_intrawarp_l1d ? ( ( ((double) total_num_coalesced_intrawarp_bigger_than_200_cyc_l1d)/ total_num_coalesced_intrawarp_l1d) * 100 ) : 0; // Avoid NaN

  long double total_percentage_coalesced_interwarp_less_equal_than_5_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_5_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_10_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_10_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_20_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_20_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_30_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_30_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_40_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_40_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_50_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_50_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_100_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_100_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_200_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_200_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_bigger_than_200_cyc_l1d = total_num_coalesced_interwarp_l1d ? ( ( ((double) total_num_coalesced_interwarp_bigger_than_200_cyc_l1d)/ total_num_coalesced_interwarp_l1d) * 100 ) : 0; // Avoid NaN

  fprintf(out, "total_percentage_coalesced_intrawarp_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_l1d);
  fprintf(out, "total_percentage_not_coalesced_l1d = %.4Lf\n", total_percentage_not_coalesced_l1d);

  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_intrawarp_bigger_than_200_cyc_l1d = %.4Lf\n", total_percentage_coalesced_intrawarp_bigger_than_200_cyc_l1d);

  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_5_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_5_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_10_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_10_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_20_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_20_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_30_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_30_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_40_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_40_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_50_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_50_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_100_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_100_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_200_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_200_cyc_l1d);
  fprintf(out, "total_percentage_coalesced_interwarp_bigger_than_200_cyc_l1d = %.4Lf\n", total_percentage_coalesced_interwarp_bigger_than_200_cyc_l1d);

  unsigned long long total_num_accesses_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_total_eval_accesses;
  unsigned long long total_num_coalesced_intrawarp_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing;
  unsigned long long total_num_coalesced_interwarp_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing;
  unsigned long long total_not_coalesced_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_not_coalesced;

  unsigned long long total_num_coalesced_intrawarp_less_equal_than_5_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_5_cyc;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_10_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_10_cyc + total_num_coalesced_intrawarp_less_equal_than_5_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_20_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_20_cyc + total_num_coalesced_intrawarp_less_equal_than_10_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_30_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_30_cyc + total_num_coalesced_intrawarp_less_equal_than_20_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_40_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_40_cyc + total_num_coalesced_intrawarp_less_equal_than_30_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_50_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_50_cyc + total_num_coalesced_intrawarp_less_equal_than_40_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_100_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_100_cyc + total_num_coalesced_intrawarp_less_equal_than_50_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_200_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_less_or_equal_than_200_cyc + total_num_coalesced_intrawarp_less_equal_than_100_cyc_const;
  unsigned long long total_num_coalesced_intrawarp_bigger_than_200_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_intrawarp_coalescing_bigger_than_200_cyc;

  unsigned long long total_num_coalesced_interwarp_less_equal_than_5_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_5_cyc;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_10_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_10_cyc + total_num_coalesced_interwarp_less_equal_than_5_cyc_const;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_20_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_20_cyc + total_num_coalesced_interwarp_less_equal_than_10_cyc_const;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_30_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_30_cyc + total_num_coalesced_interwarp_less_equal_than_20_cyc_const;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_40_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_40_cyc + total_num_coalesced_interwarp_less_equal_than_30_cyc_const;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_50_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_50_cyc + total_num_coalesced_interwarp_less_equal_than_40_cyc_const;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_100_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_100_cyc + total_num_coalesced_interwarp_less_equal_than_50_cyc_const;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_200_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_less_or_equal_than_200_cyc + total_num_coalesced_interwarp_less_equal_than_100_cyc_const;
  unsigned long long total_num_coalesced_interwarp_bigger_than_200_cyc_const = m_gpu->m_coalescing_stats_across_sms_const.m_num_interwarp_coalescing_bigger_than_200_cyc;

  long double total_percentage_coalesced_intrawarp_const = total_num_accesses_const ? ( ( ((double) total_num_coalesced_intrawarp_const)/ total_num_accesses_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_const = total_num_accesses_const ? ( ( ((double) total_num_coalesced_interwarp_const)/ total_num_accesses_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_not_coalesced_const = total_num_accesses_const ? ( ( ((double) total_not_coalesced_const)/ total_num_accesses_const) * 100 ) : 0; // Avoid NaN
  
  long double total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_5_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_10_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_20_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_30_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_40_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_50_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_100_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_200_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_bigger_than_200_cyc_const = total_num_coalesced_intrawarp_const ? ( ( ((double) total_num_coalesced_intrawarp_bigger_than_200_cyc_const)/ total_num_coalesced_intrawarp_const) * 100 ) : 0; // Avoid NaN

  long double total_percentage_coalesced_interwarp_less_equal_than_5_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_5_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_10_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_10_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_20_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_20_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_30_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_30_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_40_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_40_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_50_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_50_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_100_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_100_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_200_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_200_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_bigger_than_200_cyc_const = total_num_coalesced_interwarp_const ? ( ( ((double) total_num_coalesced_interwarp_bigger_than_200_cyc_const)/ total_num_coalesced_interwarp_const) * 100 ) : 0; // Avoid NaN

  fprintf(out, "total_percentage_coalesced_intrawarp_const = %.4Lf\n", total_percentage_coalesced_intrawarp_const);
  fprintf(out, "total_percentage_coalesced_interwarp_const = %.4Lf\n", total_percentage_coalesced_interwarp_const);
  fprintf(out, "total_percentage_not_coalesced_const = %.4Lf\n", total_percentage_not_coalesced_const);

  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_const);
  fprintf(out, "total_percentage_coalesced_intrawarp_bigger_than_200_cyc_const = %.4Lf\n", total_percentage_coalesced_intrawarp_bigger_than_200_cyc_const);

  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_5_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_5_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_10_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_10_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_20_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_20_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_30_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_30_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_40_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_40_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_50_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_50_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_100_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_100_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_200_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_200_cyc_const);
  fprintf(out, "total_percentage_coalesced_interwarp_bigger_than_200_cyc_const = %.4Lf\n", total_percentage_coalesced_interwarp_bigger_than_200_cyc_const);

  unsigned long long total_num_accesses_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_total_eval_accesses;
  unsigned long long total_num_coalesced_intrawarp_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing;
  unsigned long long total_num_coalesced_interwarp_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing;
  unsigned long long total_not_coalesced_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_not_coalesced;

  unsigned long long total_num_coalesced_intrawarp_less_equal_than_5_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_5_cyc;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_10_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_10_cyc + total_num_coalesced_intrawarp_less_equal_than_5_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_20_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_20_cyc + total_num_coalesced_intrawarp_less_equal_than_10_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_30_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_30_cyc + total_num_coalesced_intrawarp_less_equal_than_20_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_40_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_40_cyc + total_num_coalesced_intrawarp_less_equal_than_30_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_50_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_50_cyc + total_num_coalesced_intrawarp_less_equal_than_40_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_100_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_100_cyc + total_num_coalesced_intrawarp_less_equal_than_50_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_less_equal_than_200_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_less_or_equal_than_200_cyc + total_num_coalesced_intrawarp_less_equal_than_100_cyc_sharedmem;
  unsigned long long total_num_coalesced_intrawarp_bigger_than_200_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_intrawarp_coalescing_bigger_than_200_cyc;

  unsigned long long total_num_coalesced_interwarp_less_equal_than_5_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_5_cyc;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_10_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_10_cyc + total_num_coalesced_interwarp_less_equal_than_5_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_20_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_20_cyc + total_num_coalesced_interwarp_less_equal_than_10_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_30_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_30_cyc + total_num_coalesced_interwarp_less_equal_than_20_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_40_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_40_cyc + total_num_coalesced_interwarp_less_equal_than_30_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_50_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_50_cyc + total_num_coalesced_interwarp_less_equal_than_40_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_100_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_100_cyc + total_num_coalesced_interwarp_less_equal_than_50_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_less_equal_than_200_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_less_or_equal_than_200_cyc + total_num_coalesced_interwarp_less_equal_than_100_cyc_sharedmem;
  unsigned long long total_num_coalesced_interwarp_bigger_than_200_cyc_sharedmem = m_gpu->m_coalescing_stats_across_sms_sharedmem.m_num_interwarp_coalescing_bigger_than_200_cyc;

  long double total_percentage_coalesced_intrawarp_sharedmem = total_num_accesses_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_sharedmem)/ total_num_accesses_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_sharedmem = total_num_accesses_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_sharedmem)/ total_num_accesses_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_not_coalesced_sharedmem = total_num_accesses_sharedmem ? ( ( ((double) total_not_coalesced_sharedmem)/ total_num_accesses_sharedmem) * 100 ) : 0; // Avoid NaN
  
  long double total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_5_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_10_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_20_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_30_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_40_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_50_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_100_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_less_equal_than_200_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_intrawarp_bigger_than_200_cyc_sharedmem = total_num_coalesced_intrawarp_sharedmem ? ( ( ((double) total_num_coalesced_intrawarp_bigger_than_200_cyc_sharedmem)/ total_num_coalesced_intrawarp_sharedmem) * 100 ) : 0; // Avoid NaN

  long double total_percentage_coalesced_interwarp_less_equal_than_5_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_5_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_10_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_10_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_20_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_20_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_30_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_30_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_40_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_40_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_50_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_50_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_100_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_100_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_less_equal_than_200_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_less_equal_than_200_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN
  long double total_percentage_coalesced_interwarp_bigger_than_200_cyc_sharedmem = total_num_coalesced_interwarp_sharedmem ? ( ( ((double) total_num_coalesced_interwarp_bigger_than_200_cyc_sharedmem)/ total_num_coalesced_interwarp_sharedmem) * 100 ) : 0; // Avoid NaN

  fprintf(out, "total_percentage_coalesced_intrawarp_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_sharedmem);
  fprintf(out, "total_percentage_not_coalesced_sharedmem = %.4Lf\n", total_percentage_not_coalesced_sharedmem);

  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_5_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_10_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_20_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_30_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_40_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_50_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_100_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_less_equal_than_200_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_intrawarp_bigger_than_200_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_intrawarp_bigger_than_200_cyc_sharedmem);

  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_5_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_5_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_10_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_10_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_20_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_20_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_30_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_30_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_40_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_40_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_50_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_50_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_100_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_100_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_less_equal_than_200_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_less_equal_than_200_cyc_sharedmem);
  fprintf(out, "total_percentage_coalesced_interwarp_bigger_than_200_cyc_sharedmem = %.4Lf\n", total_percentage_coalesced_interwarp_bigger_than_200_cyc_sharedmem);

  unsigned long long total_accesses_coalesced = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_accesses_coalesced"]->get_value();
  unsigned long long total_accesses_not_coalesced = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_accesses_not_coalesced"]->get_value();
  unsigned long long total_accesses = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_accesses"]->get_value();
  unsigned long long total_accesses_candidate_to_coalesce = total_accesses_coalesced + total_accesses_not_coalesced;
  long double total_percentage_accesses_candidate_to_coalesce = total_accesses ? ( ( ((double) total_accesses_candidate_to_coalesce)/ total_accesses) * 100 ) : 0; // Avoid NaN
  long double total_percentage_accesses_coalesced = total_accesses_candidate_to_coalesce ? ( ( ((double) total_accesses_coalesced)/ total_accesses_candidate_to_coalesce) * 100 ) : 0; // Avoid NaN

  fprintf(out, "total_percentage_accesses_candidate_to_coalesce = %.4Lf\n", total_percentage_accesses_candidate_to_coalesce);
  fprintf(out, "total_percentage_accesses_coalesced = %.4Lf\n", total_percentage_accesses_coalesced);
}

// MOD. Begin. Remodeling
void shader_core_stats::print_remodeling_stats(FILE *fout) {
  total_num_warp_instructions = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_warp_instructions"]->get_value();
  total_num_ldst_unit_instructions = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_ldst_unit_instructions"]->get_value();
  total_num_dp_instructions = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_dp_instructions"]->get_value();
  long double total_percentage_ldst_unit_instructions = total_num_ldst_unit_instructions ? ( ( ((double) total_num_ldst_unit_instructions)/ total_num_warp_instructions) * 100 ) : 0; // Avoid NaN
  fprintf(fout, "total_percentage_ldst_unit_instructions = %.4Lf\n", total_percentage_ldst_unit_instructions);
  long double total_percentage_dp_instructions = total_num_dp_instructions ? ( ( ((double) total_num_dp_instructions)/ total_num_warp_instructions) * 100 ) : 0; // Avoid NaN
  fprintf(fout, "total_percentage_dp_instructions = %.4Lf\n", total_percentage_dp_instructions);
  total_num_cycles_issue_stage_issuing = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_issuing"]->get_value();
  unsigned long long total_num_cycles_issue_stage_stall_next_stage_not_available = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_next_stage_not_available"]->get_value();
  total_num_cycles_issue_stage_evaluated = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_evaluated"]->get_value();
  fprintf(fout, "total_percentage_cycles_issue_stage_issuing = %.4Lf\n", ((long double) total_num_cycles_issue_stage_issuing / total_num_cycles_issue_stage_evaluated) * 100);
  fprintf(fout, "total_percentage_cycles_issue_stage_not_issuing_stall_next_stage_not_available = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_next_stage_not_available / total_num_cycles_issue_stage_evaluated) * 100);

  fprintf(fout, "total_num_constant_cache_different_blocks = %zu\n", all_const_cache_accessed_blocks.size());
  fprintf(fout, "total_num_global_memory_blocks = %zu\n", all_global_memory_accessed_blocks.size());
  fprintf(fout, "total_num_different_virtual_pages = %zu\n", all_virtual_pages_accessed.size());

  unsigned long long total_num_evals_rf = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_evals_rf"]->get_value();
  unsigned long long total_num_evals_rf_with_conflict = m_gpu-> m_gpu_per_sm_stats.m_stats_map["total_num_evals_rf_with_conflict"]->get_value();
  fprintf(fout, "total_percentage_evals_rf_with_conflict = %.4Lf\n", ((long double) total_num_evals_rf_with_conflict / total_num_evals_rf) * 100);
  // OLD
  // fprintf(fout, "total_num_register_file_cache_hits = %lld\n", total_num_register_file_cache_hits);
  // fprintf(fout, "total_num_register_file_cache_allocations = %lld\n", total_num_register_file_cache_allocations);
  // long double total_ratio_hits_per_allocation_in_rfc = total_num_register_file_cache_allocations ? ( ( ((double) total_num_register_file_cache_hits)/ total_num_register_file_cache_allocations)) : 0; // Avoid NaN
  // fprintf(fout, "total_percentage_hits_per_allocation_in_register_file_cache = %.4Lf\n", total_ratio_hits_per_allocation_in_rfc * 100);
  // fprintf(fout, "total_num_regular_regfile_reads = %lld\n", total_num_regular_regfile_reads);
  // fprintf(fout, "total_num_regular_regfile_writes = %lld\n", total_num_regular_regfile_writes);
  // fprintf(fout, "total_num_uniform_regfile_reads = %lld\n", total_num_uniform_regfile_reads);
  // fprintf(fout, "total_num_uniform_regfile_writes = %lld\n", total_num_uniform_regfile_writes);
  // fprintf(fout, "total_num_predicate_regfile_reads = %lld\n", total_num_predicate_regfile_reads);
  // fprintf(fout, "total_num_predicate_regfile_writes = %lld\n", total_num_predicate_regfile_writes);
  // fprintf(fout, "total_num_uniform_predicate_regfile_reads = %lld\n", total_num_uniform_predicate_regfile_reads);
  // fprintf(fout, "total_num_uniform_predicate_regfile_writes = %lld\n", total_num_uniform_predicate_regfile_writes);
  // fprintf(fout, "total_num_constant_cache_reads = %lld\n", total_num_constant_cache_reads);
  // 

  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied = m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied"]->get_value();
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier = m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier"]->get_value();
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier = m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier"]->get_value();
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield = m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield"]->get_value();
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count = m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count"]->get_value();
  unsigned long long total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c = m_gpu->m_gpu_per_sm_stats.m_stats_map["total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c"]->get_value();

  fprintf(fout, "total_num_cycles_issue_stage_evaluated = %lld\n", total_num_cycles_issue_stage_evaluated);
  fprintf(fout, "total_num_cycles_issue_stage_issuing = %lld\n", total_num_cycles_issue_stage_issuing);
  fprintf(fout, "total_num_cycles_issue_stage_stall_issue_port_busy = %lld\n", total_num_cycles_issue_stage_stall_issue_port_busy);
  fprintf(fout, "total_num_cycles_issue_stage_stall_no_valid_instruction = %lld\n", total_num_cycles_issue_stage_stall_no_valid_instruction);
  fprintf(fout, "total_num_cycles_issue_stage_stall_no_warps_ready = %lld\n", total_num_cycles_issue_stage_stall_no_warps_ready);
  fprintf(fout, "total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied = %lld\n", total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied);
  fprintf(fout, "total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier = %lld\n", total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier);
  fprintf(fout, "total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier = %lld\n", total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier);
  fprintf(fout, "total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield = %lld\n", total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield);
  fprintf(fout, "total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count = %lld\n", total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count);
  fprintf(fout, "total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c = %lld\n", total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c);
  // fprintf(fout, "total_num_kernel_not_in_binary = %u\n", num_kernel_not_in_binary);
  
  // fprintf(fout, "total_percentage_cycles_issue_stage_issuing = %.4Lf\n", ((long double) total_num_cycles_issue_stage_issuing / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_issue_port_busy = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_issue_port_busy / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_no_valid_instruction = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_no_valid_instruction / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_no_warps_ready = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_no_warps_ready / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_at_least_one_warp_with_fu_occupied / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_inst_barrier / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_wait_barrier / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_at_least_one_warp_waiting_yield = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_yield / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_stall_count / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c = %.4Lf\n", ((long double) total_num_cycles_issue_stage_stall_at_least_one_warp_waiting_l1c / total_num_cycles_issue_stage_evaluated) * 100);
  // fprintf(fout, "total_percentage_num_kernel_not_in_binary = %.4lf\n", ((double) num_kernel_not_in_binary / m_last_kernel_id) * 100);
  // fprintf(fout, "total_percentage_conflicts_with_rf_bank_write_port = %.4Lf\n", ((long double) total_num_times_wb_port_conflict / total_num_times_wb_evaluated) * 100);
}
// MOD. End. Remodeling

// MOD. End

// MOD. Begin. IBuffer_ooo
void shader_core_stats::compute_ibuffer_ooo_stats() {
  last_ins_issued_per_kernel_per_sid_per_warp = 0;
  last_ins_released_wb_per_kernel_per_sid_per_warp = 0;
  last_ins_released_opc_per_kernel_per_sid_per_warp = 0;
  last_num_flushes_kernel_per_sid_per_warp = 0;
  last_num_times_ibooo_empty = 0;
  last_num_times_ibooo_empty_evaluated = 0;
  last_num_times_ibooo_full = 0;
  last_num_times_fetch_ibooo_tried = 0;
  for(unsigned int i = 0; i < m_config->num_shader(); i++)
  {
    for(unsigned int j = 0; j < m_config->max_warps_per_shader; j++)
    {
      last_ins_issued_per_kernel_per_sid_per_warp += ins_issued_per_kernel_per_sid_per_warp[ins_issued_per_kernel_per_sid_per_warp.size()-1][i][j];
      last_ins_released_wb_per_kernel_per_sid_per_warp += ins_released_wb_per_kernel_per_sid_per_warp[ins_released_wb_per_kernel_per_sid_per_warp.size()-1][i][j];
      last_ins_released_opc_per_kernel_per_sid_per_warp += ins_released_opc_per_kernel_per_sid_per_warp[ins_released_opc_per_kernel_per_sid_per_warp.size()-1][i][j];
      last_num_flushes_kernel_per_sid_per_warp += num_flushes_kernel_per_sid_per_warp[num_flushes_kernel_per_sid_per_warp.size()-1][i][j];
      last_num_times_ibooo_empty += num_times_ibooo_empty[num_times_ibooo_empty.size()-1][i][j];
      last_num_times_ibooo_empty_evaluated += num_times_ibooo_empty_evaluated[num_times_ibooo_empty_evaluated.size()-1][i][j];
      last_num_times_ibooo_full += num_times_ibooo_full[num_times_ibooo_full.size()-1][i][j];
      last_num_times_fetch_ibooo_tried += num_times_fetch_ibooo_tried[num_times_fetch_ibooo_tried.size()-1][i][j];
    }
  }
  total_ins_issued_per_kernel_per_sid_per_warp += last_ins_issued_per_kernel_per_sid_per_warp;
  total_ins_released_wb_per_kernel_per_sid_per_warp += last_ins_released_wb_per_kernel_per_sid_per_warp;
  total_ins_released_opc_per_kernel_per_sid_per_warp += last_ins_released_opc_per_kernel_per_sid_per_warp;
  total_num_flushes_kernel_per_sid_per_warp += last_num_flushes_kernel_per_sid_per_warp;
  total_num_times_ibooo_empty += last_num_times_ibooo_empty;
  total_num_times_ibooo_empty_evaluated += last_num_times_ibooo_empty_evaluated;
  total_num_times_ibooo_full += last_num_times_ibooo_full;
  total_num_times_fetch_ibooo_tried += last_num_times_fetch_ibooo_tried;

  last_percentage_ibooo_empty = ((long double) last_num_times_ibooo_empty / last_num_times_ibooo_empty_evaluated) * 100;
  last_percentage_ibooo_full = ((long double) last_num_times_ibooo_full / last_num_times_fetch_ibooo_tried) * 100;
  total_percentage_ibooo_empty = ((long double) total_num_times_ibooo_empty / total_num_times_ibooo_empty_evaluated) * 100;
  total_percentage_ibooo_full = ((long double) total_num_times_ibooo_full / total_num_times_fetch_ibooo_tried) * 100;
}

void shader_core_stats::print_ibuffer_ooo_stats(FILE *fout) const {
  fprintf(fout, "IBuffer_ooo stats\n");
  // for(int i = 0; i < m_config->num_shader(); i++)
  // {
  //   for(int j = 0; j < m_config->max_warps_per_shader; j++)
  //   {
  //     fprintf(fout, "sid_warp[%d][%d]. issued: %lld, released_wb: %lld, released_opc: %lld, flushes: %lld\n", i, j,
  //       ins_issued_per_kernel_per_sid_per_warp[ins_issued_per_kernel_per_sid_per_warp.size()-1][i][j],
  //       ins_released_wb_per_kernel_per_sid_per_warp[ins_released_wb_per_kernel_per_sid_per_warp.size()-1][i][j],
  //       ins_released_opc_per_kernel_per_sid_per_warp[ins_released_opc_per_kernel_per_sid_per_warp.size()-1][i][j],
  //       num_flushes_kernel_per_sid_per_warp[num_flushes_kernel_per_sid_per_warp.size()-1][i][j]);
  //   }
  // }
  fprintf(fout, "last_ins_issued_per_kernel_per_sid_per_warp = %lld\n", last_ins_issued_per_kernel_per_sid_per_warp);
  fprintf(fout, "last_ins_released_wb_per_kernel_per_sid_per_warp = %lld\n", last_ins_released_wb_per_kernel_per_sid_per_warp);
  fprintf(fout, "last_ins_released_opc_per_kernel_per_sid_per_warp = %lld\n", last_ins_released_opc_per_kernel_per_sid_per_warp);
  fprintf(fout, "last_num_flushes_kernel_per_sid_per_warp = %lld\n", last_num_flushes_kernel_per_sid_per_warp);
  fprintf(fout, "last_num_times_ibooo_empty = %lld\n", last_num_times_ibooo_empty);
  fprintf(fout, "last_num_times_ibooo_empty_evaluated = %lld\n", last_num_times_ibooo_empty_evaluated);
  fprintf(fout, "last_num_times_ibooo_full = %lld\n", last_num_times_ibooo_full);
  fprintf(fout, "last_num_times_fetch_ibooo_tried = %lld\n", last_num_times_fetch_ibooo_tried);
  fprintf(fout, "last_percentage_ibooo_empty = %.4lf\n", last_percentage_ibooo_empty);
  fprintf(fout, "last_percentage_ibooo_full = %.4lf\n", last_percentage_ibooo_full);

  fprintf(fout, "total_ins_issued_per_kernel_per_sid_per_warp = %lld\n", total_ins_issued_per_kernel_per_sid_per_warp);
  fprintf(fout, "total_ins_released_wb_per_kernel_per_sid_per_warp = %lld\n", total_ins_released_wb_per_kernel_per_sid_per_warp);
  fprintf(fout, "total_ins_released_opc_per_kernel_per_sid_per_warp = %lld\n", total_ins_released_opc_per_kernel_per_sid_per_warp);
  fprintf(fout, "total_num_flushes_kernel_per_sid_per_warp = %lld\n", total_num_flushes_kernel_per_sid_per_warp);
  fprintf(fout, "total_num_times_ibooo_empty = %lld\n", total_num_times_ibooo_empty);
  fprintf(fout, "total_num_times_ibooo_empty_evaluated = %lld\n", total_num_times_ibooo_empty_evaluated);
  fprintf(fout, "total_num_times_ibooo_full = %lld\n", total_num_times_ibooo_full);
  fprintf(fout, "total_num_times_fetch_ibooo_tried = %lld\n", total_num_times_fetch_ibooo_tried);
  fprintf(fout, "total_percentage_ibooo_empty = %.4lf\n", total_percentage_ibooo_empty);
  fprintf(fout, "total_percentage_ibooo_full = %.4lf\n", total_percentage_ibooo_full);
  fprintf(fout, "total_num_barriers = %lld\n", total_num_barriers);
  fprintf(fout, "total_num_returns = %lld\n", total_num_returns);
  fprintf(fout, "total_num_branches = %lld\n", total_num_branches);
  fprintf(fout, "total_num_jumps = %lld\n", total_num_jumps);
  fprintf(fout, "total_num_warpsyncs = %lld\n", total_num_warpsyncs);
  fprintf(fout, "total_num_bsyncs = %lld\n", total_num_bsyncs);
  fprintf(fout, "total_num_rpcmovs = %lld\n", total_num_rpcmovs);
  fprintf(fout, "total_num_yields = %lld\n", total_num_yields);
  fprintf(fout, "total_num_barriers_and_controlflows = %lld\n", total_num_barriers_and_controlflows);

  fprintf(fout, "total_instructions_inserted_in_ibooo = %lld\n", total_instructions_inserted_in_ibooo);
  fprintf(fout, "total_war_waw_dependencies = %lld\n", total_war_waw_dependencies);
  fprintf(fout, "total_raw_dependencies = %lld\n", total_raw_dependencies);
  fprintf(fout, "total_stop_point_dependencies = %lld\n", total_stop_point_dependencies);
  fprintf(fout, "total_memory_reordering_dependencies = %lld\n", total_memory_reordering_dependencies);
  fprintf(fout, "total_war_waw_dependencies_per_decoded_instructions = %.4lf\n", double(total_war_waw_dependencies) / total_instructions_inserted_in_ibooo);
  fprintf(fout, "total_raw_dependencies_per_decoded_instructions = %.4lf\n", double(total_raw_dependencies) / total_instructions_inserted_in_ibooo);
  fprintf(fout, "total_stop_point_dependencies_per_decoded_instructions = %.4lf\n", double(total_stop_point_dependencies) / total_instructions_inserted_in_ibooo);
  fprintf(fout, "total_memory_reordering_dependencies_per_decoded_instructions = %.4lf\n", double(total_memory_reordering_dependencies) / total_instructions_inserted_in_ibooo);

  double total_avg_ibooo_num_entries_valid_and_not_issued = ((long double) total_ibooo_num_entries_valid_and_not_issued / total_ibooo_evaluations_compute_selection_stats);
  double total_avg_ibooo_num_entries_valid_not_issued_and_ready = ((long double) total_ibooo_num_entries_valid_not_issued_and_ready / total_ibooo_evaluations_compute_selection_stats);
  double total_avg_ibooo_num_entries = ((long double) total_ibooo_num_entries / total_ibooo_evaluations_compute_selection_stats);
  double total_percentage_times_without_any_candidate = ((long double) total_ibooo_num_times_without_any_candidate / total_ibooo_evaluations_compute_selection_stats) * 100;
  double total_percentage_times_without_any_ready_candidate = ((long double) total_ibooo_num_times_without_any_ready_candidate / total_ibooo_evaluations_compute_selection_stats) * 100;
  fprintf(fout, "total_avg_ibooo_num_entries_valid_and_not_issued = %.4lf\n", total_avg_ibooo_num_entries_valid_and_not_issued);
  fprintf(fout, "total_avg_ibooo_num_entries_valid_not_issued_and_ready = %.4lf\n", total_avg_ibooo_num_entries_valid_not_issued_and_ready);
  fprintf(fout, "total_avg_ibooo_num_entries = %.4lf\n", total_avg_ibooo_num_entries);
  fprintf(fout, "total_percentage_times_without_any_candidate = %.4lf\n", total_percentage_times_without_any_candidate);
  fprintf(fout, "total_percentage_times_without_any_ready_candidate = %.4lf\n", total_percentage_times_without_any_ready_candidate);
}

// MOD. End. IBuffer_ooo

// MOD. Begin. VPREG
void shader_core_stats::print_vpreg_stats(FILE *fout) const {
  fprintf(fout, "total_number_of_kernels_limited_by_regs = %u\n", total_number_of_kernels_limited_by_regs);
  fprintf(fout, "total_number_of_kernels_limited_by_ctas = %u\n", total_number_of_kernels_limited_by_ctas);
  fprintf(fout, "total_number_of_kernels_limited_by_threads = %u\n", total_number_of_kernels_limited_by_threads);
  fprintf(fout, "total_number_of_kernels_limited_by_shared_memory = %u\n", total_number_of_kernels_limited_by_shared_memory);
  fprintf(fout, "total_percentage_of_kernels_limited_by_regs = %.4f\n", (  (static_cast<double>(total_number_of_kernels_limited_by_regs) / m_last_kernel_id) * 100  )  );
  fprintf(fout, "total_percentage_of_kernels_limited_by_ctas = %.4f\n", (  (static_cast<double>(total_number_of_kernels_limited_by_ctas) / m_last_kernel_id) * 100  )  );
  fprintf(fout, "total_percentage_of_kernels_limited_by_threads = %.4f\n", (  (static_cast<double>(total_number_of_kernels_limited_by_threads) / m_last_kernel_id) * 100  )  );
  fprintf(fout, "total_percentage_of_kernels_limited_by_shared_memory = %.4f\n", (  (static_cast<double>(total_number_of_kernels_limited_by_shared_memory) / m_last_kernel_id) * 100  )  );
  fprintf(fout, "total_number_of_opc_conflicts = %lld\n", total_number_of_opc_conflicts);
  fprintf(fout, "total_number_of_opc_requests = %lld\n", total_number_of_opc_requests);
  fprintf(fout, "total_percentage_of_opc_conflicts = %.4f\n", (  (static_cast<double>(total_number_of_opc_conflicts) / total_number_of_opc_requests) * 100  )  );
  fprintf(fout, "total_cycles_instructions_in_cu = %lld\n", total_cycles_instructions_in_cu); // MOD. CU stats

  //MOD. OPC custom stats
  fprintf(fout, "total_num_times_cu_subcore_custom_stats_evaluated = %lld\n", num_times_cu_subcore_custom_stats_evaluated);
  fprintf(fout, "total_num_times_no_cu_dispatched = %lld\n", num_times_no_cu_dispatched);
  fprintf(fout, "total_num_times_no_cu_allocated = %lld\n", num_times_no_cu_allocated);
  fprintf(fout, "total_num_times_no_cu_allocated_and_nothing_to_allocate = %lld\n", num_times_no_cu_allocated_and_nothing_to_allocate);
  fprintf(fout, "total_num_times_no_cu_allocated_due_to_cus_are_full = %lld\n", num_times_no_cu_allocated_due_to_cus_are_full);
  fprintf(fout, "total_num_times_no_cu_dispatched_due_to_dispatch_reg_full = %lld\n", num_times_no_cu_dispatched_due_to_dispatch_reg_full);
  fprintf(fout, "total_num_times_no_cu_dispatched_due_to_no_ready_operands = %lld\n", num_times_no_cu_dispatched_due_to_no_ready_operands);
  fprintf(fout, "total_num_times_no_cu_dispatched_due_to_all_cus_empty = %lld\n", num_times_no_cu_dispatched_due_to_all_cus_empty);
  fprintf(fout, "total_num_times_no_cu_dispatched_and_all_cus_full = %lld\n", num_times_no_cu_dispatched_and_all_cus_full);
  fprintf(fout, "total_num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready = %lld\n", num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready);
  fprintf(fout, "total_num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready = %lld\n", num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready);
  fprintf(fout, "total_num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled = %lld\n", num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled);
  fprintf(fout, "total_num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled = %lld\n", num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled);
  fprintf(fout, "total_percentage_of_no_cu_dispatched = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_due_to_all_cus_empty = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_due_to_all_cus_empty) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_and_all_cus_full = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_and_all_cus_full) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_and_all_cus_full_and_not_any_ready = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_and_all_cus_full_and_not_any_ready) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled_when_dispatch_reg_is_full = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_and_all_cus_full_and_at_least_one_ready_mem_dispatch_full_and_ldst_unit_stalled) / num_times_no_cu_dispatched_due_to_dispatch_reg_full) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_allocated = %.4f\n", (  (static_cast<double>(num_times_no_cu_allocated) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_allocated_and_nothing_to_allocate = %.4f\n", (  (static_cast<double>(num_times_no_cu_allocated_and_nothing_to_allocate) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_allocated_due_to_cus_are_full = %.4f\n", (  (static_cast<double>(num_times_no_cu_allocated_due_to_cus_are_full) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  // These three stats can have a % bigger than because they can be incremented more than than once than the denominator
  fprintf(fout, "total_percentage_of_no_cu_dispatched_due_to_dispatch_reg_full = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_due_to_dispatch_reg_full) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_due_to_no_ready_operands = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_due_to_no_ready_operands) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled = %.4f\n", (  (static_cast<double>(num_times_no_cu_dispatched_due_to_dispatch_reg_full_is_mem_op_and_ldst_unit_stalled) / num_times_cu_subcore_custom_stats_evaluated) * 100  )  );
  fprintf(fout, "total_percentage_of_ldst_unit_dispatches_failed_due_to_not_empty_dispatch_reg = %.4f\n", (  (static_cast<double>(total_num_ldst_unit_dispatches_failed_due_to_not_empty_dispatch_reg) / total_num_try_ldst_unit_dispatches) * 100  )  );

  // MOD. OPC custom stats

  fprintf(fout, "total_number_of_vpreg_decode_rollbacks = %u\n", total_number_of_vpreg_decode_rollbacks);
  fprintf(fout, "total_number_of_vpreg_reissues = %u\n", total_number_of_vpreg_reissues);
  fprintf(fout, "total_number_of_vpreg_not_enough_virtual_at_decode = %u\n", total_number_of_vpreg_not_enough_virtual_at_decode);
  fprintf(fout, "max_vpreg_virtual_regs_used_in_subcore = %u\n", max_vpreg_virtual_regs_used_in_subcore);
  fprintf(fout, "max_vpreg_physical_regs_used_in_subcore = %u\n", max_vpreg_physical_regs_used_in_subcore);
  fprintf(fout, "max_vpreg_physical_freepool_usage_in_bank = %u\n", max_vpreg_physical_freepool_usage_in_bank);
  fprintf(fout, "max_vpreg_number_of_consumers = %u\n", max_vpreg_number_of_consumers);
  fprintf(fout, "total_vpreg_predication_dependencies = %lld\n", total_vpreg_predication_dependencies);
  fprintf(fout, "total_vpreg_predication_dependencies_per_decoded_instructions = %.4lf\n", double(total_vpreg_predication_dependencies) / total_instructions_inserted_in_ibooo);
  fprintf(fout, "total_vpreg_merges = %lld\n", total_vpreg_merges);
  fprintf(fout, "total_vpreg_extra_rf_reads = %lld\n", total_vpreg_extra_rf_reads);
  fprintf(fout, "total_rf_reads = %lld\n", total_rf_reads);
  fprintf(fout, "total_percentage_vpreg_extra_reads = %.4lf\n", double(total_vpreg_extra_rf_reads) / total_rf_reads * 100);
}
// MOD. End. VPREG

void shader_core_stats::print(FILE *fout) {
  unsigned long long thread_icount_uarch = 0;
  unsigned long long warp_icount_uarch = 0;
  print_remodeling_stats(fout); // MOD. Remodeling
  print_coalescing_stats(fout); // MOD. Remodeling
  // print_custom_shader_stats(fout); // MOD. Custom stats
  // print_ibuffer_ooo_stats(fout); // MOD. IBuffer_ooo custom stats
  // print_vpreg_stats(fout); // MOD. VPREG stats
  for (unsigned i = 0; i < m_config->num_shader(); i++) {
    thread_icount_uarch += m_num_sim_insn[i];
    warp_icount_uarch += m_num_sim_winsn[i];
  }
  fprintf(fout, "gpgpu_n_tot_thrd_icount = %lld\n", thread_icount_uarch);
  fprintf(fout, "gpgpu_n_tot_w_icount = %lld\n", warp_icount_uarch);

  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_stall_dispatch_to_subpipeline_mem"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_mem_read_local"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_mem_write_local"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_mem_read_global"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_mem_write_global"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_mem_texture"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_mem_const"]->print(fout);

  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_load_insn"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_store_insn"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_shmem_insn"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_sstarr_insn"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_tex_insn"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_const_mem_insn"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_param_mem_insn"]->print(fout);

  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_shmem_bkconflict"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_shmem_port_conflict"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_l1cache_bkconflict"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_l1cache_coalescing_conflicts"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_directly_to_l2_coalescing_conflicts"]->print(fout);

  fprintf(fout, "gpgpu_n_intrawarp_mshr_merge = %d\n",
          gpgpu_n_intrawarp_mshr_merge);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_cmem_portconflict"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_n_cmem_coalescing_conflicts"]->print(fout);
          
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[C_MEM][BK_CONF]"]->print(fout);
  m_gpu->m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[S_MEM][BK_CONF]"]->print(fout);

  unsigned long long coalescing_stall_at_data_cache = m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_LD][BK_CONF]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_ST][BK_CONF]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_LD][BK_CONF]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_ST][BK_CONF]"]->get_value();
  
  unsigned long long coalescing_stall_or_bank_conf_at_data_cache = m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_LD][COAL_STALL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_ST][COAL_STALL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_LD][COAL_STALL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_ST][COAL_STALL]"]->get_value();
  
  unsigned long long data_port_stall_at_data_cache = m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_LD][DATA_PORT_STALL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_ST][DATA_PORT_STALL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_LD][DATA_PORT_STALL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_ST][DATA_PORT_STALL]"]->get_value();

  unsigned long long icnt_stal_at_data_cache = m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_LD][ICNT_RC_FAIL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[G_MEM_ST][ICNT_RC_FAIL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_LD][ICNT_RC_FAIL]"]->get_value() +
                                                        m_gpu-> m_gpu_per_sm_stats.m_stats_map["gpgpu_stall_shd_mem[L_MEM_ST][ICNT_RC_FAIL]"]->get_value();

  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][resource_stall] = %llu\n",coalescing_stall_at_data_cache);  // coalescing stall at data cache
  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][coal_stall] = %llu\n", coalescing_stall_or_bank_conf_at_data_cache);  // coalescing stall + bank conflict at data cache
  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][data_port_stall] = %llu\n", data_port_stall_at_data_cache);  // data port stall at data cache
  fprintf(fout, "gpgpu_stall_shd_mem[gl_mem][icnt_stall] = %llu\n", icnt_stal_at_data_cache);

  fprintf(fout, "gpu_reg_bank_conflict_stalls = %d\n",
          gpu_reg_bank_conflict_stalls);

  // MOD. Begin. Custom Stats
  numEffectiveIncompleteWarps = m_gpu-> m_gpu_per_sm_stats.m_stats_map["Total_effective_incomplete_warps"]->get_value();
  fprintf(fout, "Total_effective_incomplete_warps: %d\n", numEffectiveIncompleteWarps);
  double total_percentage_effective_incomplete_warps = (((double)numEffectiveIncompleteWarps)/warp_icount_uarch) * 100;
  fprintf(fout, "Total_percentage_incomplete_warps: %.2f\n", total_percentage_effective_incomplete_warps);
  // MOD. End. Custom Stats

  fprintf(fout, "Warp Occupancy Distribution:\n");
  fprintf(fout, "Stall:%d\t", shader_cycle_distro[2]);
  fprintf(fout, "W0_Idle:%d\t", shader_cycle_distro[0]);
  fprintf(fout, "W0_Scoreboard:%d", shader_cycle_distro[1]);
  for(unsigned int i = 1; i < m_config->warp_size + 1; i++) {
    shader_cycle_distro[2 + i] += m_gpu-> m_gpu_per_sm_stats.m_stats_map["warp_occ_dist" + std::to_string(i)]->get_value();
    fprintf(fout, "\tW%d:%d", i, shader_cycle_distro[2 + i]);
  }
  fprintf(fout, "\n");
  fprintf(fout, "single_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, single_issue_nums[i]);
  fprintf(fout, "\n");
  fprintf(fout, "dual_issue_nums: ");
  for (unsigned i = 0; i < m_config->gpgpu_num_sched_per_core; i++)
    fprintf(fout, "WS%d:%d\t", i, dual_issue_nums[i]);
  fprintf(fout, "\n");

  // Not paralel safe yet. Needs to be fixed
  m_outgoing_traffic_stats->print(fout);
  m_incoming_traffic_stats->print(fout);
}

void shader_core_stats::event_warp_issued(unsigned s_id, unsigned warp_id,
                                          unsigned num_issued,
                                          unsigned dynamic_warp_id) {
  assert(warp_id <= m_config->max_warps_per_shader);
  for (unsigned i = 0; i < num_issued; ++i) {
    if (m_shader_dynamic_warp_issue_distro[s_id].size() <= dynamic_warp_id) {
      m_shader_dynamic_warp_issue_distro[s_id].resize(dynamic_warp_id + 1);
    }
    ++m_shader_dynamic_warp_issue_distro[s_id][dynamic_warp_id];
    if (m_shader_warp_slot_issue_distro[s_id].size() <= warp_id) {
      m_shader_warp_slot_issue_distro[s_id].resize(warp_id + 1);
    }
    ++m_shader_warp_slot_issue_distro[s_id][warp_id];
  }
}

void shader_core_stats::visualizer_print(gzFile visualizer_file) {
  // warp divergence breakdown
  gzprintf(visualizer_file, "WarpDivergenceBreakdown:");
  unsigned int total = 0;
  unsigned int cf =
      (m_config->gpgpu_warpdistro_shader == -1) ? m_config->num_shader() : 1;
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[0] - last_shader_cycle_distro[0]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[1] - last_shader_cycle_distro[1]) / cf);
  gzprintf(visualizer_file, " %d",
           (shader_cycle_distro[2] - last_shader_cycle_distro[2]) / cf);
  for (unsigned i = 0; i < m_config->warp_size + 3; i++) {
    if (i >= 3) {
      total += (shader_cycle_distro[i] - last_shader_cycle_distro[i]);
      if (((i - 3) % (m_config->warp_size / 8)) ==
          ((m_config->warp_size / 8) - 1)) {
        gzprintf(visualizer_file, " %d", total / cf);
        total = 0;
      }
    }
    last_shader_cycle_distro[i] = shader_cycle_distro[i];
  }
  gzprintf(visualizer_file, "\n");
  ctas_completed = m_gpu->m_gpu_per_sm_stats.m_stats_map["ctas_completed"]->get_value();
  gzprintf(visualizer_file, "ctas_completed: %d\n", ctas_completed);
  ctas_completed = 0;
  // warp issue breakdown
  unsigned sid = m_config->gpgpu_warp_issue_shader;
  unsigned count = 0;
  unsigned warp_id_issued_sum = 0;
  gzprintf(visualizer_file, "WarpIssueSlotBreakdown:");
  if (m_shader_warp_slot_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_warp_slot_issue_distro[sid].begin();
         iter != m_shader_warp_slot_issue_distro[sid].end(); iter++, count++) {
      unsigned diff = count < m_last_shader_warp_slot_issue_distro.size()
                          ? *iter - m_last_shader_warp_slot_issue_distro[count]
                          : *iter;
      gzprintf(visualizer_file, " %d", diff);
      warp_id_issued_sum += diff;
    }
    m_last_shader_warp_slot_issue_distro = m_shader_warp_slot_issue_distro[sid];
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

#define DYNAMIC_WARP_PRINT_RESOLUTION 32
  unsigned total_issued_this_resolution = 0;
  unsigned dynamic_id_issued_sum = 0;
  count = 0;
  gzprintf(visualizer_file, "WarpIssueDynamicIdBreakdown:");
  if (m_shader_dynamic_warp_issue_distro[sid].size() > 0) {
    for (std::vector<unsigned>::const_iterator iter =
             m_shader_dynamic_warp_issue_distro[sid].begin();
         iter != m_shader_dynamic_warp_issue_distro[sid].end();
         iter++, count++) {
      unsigned diff =
          count < m_last_shader_dynamic_warp_issue_distro.size()
              ? *iter - m_last_shader_dynamic_warp_issue_distro[count]
              : *iter;
      total_issued_this_resolution += diff;
      if ((count + 1) % DYNAMIC_WARP_PRINT_RESOLUTION == 0) {
        gzprintf(visualizer_file, " %d", total_issued_this_resolution);
        dynamic_id_issued_sum += total_issued_this_resolution;
        total_issued_this_resolution = 0;
      }
    }
    if (count % DYNAMIC_WARP_PRINT_RESOLUTION != 0) {
      gzprintf(visualizer_file, " %d", total_issued_this_resolution);
      dynamic_id_issued_sum += total_issued_this_resolution;
    }
    m_last_shader_dynamic_warp_issue_distro =
        m_shader_dynamic_warp_issue_distro[sid];
    assert(warp_id_issued_sum == dynamic_id_issued_sum);
  } else {
    gzprintf(visualizer_file, " 0");
  }
  gzprintf(visualizer_file, "\n");

  // overall cache miss rates
  gzprintf(visualizer_file, "gpgpu_n_l1cache_bkconflict: %d\n",
           gpgpu_n_l1cache_bkconflict);
  gzprintf(visualizer_file, "gpgpu_n_shmem_bkconflict: %d\n",
           gpgpu_n_shmem_bkconflict);

  // instruction count per shader core
  gzprintf(visualizer_file, "shaderinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_insn[i]);
  gzprintf(visualizer_file, "\n");
  // warp instruction count per shader core
  gzprintf(visualizer_file, "shaderwarpinsncount:  ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_num_sim_winsn[i]);
  gzprintf(visualizer_file, "\n");
  // warp divergence per shader core
  gzprintf(visualizer_file, "shaderwarpdiv: ");
  for (unsigned i = 0; i < m_config->num_shader(); i++)
    gzprintf(visualizer_file, "%u ", m_n_diverge[i]);
  gzprintf(visualizer_file, "\n");
}

address_type coalesced_segment(address_type addr,
                               unsigned segment_size_lg2bytes) {
  return (addr >> segment_size_lg2bytes);
}

// Returns numbers of addresses in translated_addrs, each addr points to a 4B
// (32-bit) word

/////////////////////////////////////////////////////////////////////////////////////////

// MOD. End. VPREG

/*
    virtual void issue( register_set& source_reg )
    {
        //move_warp(m_dispatch_reg,source_reg);
        //source_reg.move_out_to(m_dispatch_reg);
    }
*/

/*
*/

void gpgpu_sim::shader_print_runtime_stat(FILE *fout) {
  /*
 fprintf(fout, "SHD_INSN: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_num_sim_insn());
 fprintf(fout, "\n");
 fprintf(fout, "SHD_THDS: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_not_completed());
 fprintf(fout, "\n");
 fprintf(fout, "SHD_DIVG: ");
 for (unsigned i=0;i<m_n_shader;i++)
    fprintf(fout, "%u ",m_sc[i]->get_n_diverge());
 fprintf(fout, "\n");

 fprintf(fout, "THD_INSN: ");
 for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
    fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn(i) );
 fprintf(fout, "\n");
 */
}

void gpgpu_sim::shader_print_scheduler_stat(FILE *fout,
                                            bool print_dynamic_info) {
  m_shader_stats->ctas_completed = m_gpu_per_sm_stats.m_stats_map["ctas_completed"]->get_value();
  fprintf(fout, "ctas_completed %d, ", m_shader_stats->ctas_completed);

  // Print out the stats from the sampling shader core
  const unsigned scheduler_sampling_core =
      m_shader_config->gpgpu_warp_issue_shader;
#define STR_SIZE 55
  char name_buff[STR_SIZE];
  name_buff[STR_SIZE - 1] = '\0';
  const std::vector<unsigned> &distro =
      print_dynamic_info
          ? m_shader_stats->get_dynamic_warp_issue()[scheduler_sampling_core]
          : m_shader_stats->get_warp_slot_issue()[scheduler_sampling_core];
  if (print_dynamic_info) {
    snprintf(name_buff, STR_SIZE - 1, "dynamic_warp_id");
  } else {
    snprintf(name_buff, STR_SIZE - 1, "warp_id");
  }
  fprintf(fout, "Shader %d %s issue ditsribution:\n", scheduler_sampling_core,
          name_buff);
  const unsigned num_warp_ids = distro.size();
  // First print out the warp ids
  fprintf(fout, "%s:\n", name_buff);
  for (unsigned warp_id = 0; warp_id < num_warp_ids; ++warp_id) {
    fprintf(fout, "%d, ", warp_id);
  }

  fprintf(fout, "\ndistro:\n");
  // Then print out the distribution of instuctions issued
  for (std::vector<unsigned>::const_iterator iter = distro.begin();
       iter != distro.end(); iter++) {
    fprintf(fout, "%d, ", *iter);
  }
  fprintf(fout, "\n");
}

void gpgpu_sim::shader_print_cache_stats(FILE *fout) const {
  // L1I
  struct cache_sub_stats total_css;
  struct cache_sub_stats css;

  fprintf(fout, "\n========= Core cache stats =========\n");

  // MOD. Begin. L0I
  if (m_shader_config->is_L0I_enabled) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L0I_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L0I_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL0I_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL0I_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL0I_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL0I_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL0I_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }
  // MOD. End. L0I

  if (!m_shader_config->m_L1I_L1_half_C_cache_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1I_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1I_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1I_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1I_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1I_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1I_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1I_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }

  // L1D
  if (!m_shader_config->m_L1D_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1D_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->get_L1D_sub_stats(css);

      fprintf(stdout,
              "\tL1D_cache_core[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, css.accesses, css.misses,
              (double)css.misses / (double)css.accesses, css.pending_hits,
              css.res_fails);

      total_css += css;
    }
    fprintf(fout, "\tL1D_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1D_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1D_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1D_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1D_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
    total_css.print_port_stats(fout, "\tL1D_cache");
  }

  // L1C
  if (!m_shader_config->m_L1C_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1C_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1C_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1C_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1C_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1C_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1C_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1C_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }

  // L1T
  if (!m_shader_config->m_L1T_config.disabled()) {
    total_css.clear();
    css.clear();
    fprintf(fout, "L1T_cache:\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
      m_cluster[i]->get_L1T_sub_stats(css);
      total_css += css;
    }
    fprintf(fout, "\tL1T_total_cache_accesses = %llu\n", total_css.accesses);
    fprintf(fout, "\tL1T_total_cache_misses = %llu\n", total_css.misses);
    if (total_css.accesses > 0) {
      fprintf(fout, "\tL1T_total_cache_miss_rate = %.4lf\n",
              (double)total_css.misses / (double)total_css.accesses);
    }
    fprintf(fout, "\tL1T_total_cache_pending_hits = %llu\n",
            total_css.pending_hits);
    fprintf(fout, "\tL1T_total_cache_reservation_fails = %llu\n",
            total_css.res_fails);
  }
}

void gpgpu_sim::shader_print_l1_miss_stat(FILE *fout) const {
  unsigned total_d1_misses = 0, total_d1_accesses = 0;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; ++i) {
    unsigned custer_d1_misses = 0, cluster_d1_accesses = 0;
    m_cluster[i]->print_cache_stats(fout, cluster_d1_accesses,
                                    custer_d1_misses);
    total_d1_misses += custer_d1_misses;
    total_d1_accesses += cluster_d1_accesses;
  }
  fprintf(fout, "total_dl1_misses=%d\n", total_d1_misses);
  fprintf(fout, "total_dl1_accesses=%d\n", total_d1_accesses);
  fprintf(fout, "total_dl1_miss_rate= %f\n",
          (float)total_d1_misses / (float)total_d1_accesses);
  /*
  fprintf(fout, "THD_INSN_AC: ");
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_insn_ac(i));
  fprintf(fout, "\n");
  fprintf(fout, "T_L1_Mss: "); //l1 miss rate per thread
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i));
  fprintf(fout, "\n");
  fprintf(fout, "T_L1_Mgs: "); //l1 merged miss rate per thread
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++)
     fprintf(fout, "%d ", m_sc[0]->get_thread_n_l1_mis_ac(i) -
  m_sc[0]->get_thread_n_l1_mrghit_ac(i)); fprintf(fout, "\n"); fprintf(fout,
  "T_L1_Acc: "); //l1 access per thread for (unsigned i=0;
  i<m_shader_config->n_thread_per_shader; i++) fprintf(fout, "%d ",
  m_sc[0]->get_thread_n_l1_access_ac(i)); fprintf(fout, "\n");

  //per warp
  int temp =0;
  fprintf(fout, "W_L1_Mss: "); //l1 miss rate per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += m_sc[0]->get_thread_n_l1_mis_ac(i);
     if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  temp=0;
  fprintf(fout, "W_L1_Mgs: "); //l1 merged miss rate per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += (m_sc[0]->get_thread_n_l1_mis_ac(i) -
  m_sc[0]->get_thread_n_l1_mrghit_ac(i) ); if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  temp =0;
  fprintf(fout, "W_L1_Acc: "); //l1 access per warp
  for (unsigned i=0; i<m_shader_config->n_thread_per_shader; i++) {
     temp += m_sc[0]->get_thread_n_l1_access_ac(i);
     if (i%m_shader_config->warp_size ==
  (unsigned)(m_shader_config->warp_size-1)) { fprintf(fout, "%d ", temp); temp =
  0;
     }
  }
  fprintf(fout, "\n");
  */
}

void warp_inst_t::print(FILE *fout) const {
  if (empty()) {
    fprintf(fout, "bubble\n");
    return;
  } else
    fprintf(fout, "0x%04llx ", pc);
  fprintf(fout, "w%02d[", m_warp_id);
  for (unsigned j = 0; j < m_config->warp_size; j++)
    fprintf(fout, "%c", (active(j) ? '1' : '0'));
  fprintf(fout, "]: ");
  m_config->gpgpu_ctx->func_sim->ptx_print_insn(pc, fout);
  fprintf(fout, "\n");
}

unsigned int shader_core_config::max_cta(const kernel_info_t &k) const {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  // Limit by n_threads/shader
  unsigned int result_thread = n_thread_per_shader / padded_cta_size;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  // Limit by shmem/shader
  unsigned int result_shmem = (unsigned)-1;
  if (kernel_info->smem > 0)
    result_shmem = gpgpu_shmem_size / kernel_info->smem;

  // Limit by register count, rounded up to multiple of 4.
  unsigned int result_regs = (unsigned)-1;

  unsigned int num_configured_regs = gpgpu_shader_registers;
  if (kernel_info->regs > 0)
    result_regs = num_configured_regs /
                  (padded_cta_size * ((kernel_info->regs + 3) & ~3));

  // Limit by CTA
  unsigned int result_cta = max_cta_per_core;

  unsigned result = result_thread;
  result = gs_min2(result, result_shmem);
  if(!is_skip_rf_limit_enabled) result = gs_min2(result, result_regs); // MOD. Begin. Skip RF limitation.
  result = gs_min2(result, result_cta);

  static const struct gpgpu_ptx_sim_info *last_kinfo = NULL;
  if (last_kinfo !=
      kernel_info) {  // Only print out stats if kernel_info struct changes
    last_kinfo = kernel_info;
    printf("GPGPU-Sim uArch: CTA/core = %u, limited by:", result);
    if (result == result_thread) printf(" threads");
    if (result == result_shmem) printf(" shmem");
    if (result == result_regs) printf(" regs");
    if (result == result_cta) printf(" cta_limit");
    printf("\n");
  }

  // gpu_max_cta_per_shader is limited by number of CTAs if not enough to keep
  // all cores busy
  if (k.num_blocks() < result * num_shader()) {
    result = k.num_blocks() / num_shader();
    if (k.num_blocks() % num_shader()) result++;
  }

  assert(result <= MAX_CTA_PER_SHADER);
  if (result < 1) {
    printf(
        "GPGPU-Sim uArch: ERROR ** Kernel requires more resources than shader "
        "has.\n");
    if (gpgpu_ignore_resources_limitation) {
      printf(
          "GPGPU-Sim uArch: gpgpu_ignore_resources_limitation is set, ignore "
          "the ERROR!\n");
      return 1;
    }
    abort();
  }

  if (adaptive_cache_config && !k.cache_config_set) {
    // For more info about adaptive cache, see
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
    unsigned total_shmem = kernel_info->smem * result;
    assert(total_shmem >= 0 && total_shmem <= shmem_opt_list.back());

    // Unified cache config is in KB. Converting to B
    unsigned total_unified = m_L1D_config.m_unified_cache_size * 1024;

    bool l1d_configured = false;
    unsigned max_assoc = m_L1D_config.get_max_assoc();

    for (std::vector<unsigned>::const_iterator it = shmem_opt_list.begin();
         it < shmem_opt_list.end(); it++) {
      if (total_shmem <= *it) {
        float l1_ratio = 1 - ((float)*(it) / total_unified);
        // make sure the ratio is between 0 and 1
        assert(0 <= l1_ratio && l1_ratio <= 1);
        // round to nearest instead of round down
        m_L1D_config.set_assoc(max_assoc * l1_ratio + 0.5f);
        l1d_configured = true;
        break;
      }
    }

    assert(l1d_configured && "no shared memory option found");

    if (m_L1D_config.is_streaming()) {
      // for streaming cache, if the whole memory is allocated
      // to the L1 cache, then make the allocation to be on_MISS
      // otherwise, make it ON_FILL to eliminate line allocation fails
      // i.e. MSHR throughput is the same, independent on the L1 cache
      // size/associativity
      if (total_shmem == 0) {
        m_L1D_config.set_allocation_policy(ON_MISS);
        printf("GPGPU-Sim: Reconfigure L1 allocation to ON_MISS\n");
      } else {
        m_L1D_config.set_allocation_policy(ON_FILL);
        printf("GPGPU-Sim: Reconfigure L1 allocation to ON_FILL\n");
      }
    }
    printf("GPGPU-Sim: Reconfigure L1 cache to %uKB\n",
           m_L1D_config.get_total_size_inKB());

    k.cache_config_set = true;
  }

  return result;
}

void shader_core_config::set_pipeline_latency() {
  // calculate the max latency  based on the input

  unsigned int int_latency[6];
  unsigned int fp_latency[5];
  unsigned int dp_latency[5];
  unsigned int sfu_latency;
  unsigned int tensor_latency;

  int_latency[0] = fp_latency[0] = dp_latency[0] = 0;
  int_latency[1] = fp_latency[1] = dp_latency[1] = 0;
  int_latency[2] = fp_latency[2] = dp_latency[2] = 0;
  int_latency[3] = fp_latency[3] = dp_latency[3] = 0;
  int_latency[4] = fp_latency[4] = dp_latency[4] = 0;
  int_latency[5] = 0;
  /*
   * [0] ADD,SUB
   * [1] MAX,Min
   * [2] MUL
   * [3] MAD
   * [4] DIV
   * [5] SHFL
   */
  sscanf(gpgpu_ctx->func_sim->opcode_latency_int, "%u,%u,%u,%u,%u,%u",
         &int_latency[0], &int_latency[1], &int_latency[2], &int_latency[3],
         &int_latency[4], &int_latency[5]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_fp, "%u,%u,%u,%u,%u",
         &fp_latency[0], &fp_latency[1], &fp_latency[2], &fp_latency[3],
         &fp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_dp, "%u,%u,%u,%u,%u",
         &dp_latency[0], &dp_latency[1], &dp_latency[2], &dp_latency[3],
         &dp_latency[4]);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_sfu, "%u", &sfu_latency);
  sscanf(gpgpu_ctx->func_sim->opcode_latency_tensor, "%u", &tensor_latency);

  // all div operation are executed on sfu
  // assume that the max latency are dp div or normal sfu_latency
  max_sfu_latency = std::max(dp_latency[4], sfu_latency);
  // assume that the max operation has the max latency
  max_sp_latency = fp_latency[1];
  max_int_latency = std::max(int_latency[1], int_latency[5]);
  max_int_latency = std::max(max_int_latency, predicate_latency);
  max_dp_latency = dp_latency[1];
  max_tensor_core_latency = tensor_latency;
}

// Flushes all content of the cache to memory

// modifiers

barrier_set_t::barrier_set_t(shader_core_ctx_wrapper *shader,
                             unsigned max_warps_per_core,
                             unsigned max_cta_per_core,
                             unsigned max_barriers_per_cta,
                             unsigned warp_size) {
  m_max_warps_per_core = max_warps_per_core;
  m_max_cta_per_core = max_cta_per_core;
  m_max_barriers_per_cta = max_barriers_per_cta;
  m_warp_size = warp_size;
  m_shader = shader;
  if (max_warps_per_core > WARP_PER_CTA_MAX) {
    printf(
        "ERROR ** increase WARP_PER_CTA_MAX in shader.h from %u to >= %u or "
        "warps per cta in gpgpusim.config\n",
        WARP_PER_CTA_MAX, max_warps_per_core);
    exit(1);
  }
  if (max_barriers_per_cta > MAX_BARRIERS_PER_CTA) {
    printf(
        "ERROR ** increase MAX_BARRIERS_PER_CTA in abstract_hardware_model.h "
        "from %u to >= %u or barriers per cta in gpgpusim.config\n",
        MAX_BARRIERS_PER_CTA, max_barriers_per_cta);
    exit(1);
  }
  m_warp_active.reset();
  m_warp_at_barrier.reset();
  for (unsigned i = 0; i < max_barriers_per_cta; i++) {
    m_bar_id_to_warps[i].reset();
  }
}

// during cta allocation
void barrier_set_t::allocate_barrier(unsigned cta_id, warp_set_t warps) {
  assert(cta_id < m_max_cta_per_core);
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  assert(w == m_cta_to_warps.end());  // cta should not already be active or
                                      // allocated barrier resources
  m_cta_to_warps[cta_id] = warps;
  assert(m_cta_to_warps.size() <=
         m_max_cta_per_core);  // catch cta's that were not properly deallocated

  m_warp_active |= warps;
  m_warp_at_barrier &= ~warps;
  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    m_bar_id_to_warps[i] &= ~warps;
  }
}

// during cta deallocation
void barrier_set_t::deallocate_barrier(unsigned cta_id) {
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);
  if (w == m_cta_to_warps.end()) return;
  warp_set_t warps = w->second;
  warp_set_t at_barrier = warps & m_warp_at_barrier;
  assert(at_barrier.any() == false);  // no warps stuck at barrier
  warp_set_t active = warps & m_warp_active;
  assert(active.any() == false);  // no warps in CTA still running
  m_warp_active &= ~warps;
  m_warp_at_barrier &= ~warps;

  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t at_a_specific_barrier = warps & m_bar_id_to_warps[i];
    assert(at_a_specific_barrier.any() == false);  // no warps stuck at barrier
    m_bar_id_to_warps[i] &= ~warps;
  }
  m_cta_to_warps.erase(w);
}

// individual warp hits barrier
void barrier_set_t::warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                                         warp_inst_t *inst) {
  barrier_type bar_type = inst->bar_type;
  unsigned bar_id = inst->bar_id;
  unsigned bar_count = inst->bar_count;
  assert(bar_id != (unsigned)-1);
  cta_to_warp_t::iterator w = m_cta_to_warps.find(cta_id);

  if (w == m_cta_to_warps.end()) {  // cta is active
    printf(
        "ERROR ** cta_id %u not found in barrier set on cycle %llu+%llu...\n",
        cta_id, m_shader->get_gpu()->gpu_tot_sim_cycle,
        m_shader->get_gpu()->gpu_sim_cycle);
    dump();
    abort();
  }
  assert(w->second.test(warp_id) == true);  // warp is in cta

  m_bar_id_to_warps[bar_id].set(warp_id);
  if (bar_type == SYNC || bar_type == RED) {
    m_warp_at_barrier.set(warp_id);
  }
  warp_set_t warps_in_cta = w->second;
  warp_set_t at_barrier = warps_in_cta & m_bar_id_to_warps[bar_id];
  warp_set_t active = warps_in_cta & m_warp_active;
  if (bar_count == (unsigned)-1) {
    if (at_barrier == active) {
      // all warps have reached barrier, so release waiting warps...
      m_bar_id_to_warps[bar_id] &= ~at_barrier;
      m_warp_at_barrier &= ~at_barrier;
      if (bar_type == RED) {
        m_shader->broadcast_barrier_reduction(cta_id, bar_id, at_barrier);
      }else if(inst->op == MEMORY_BARRIER_OP) {
        m_shader->num_cycles_to_stall_SM(inst->m_num_cycles_to_stall_SM);
      }
    }
  } else {
    // TODO: check on the hardware if the count should include warp that exited
    if ((at_barrier.count() * m_warp_size) == bar_count) {
      // required number of warps have reached barrier, so release waiting
      // warps...
      m_bar_id_to_warps[bar_id] &= ~at_barrier;
      m_warp_at_barrier &= ~at_barrier;
      if (bar_type == RED) {
        m_shader->broadcast_barrier_reduction(cta_id, bar_id, at_barrier);
      }
    }
  }
}

// warp reaches exit
void barrier_set_t::warp_exit(unsigned warp_id) {
  // caller needs to verify all threads in warp are done, e.g., by checking PDOM
  // stack to see it has only one entry during exit_impl()
  m_warp_active.reset(warp_id);

  // test for barrier release
  cta_to_warp_t::iterator w = m_cta_to_warps.begin();
  for (; w != m_cta_to_warps.end(); ++w) {
    if (w->second.test(warp_id) == true) break;
  }
  warp_set_t warps_in_cta = w->second;
  warp_set_t active = warps_in_cta & m_warp_active;

  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t at_a_specific_barrier = warps_in_cta & m_bar_id_to_warps[i];
    if (at_a_specific_barrier == active) {
      // all warps have reached barrier, so release waiting warps...
      m_bar_id_to_warps[i] &= ~at_a_specific_barrier;
      m_warp_at_barrier &= ~at_a_specific_barrier;
    }
  }
}

// assertions
bool barrier_set_t::warp_waiting_at_barrier(unsigned warp_id) const {
  return m_warp_at_barrier.test(warp_id);
}

void barrier_set_t::dump() {
  printf("barrier set information\n");
  printf("  m_max_cta_per_core = %u\n", m_max_cta_per_core);
  printf("  m_max_warps_per_core = %u\n", m_max_warps_per_core);
  printf(" m_max_barriers_per_cta =%u\n", m_max_barriers_per_cta);
  printf("  cta_to_warps:\n");

  cta_to_warp_t::const_iterator i;
  for (i = m_cta_to_warps.begin(); i != m_cta_to_warps.end(); i++) {
    unsigned cta_id = i->first;
    warp_set_t warps = i->second;
    printf("    cta_id %u : %s\n", cta_id, warps.to_string().c_str());
  }
  printf("  warp_active: %s\n", m_warp_active.to_string().c_str());
  printf("  warp_at_barrier: %s\n", m_warp_at_barrier.to_string().c_str());
  for (unsigned i = 0; i < m_max_barriers_per_cta; i++) {
    warp_set_t warps_reached_barrier = m_bar_id_to_warps[i];
    printf("  warp_at_barrier %u: %s\n", i,
           warps_reached_barrier.to_string().c_str());
  }
  fflush(stdout);
}

// MOD. Begin. L0I
// MOD. End. L0I

kernel_info_t* shd_warp_t::get_kernel_info() const { return m_shader->get_kernel_info(); }

bool shd_warp_t::functional_done() const {
  return get_n_completed() == m_warp_size;
}

bool shd_warp_t::hardware_done() const {
  return functional_done() && stores_done() && !inst_in_pipeline();
}

bool shd_warp_t::waiting() {
  if (functional_done()) {
    // waiting to be initialized with a kernel
    return true;
  } else if (m_shader->warp_waiting_at_barrier(m_warp_id)) {
    // waiting for other warps in CTA to reach barrier
    return true;
  } else if (m_shader->warp_waiting_at_mem_barrier(m_warp_id)) {
    // waiting for memory barrier
    return true;
  }else if(m_shader->warp_waiting_grid_barrier(m_warp_id)) {
    return true;
  }
  // else if (m_n_atomic > 0) {
  //   // waiting for atomic operation to complete at memory:
  //   // this stall is not required for accurate timing model, but rather we
  //   // stall here since if a call/return instruction occurs in the meantime
  //   // the functional execution of the atomic when it hits DRAM can cause
  //   // the wrong register to be read.
  //   return true;
  // }
  return false;
}

void shd_warp_t::print(FILE *fout) const {
  if (!done_exit()) {
    fprintf(fout, "w%02u npc: 0x%04llx, done:%c%c%c%c:%2u i:%u s:%u a:%u (done: ",
            m_warp_id, m_next_pc, (functional_done() ? 'f' : ' '),
            (stores_done() ? 's' : ' '), (inst_in_pipeline() ? ' ' : 'i'),
            (done_exit() ? 'e' : ' '), n_completed, m_inst_in_pipeline,
            m_stores_outstanding, m_n_atomic);
    for (unsigned i = m_warp_id * m_warp_size;
         i < (m_warp_id + 1) * m_warp_size; i++) {
      if (m_shader->ptx_thread_done(i))
        fprintf(fout, "1");
      else
        fprintf(fout, "0");
      if ((((i + 1) % 4) == 0) && (i + 1) < (m_warp_id + 1) * m_warp_size)
        fprintf(fout, ",");
    }
    fprintf(fout, ") ");
    fprintf(fout, " active=%s", m_active_threads.to_string().c_str());
    fprintf(fout, " last fetched @ %5llu", m_last_fetch);
    if (m_imiss_pending) fprintf(fout, " i-miss pending");
    fprintf(fout, "\n");
  }
}

void shd_warp_t::print_ibuffer(FILE *fout) const {
  fprintf(fout, "  ibuffer[%2u] : ", m_warp_id);
  for (unsigned i = 0; i < IBUFFER_SIZE; i++) {
    const inst_t *inst = m_ibuffer[i].m_inst;
    if (inst)
      inst->print_insn(fout);
    else if (m_ibuffer[i].m_valid)
      fprintf(fout, " <invalid instruction> ");
    else
      fprintf(fout, " <empty> ");
  }
  fprintf(fout, "\n");
}

int register_bank(int regnum, int wid, unsigned num_banks,
                  unsigned bank_warp_shift, bool sub_core_model,
                  int banks_per_sched, unsigned sched_id) { 
  // MOD. Begin. Predication
  if(regnum >= FIRST_PRED_REG) {
    return BANK_ID_PREDICATE_REGS_TO_DETECT_SKIP;
  }
  // MOD. End. Predication
  int bank = regnum;
  if (bank_warp_shift) bank += wid;
  if (sub_core_model) {
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks;
}

// MOD. Begin. OPC custom stats

// MOD. End. OPC custom stats

simt_core_cluster::simt_core_cluster(class gpgpu_sim *gpu, unsigned cluster_id,
                                     const shader_core_config *config,
                                     const memory_config *mem_config,
                                     shader_core_stats *stats,
                                     class memory_stats_t *mstats) : m_cluster_stats("Cluster_" + std::to_string(cluster_id)), m_outgoing_traffic_stats(""), m_incoming_traffic_stats("") {
  m_config = config;
  m_cta_issue_next_core = m_config->n_simt_cores_per_cluster -
                          1;  // this causes first launch to use hw cta 0
  m_cluster_id = cluster_id;
  m_gpu = gpu;
  m_stats = stats;
  m_memory_stats = mstats;
  m_mem_config = mem_config;
}

void simt_core_cluster::core_cycle() {
  for (std::list<unsigned>::iterator it = m_core_sim_order.begin();
       it != m_core_sim_order.end(); ++it) {
    m_core[*it]->cycle();
  }

  if (m_config->simt_core_sim_order == 1) {
    m_core_sim_order.splice(m_core_sim_order.end(), m_core_sim_order,
                            m_core_sim_order.begin());
  }
}

void simt_core_cluster::reinit() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->reinit(0, m_config->n_thread_per_shader, true);
}

unsigned simt_core_cluster::max_cta(const kernel_info_t &kernel) {
  return m_config->n_simt_cores_per_cluster * m_config->max_cta(kernel);
}

unsigned simt_core_cluster::get_not_completed() const {
  unsigned not_completed = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    not_completed += m_core[i]->get_not_completed();
  return not_completed;
}

void simt_core_cluster::print_not_completed(FILE *fp) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned not_completed = m_core[i]->get_not_completed();
    unsigned sid = m_config->cid_to_sid(i, m_cluster_id);
    fprintf(fp, "%u(%u) ", sid, not_completed);
  }
}

float simt_core_cluster::get_current_occupancy(
    unsigned long long &active, unsigned long long &total) const {
  float aggregate = 0.f;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    aggregate += m_core[i]->get_current_occupancy(active, total);
  }
  return aggregate / m_config->n_simt_cores_per_cluster;
}

unsigned simt_core_cluster::get_n_active_cta() const {
  unsigned n = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    n += m_core[i]->get_n_active_cta();
  return n;
}

unsigned simt_core_cluster::get_n_active_sms() const {
  unsigned n = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    n += m_core[i]->isactive();
  return n;
}

unsigned simt_core_cluster::issue_block2core() {
  unsigned num_blocks_issued = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++) {
    unsigned core =
        (i + m_cta_issue_next_core + 1) % m_config->n_simt_cores_per_cluster;

    kernel_info_t *kernel;
    // Jin: fetch kernel according to concurrent kernel setting
    if (m_config->gpgpu_concurrent_kernel_sm) {  // concurrent kernel on sm
      // always select latest issued kernel
      kernel_info_t *k = m_gpu->select_kernel();
      kernel = k;
    } else {
      // first select core kernel, if no more cta, get a new kernel
      // only when core completes
      kernel = m_core[core]->get_kernel();
      if (!m_gpu->kernel_more_cta_left(kernel)) {
        // wait till current kernel finishes
        if (m_core[core]->get_not_completed() == 0) {
          kernel_info_t *k = m_gpu->select_kernel();
          if (k) m_core[core]->set_kernel(k);
          kernel = k;
        }
      }
    }

    if (m_gpu->kernel_more_cta_left(kernel) &&
        //            (m_core[core]->get_n_active_cta() <
        //            m_config->max_cta(*kernel)) ) {
        m_core[core]->can_issue_1block(*kernel)) {
      m_core[core]->issue_block2core(*kernel);
      m_gpu->increase_num_threads_kernel(kernel->get_uid(), kernel->threads_per_cta());
      num_blocks_issued++;
      m_cta_issue_next_core = core;
      check_kernel_launch_limitation(*kernel, m_config, m_gpu->get_shader_stats()); 
      break;
    }
  }
  
  return num_blocks_issued;
}

void simt_core_cluster::cache_flush() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_flush();
}

void simt_core_cluster::cache_invalidate() {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; i++)
    m_core[i]->cache_invalidate();
}

bool simt_core_cluster::icnt_injection_buffer_full(unsigned size, bool write) {
  unsigned request_size = size;
  if (!write) request_size = READ_PACKET_SIZE;
  return !::icnt_has_buffer(m_cluster_id, request_size, 0);
}

void simt_core_cluster::icnt_inject_request_packet(class mem_fetch *mf) {
  // If the cluster starts allocating more than one core per cluster, this calls must change
  // stats
  if (mf->get_is_write())
    m_stats->made_write_mfs++;
  else
    m_stats->made_read_mfs++;
  switch (mf->get_access_type()) {
    case CONST_ACC_R:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_const", 1);
      break;
    case TEXTURE_ACC_R:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_texture", 1);
      break;
    case GLOBAL_ACC_R:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_read_global", 1);
      break;
    // case GLOBAL_ACC_R: m_stats->gpgpu_n_mem_read_global++;
    // printf("read_global%d\n",m_stats->gpgpu_n_mem_read_global); break;
    case GLOBAL_ACC_W:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_write_global", 1);
      break;
    case LOCAL_ACC_R:
       m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_read_local", 1);
      break;
    case LOCAL_ACC_W:
      //  m_core[0]->m_stats_map["gpgpu_n_mem_write_local"]->increment(1);
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_write_local", 1);
      break;
    case INST_ACC_R:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_read_inst", 1);
      break;
    case L1_WRBK_ACC:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_write_global", 1);
      break;
    case L2_WRBK_ACC:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_l2_writeback", 1);
      break;
    case L1_WR_ALLOC_R:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_l1_write_allocate", 1);
      break;
    case L2_WR_ALLOC_R:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_l2_write_allocate", 1);
      break;
    case GRID_BARRIER_ACC:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_grid_barrier", 1);
      break;
    case TLB_MISS_ACC_DATA:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_tlb_miss_data", 1);
      break;
    case TLB_MISS_ACC_INST:
      m_core[0]->increment_sm_stat_by_integer("gpgpu_n_mem_tlb_miss_inst", 1);
      break;
    default:
      assert(0);
  }

  // The packet size varies depending on the type of request:
  // - For write request and atomic request, the packet contains the data
  // - For read request (i.e. not write nor atomic), the packet only has control
  // metadata
  unsigned int packet_size = mf->size();
  if (!mf->get_is_write() && !mf->isatomic()) {
    packet_size = mf->get_ctrl_size();
  }

  m_outgoing_traffic_stats.record_traffic(mf, packet_size);

  unsigned destination = mf->get_sub_partition_id();
  mf->set_status(IN_ICNT_TO_MEM,
                 m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
  if (!mf->get_is_write() && !mf->isatomic())
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->get_ctrl_size(), 0);
  else
    ::icnt_push(m_cluster_id, m_config->mem2device(destination), (void *)mf,
                mf->size(), 0);
}

void simt_core_cluster::icnt_cycle() {
  if (!m_response_fifo.empty()) {
    mem_fetch *mf = m_response_fifo.front();
    unsigned cid = m_config->sid_to_cid(mf->get_sid());
    if ((mf->get_access_type() == INST_ACC_R) || (mf->get_access_type() == CONST_ACC_R) || (mf->get_access_type() == TLB_MISS_ACC_INST)) {
      // instruction fetch response
      if (!m_core[cid]->fetch_unit_response_buffer_full()) {
        m_response_fifo.pop_front();
        m_core[cid]->accept_fetch_response(mf);
      }
    } else {
      // data response
      if (!m_core[cid]->ldst_unit_response_buffer_full()) {
        m_response_fifo.pop_front();
        m_memory_stats->memlatstat_read_done(mf);
        m_core[cid]->accept_ldst_unit_response(mf);
      }
    }
  }
  if (m_response_fifo.size() < m_config->n_simt_ejection_buffer_size) {
    mem_fetch *mf = (mem_fetch *)::icnt_pop(m_cluster_id, 0);
    if (!mf) return;
    assert(mf->get_tpc() == m_cluster_id);
    assert((mf->get_type() == READ_REPLY) || (mf->get_type() == WRITE_ACK));

    // The packet size varies depending on the type of request:
    // - For read request and atomic request, the packet contains the data
    // - For write-ack, the packet only has control metadata
    unsigned int packet_size =
        (mf->get_is_write()) ? mf->get_ctrl_size() : mf->size();
    m_incoming_traffic_stats.record_traffic(mf, packet_size);
    mf->set_status(IN_CLUSTER_TO_SHADER_QUEUE,
                   m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle);
    // m_memory_stats->memlatstat_read_done(mf,m_shader_config->max_warps_per_shader);
    m_response_fifo.push_back(mf);
    m_stats->n_mem_to_simt[m_cluster_id] += mf->get_num_flits(false);
  }
}

void simt_core_cluster::get_pdom_stack_top_info(unsigned sid, unsigned tid,
                                                unsigned *pc,
                                                unsigned *rpc) const {
  unsigned cid = m_config->sid_to_cid(sid);
  m_core[cid]->get_pdom_stack_top_info(tid, pc, rpc);
}

void simt_core_cluster::display_pipeline(unsigned sid, FILE *fout,
                                         int print_mem, int mask) {
  m_core[m_config->sid_to_cid(sid)]->display_pipeline(fout, print_mem, mask);

  fprintf(fout, "\n");
  fprintf(fout, "Cluster %u pipeline state\n", m_cluster_id);
  fprintf(fout, "Response FIFO (occupancy = %zu):\n", m_response_fifo.size());
  for (std::list<mem_fetch *>::const_iterator i = m_response_fifo.begin();
       i != m_response_fifo.end(); i++) {
    const mem_fetch *mf = *i;
    mf->print(fout);
  }
}

void simt_core_cluster::print_cache_stats(FILE *fp, unsigned &dl1_accesses,
                                          unsigned &dl1_misses) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->print_cache_stats(fp, dl1_accesses, dl1_misses);
  }
}

void simt_core_cluster::get_icnt_stats(long &n_simt_to_mem,
                                       long &n_mem_to_simt) const {
  long simt_to_mem = 0;
  long mem_to_simt = 0;
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_icnt_power_stats(simt_to_mem, mem_to_simt);
  }
  n_simt_to_mem = simt_to_mem;
  n_mem_to_simt = mem_to_simt;
}

void simt_core_cluster::get_cache_stats(cache_stats &cs) const {
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_cache_stats(cs);
  }
}

void simt_core_cluster::get_L1I_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1I_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

// MOD. Begin. L0I
void simt_core_cluster::get_L0I_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L0I_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}
// MOD. End. L0I

void simt_core_cluster::get_L1D_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1D_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}
void simt_core_cluster::get_L1C_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1C_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}
void simt_core_cluster::get_L1T_sub_stats(struct cache_sub_stats &css) const {
  struct cache_sub_stats temp_css;
  struct cache_sub_stats total_css;
  temp_css.clear();
  total_css.clear();
  for (unsigned i = 0; i < m_config->n_simt_cores_per_cluster; ++i) {
    m_core[i]->get_L1T_sub_stats(temp_css);
    total_css += temp_css;
  }
  css = total_css;
}

