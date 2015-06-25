#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "hash.h"
#include "rainbow_table.h"
#include "utils.h"
#include "opencl.h"

namespace ocl_code {
#include "md5.cl.h"
#include "kernels.cl.h"
  std::string md5_cl_str(md5_cl, md5_cl + md5_cl_len);
  std::string kernels_cl_str(kernels_cl, kernels_cl + kernels_cl_len);
}

struct GPUImplementation {
  RainbowTableParams p;
  OpenCLApp cl;
  bool verify;
  CPUImplementation& cpu;
  utils::Stats& stats;
  uint64_t block_size, local_size, global_size;

  cl::Kernel kernel_generate_chains;

  GPUImplementation(
      const RainbowTableParams& p,
      OpenCLApp& cl,
      CPUImplementation& cpu,
      utils::Stats& stats,
      bool verify,
      uint64_t local_size,
      uint64_t global_size,
      uint64_t block_size
      )
    : p(p), cl(cl), cpu(cpu), stats(stats), verify(verify)
    , block_size(block_size), local_size(local_size), global_size(global_size)
  {
    auto prog = cl.build_program(std::vector<std::string> {
      ocl_code::md5_cl_str,
      ocl_code::kernels_cl_str
    });
    kernel_generate_chains = cl.get_kernel(prog, "generate_chains");
  }

  void build(RainbowTable& rt) {
    auto chain_buf = cl.alloc<cl_ulong>(2 * global_size * block_size);
    auto alphabet_buf = cl.alloc<char>(p.alphabet.size(), CL_MEM_READ_ONLY);
    cl.write_async(alphabet_buf, p.alphabet.c_str(), p.alphabet.size());

    uint64_t lo = p.table_index * p.num_start_values;
    uint64_t hi = lo + p.num_start_values;

    kernel_generate_chains.setArg(1, (cl_ulong)hi);
    kernel_generate_chains.setArg(2, (cl_ulong)p.num_strings);
    kernel_generate_chains.setArg(3, (cl_int)p.chain_len);
    kernel_generate_chains.setArg(4, (cl_int)p.table_index);
    kernel_generate_chains.setArg(5, alphabet_buf);
    kernel_generate_chains.setArg(6, (cl_int)p.alphabet.size());
    kernel_generate_chains.setArg(7, chain_buf);
    kernel_generate_chains.setArg(8, block_size);

    rt.table.resize(p.num_start_values);
    stats.add_timing("time_generate", [&]() {
      utils::Progress progress(hi - lo);
      for (uint64_t offset = lo; offset < hi; offset += block_size * global_size) {
        progress.report(offset - lo);
        kernel_generate_chains.setArg(0, (cl_ulong)offset);
        size_t count = std::min(block_size * global_size, hi - offset);
        cl.run_kernel(
            kernel_generate_chains,
            cl::NDRange((global_size + local_size - 1) / local_size * local_size),
            cl::NDRange(local_size));
        cl.finish_queue();
        cl.read_sync(chain_buf, &rt.table[offset - lo], count);
      }
      progress.finish();
    });

    stats.add_timing("time_sort", [&]() {
      cpu.sort_and_uniqify(rt);
    });
    if (verify) {
      stats.add_timing("time_verify", [&]() {
        int i = 0;
        for (auto& it : rt.table) {
          if (it.first != cpu.construct_chain(it.second,0)) {
            std::cout << i << " " << it.first << " " << it.second << std::endl;
          }
          assert(it.first == cpu.construct_chain(it.second, 0));
          ++i;
        }
      });
    }
    //for (int i = 0; i < rt.table.size(); ++i)
      //assert(rt.table[i].first == i && rt.table[i].second == i);
  }
};
