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
#include "radix_sort.cl.h"
  std::string md5_cl_str(md5_cl, md5_cl + md5_cl_len);
  std::string kernels_cl_str(kernels_cl, kernels_cl + kernels_cl_len);
  std::string radix_sort_cl_str(radix_sort_cl, radix_sort_cl + radix_sort_cl_len);
}

struct GPUImplementation {
  RainbowTableParams p;
  OpenCLApp cl;
  bool verify;
  CPUImplementation& cpu;
  utils::Stats& stats;
  uint64_t block_size, local_size, global_size;
  cl::Buffer alphabet_buf;

  cl::Kernel
    kernel_generate_chains,
    kernel_compute_endpoints,
    kernel_lookup_endpoints,
    kernel_fill_ulong;

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
    kernel_compute_endpoints = cl.get_kernel(prog, "compute_endpoints");
    kernel_lookup_endpoints = cl.get_kernel(prog, "lookup_endpoints");
    kernel_fill_ulong = cl.get_kernel(prog, "fill_ulong");
    alphabet_buf = cl.alloc<char>((p.alphabet.size() + 3) / 4 * 4, CL_MEM_READ_ONLY);
    cl.write_async(alphabet_buf, p.alphabet.c_str(), p.alphabet.size());
  }

  void prefix_scan(cl::Buffer in, cl::Buffer out, uint64_t size) {
    //assert(in != out);
    std::vector<uint32_t> local(size);
    cl.read_sync<uint32_t>(in, local.data(), size);
    uint32_t acc = 0;
    for (uint64_t i = 0; i < size; ++i) {
      uint32_t x = local[i];
      local[i] = acc;
      acc += x;
    }
    cl.write_sync<uint32_t>(out, local.data(), size);
  }

  void sort(
      cl::Buffer buf, int objsize, std::uint64_t bufsize,
      int bits,
      const std::string& type, const std::string& getbit
      )
  {
    assert(bufsize <= std::numeric_limits<std::uint32_t>::max());
    size_t scan_count = (bufsize + 2*local_size - 1) / local_size * local_size;
    auto offset_buf = cl.alloc<std::uint32_t>(scan_count);
    auto buf1 = buf;
    auto buf2 = cl.alloc<char>(objsize * bufsize);
    auto defines = std::string()
      + "#define GET_BIT(x, b) (" + getbit + ")\n"
      + "#define T " + type + "\n"
      + "#define LOCAL_SIZE " + std::to_string(local_size) + "\n";
    auto prog = cl.build_program(std::vector<std::string> {
      defines,
      ocl_code::radix_sort_cl_str
    });
    auto kernel_write_bits = cl.get_kernel(prog, "write_bits");
    auto kernel_partition = cl.get_kernel(prog, "partition");
    kernel_write_bits.setArg(1, (cl_uint)bufsize);
    kernel_partition.setArg(2, (cl_uint)bufsize);
    double t = 0;
    for (int b = 0; b < bits; ++b) {
      kernel_write_bits.setArg(0, buf1);
      kernel_write_bits.setArg(2, (cl_int)b);
      kernel_write_bits.setArg(3, offset_buf);

      kernel_partition.setArg(0, buf1);
      kernel_partition.setArg(1, buf2);
      kernel_partition.setArg(3, offset_buf);

      for (uint64_t offset = 0; offset < scan_count; offset += global_size) {
        kernel_write_bits.setArg(4, (cl_uint)offset);
        uint64_t count = std::min(global_size, scan_count - offset);
        cl.run_kernel(
            kernel_write_bits,
            cl::NDRange((count + local_size - 1) / local_size * local_size),
            cl::NDRange(local_size));
        cl.finish_queue();
      }

      /*
        kernel_write_bits.setArg(4, (cl_uint)0);
        cl.run_kernel(
            kernel_write_bits,
            cl::NDRange(scan_count),
            cl::NDRange(local_size));
      */
      double t0 = utils::get_time();
      prefix_scan(offset_buf, offset_buf, scan_count);
      t += utils::get_time()-t0;
      //std::cout << "FOO" << std::endl;
      //std::vector<uint32_t> x(scan_count);
      //cl.read_sync(offset_buf, x.data(), scan_count);
      //int i = 0;
      //for (auto a: x) std::cout << (i++)<<":" << a << " "; std::cout << std::endl;

      for (uint64_t offset = 0; offset < bufsize; offset += global_size) {
        kernel_partition.setArg(4, (cl_uint)offset);
        uint64_t count = std::min(global_size, bufsize - offset);
        cl.run_kernel(
            kernel_partition,
            cl::NDRange((count + local_size - 1) / local_size * local_size),
            cl::NDRange(local_size));
        cl.finish_queue();
      }
      std::swap(buf1, buf2);
    }
    std::cout << "scan time " << t << std::endl;
    if (bits & 1)
      cl.copy<char>(buf1, buf, objsize * bufsize);
  }

  void build(RainbowTable& rt) {
    auto chain_buf = cl.alloc<cl_ulong>(2 * global_size * block_size);
    //auto debug_buf = cl.alloc<cl_ulong>(global_size * block_size);

    uint64_t lo = p.table_index * p.num_start_values;
    uint64_t hi = lo + p.num_start_values;

    kernel_generate_chains.setArg(1, (cl_ulong)hi);
    kernel_generate_chains.setArg(2, (cl_ulong)p.num_strings);
    kernel_generate_chains.setArg(3, (cl_int)p.chain_len);
    kernel_generate_chains.setArg(4, (cl_int)p.table_index);
    kernel_generate_chains.setArg(5, alphabet_buf);
    kernel_generate_chains.setArg(6, (cl_int)p.alphabet.size());
    kernel_generate_chains.setArg(7, chain_buf);
    kernel_generate_chains.setArg(8, (cl_int)block_size);
    //kernel_generate_chains.setArg(9, debug_buf);

    rt.table.resize(p.num_start_values);
    stats.add_timing("time_generate", [&]() {
      utils::Progress progress(hi - lo);
      for (uint64_t offset = lo; offset < hi; offset += block_size * global_size) {
        progress.report(offset - lo);
        kernel_generate_chains.setArg(0, (cl_ulong)offset);
        size_t count = std::min(block_size * global_size, hi - offset);
        cl.run_kernel(
            kernel_generate_chains,
            cl::NDRange((count + local_size - 1) / local_size * local_size),
            cl::NDRange(local_size));
        cl.read_async(chain_buf, &rt.table[offset - lo], count);
        cl.finish_queue();
        //std::vector<uint64_t> debug(global_size*block_size);
        //cl.read_sync(debug_buf, debug.data(), count);
        //std::cout << std::endl;
        //for (int i = 0; i < count; ++i)
          //std::cout << debug[i] << " ";
        //std::cout << std::endl;
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
          if (it.first != cpu.construct_chain(it.second, 0, p.chain_len).first) {
            std::cout << i << " " << it.first << " " << it.second << std::endl;
            assert(0);
          }
          ++i;
        }
      });
    }
    //for (int i = 0; i < rt.table.size(); ++i)
      //assert(rt.table[i].first == i && rt.table[i].second == i);
  }

  void fill_ulong(cl::Buffer buf, std::size_t size, std::uint64_t val) {
    kernel_fill_ulong.setArg(0, buf);
    kernel_fill_ulong.setArg(1, (cl_ulong)size);
    kernel_fill_ulong.setArg(2, (cl_ulong)val);
    cl.run_kernel(
        kernel_fill_ulong,
        cl::NDRange((size + local_size - 1) / local_size * local_size),
        cl::NDRange(local_size));
  }

  std::vector<std::uint64_t> lookup(
      const RainbowTable& rt,
      const std::vector<Hash>& queries)
  {
    std::uint64_t hi = p.chain_len * queries.size();
    auto query_buf = cl.alloc<Hash>(queries.size(), CL_MEM_READ_ONLY);
    auto lookup_buf = cl.alloc<cl_ulong>(4 * hi);
    cl.write_async(query_buf, queries.data(), queries.size());
    auto debug_buf = cl.alloc<cl_ulong>(hi);

    kernel_compute_endpoints.setArg(1, (cl_ulong)hi);
    kernel_compute_endpoints.setArg(2, (cl_ulong)p.num_strings);
    kernel_compute_endpoints.setArg(3, (cl_int)p.chain_len);
    kernel_compute_endpoints.setArg(4, (cl_int)p.table_index);
    kernel_compute_endpoints.setArg(5, alphabet_buf);
    kernel_compute_endpoints.setArg(6, (cl_int)p.alphabet.size());
    kernel_compute_endpoints.setArg(7, query_buf);
    kernel_compute_endpoints.setArg(8, (cl_int)queries.size());
    kernel_compute_endpoints.setArg(9, lookup_buf);

    stats.add_timing("time_compute_endpoints", [&]() {
      utils::Progress progress(hi);
      for (uint64_t offset = 0; offset < hi; offset += global_size) {
        progress.report(offset);
        kernel_compute_endpoints.setArg(0, (cl_ulong)offset);
        size_t count = std::min(global_size, hi - offset);
        cl.run_kernel(
            kernel_compute_endpoints,
            cl::NDRange((count + local_size - 1) / local_size * local_size),
            cl::NDRange(local_size));
        cl.finish_queue();
      }
      progress.finish();
    });

    if (verify) {
      stats.add_timing("time_verify", [&]() {
        std::vector<std::array<std::uint64_t,4>> lookup(hi);
        cl.read_sync(lookup_buf, lookup.data(), lookup.size());
        for (std::uint64_t i = 0; i < hi; ++i) {
          int start_iteration = i / queries.size();
          int query_idx = i % queries.size();
          assert(lookup[i][1] == start_iteration);
          assert(lookup[i][2] == query_idx);
          std::uint64_t endpoint = cpu.construct_chain(
            queries[query_idx], start_iteration, p.chain_len).first;
          assert(lookup[i][0] == endpoint);
        }
      });
    }

    std::vector<std::array<std::uint64_t,4>> lookup(hi);
    cl.read_sync(lookup_buf, lookup.data(), lookup.size());
    stats.add_timing("time_query_sort", [&]() {
      std::sort(std::begin(lookup), std::end(lookup));
    });
    cl.write_async(lookup_buf, lookup.data(), lookup.size());

    auto result_buf = cl.alloc<std::uint64_t>(queries.size(), CL_MEM_WRITE_ONLY);
    fill_ulong(result_buf, queries.size(), NOT_FOUND);
    auto rt_buf = cl.alloc<RainbowTable::Entry>(rt.table.size(), CL_MEM_READ_ONLY);
    cl.write_async(rt_buf, rt.table.data(), rt.table.size());

    kernel_lookup_endpoints.setArg(1, (cl_ulong)hi);
    kernel_lookup_endpoints.setArg(2, (cl_ulong)p.num_strings);
    kernel_lookup_endpoints.setArg(3, (cl_int)p.chain_len);
    kernel_lookup_endpoints.setArg(4, (cl_int)p.table_index);
    kernel_lookup_endpoints.setArg(5, alphabet_buf);
    kernel_lookup_endpoints.setArg(6, (cl_int)p.alphabet.size());
    kernel_lookup_endpoints.setArg(7, query_buf);
    kernel_lookup_endpoints.setArg(8, lookup_buf);
    kernel_lookup_endpoints.setArg(9, result_buf);
    kernel_lookup_endpoints.setArg(10, rt_buf);
    kernel_lookup_endpoints.setArg(11, (cl_ulong)0);
    kernel_lookup_endpoints.setArg(12, (cl_ulong)rt.table.size());
    //kernel_lookup_endpoints.setArg(12, debug_buf);

    stats.add_timing("time_lookup_endpoints", [&]() {
      utils::Progress progress(hi);
      for (uint64_t offset = 0; offset < hi; offset += global_size) {
        progress.report(offset);
        kernel_lookup_endpoints.setArg(0, (cl_ulong)offset);

        size_t count = std::min(global_size, hi - offset);

        /*
        std::array<std::uint64_t, 4> first, last;
        cl.read_sync(lookup_buf, &first, 1, offset);
        cl.read_sync(lookup_buf, &last, 1, offset + count - 1);
        auto lo = std::lower_bound(std::begin(rt.table), std::end(rt.table),
            std::make_pair(first[0], std::uint64_t{0}));
        auto hi = std::upper_bound(std::begin(rt.table), std::end(rt.table),
            std::make_pair(last[0], std::numeric_limits<std::uint64_t>::max()));
        //std::cout << (lo - std::begin(rt.table)) << " - ";
        //std::cout << (hi - std::begin(rt.table)) << std::endl;
        kernel_lookup_endpoints.setArg(11, (cl_ulong)(lo - std::begin(rt.table)));
        kernel_lookup_endpoints.setArg(12, (cl_ulong)(hi - std::begin(rt.table)));
        */

        cl.run_kernel(
            kernel_lookup_endpoints,
            cl::NDRange((count + local_size - 1) / local_size * local_size),
            cl::NDRange(local_size));
        cl.finish_queue();
      }
      progress.finish();
    });

    //assert(std::is_sorted(std::begin(rt.table),std::end(rt.table)));
    std::vector<std::uint64_t> res(queries.size());
    cl.read_sync(result_buf, res.data(), res.size());
    if (verify) {
      stats.add_timing("time_verify", [&]() {
        for (std::uint64_t i = 0; i < queries.size(); ++i) {
          std::uint64_t cmp = cpu.lookup_single(rt, queries[i]);
          assert(cmp == res[i]);
        }
      });
    }
    return res;
  }
};
