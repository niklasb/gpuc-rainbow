#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <sstream>
#include <vector>

#include <unistd.h>

#include "hash.h"
#include "rainbow_table.h"
#include "utils.h"
#include "opencl.h"
#include "filter.h"

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
  uint64_t block_size;
  const OpenCLConfig& clcfg;
  cl::Buffer alphabet_buf;

  cl::Kernel
    kernel_generate_chains,
    kernel_compute_endpoints,
    kernel_lookup_endpoints,
    kernel_fill_ulong,
    kernel_hash_and_reduce;

  GPUImplementation(
      const RainbowTableParams& p,
      OpenCLApp& cl,
      CPUImplementation& cpu,
      utils::Stats& stats,
      bool verify,
      const OpenCLConfig& clcfg,
      uint64_t block_size
      )
    : p(p), cl(cl), cpu(cpu), stats(stats), verify(verify)
    , block_size(block_size), clcfg(clcfg)
  {
    std::stringstream defines;
    defines
      << "#define ALPHA_SIZE " << p.alphabet.size() << std::endl
      << "#define TABLE_INDEX " << p.table_index << std::endl
      << "#define NUM_STRINGS " << p.num_strings << std::endl
      << "#define CHAIN_LEN " << p.chain_len << std::endl
      << "#define BLOCK_SIZE " << block_size << std::endl
      << "#define LOCAL_SIZE " << clcfg.local_size << std::endl
      << "#define GLOBAL_SIZE " << clcfg.global_size << std::endl
      ;
    auto prog = cl.build_program(std::vector<std::string> {
      defines.str(),
      ocl_code::md5_cl_str,
      ocl_code::kernels_cl_str
    });
    std::ofstream f("kernel.ptx");
    f << cl.get_binary(prog);
    kernel_generate_chains = cl.get_kernel(prog, "generate_chains");
    kernel_compute_endpoints = cl.get_kernel(prog, "compute_endpoints");
    kernel_lookup_endpoints = cl.get_kernel(prog, "lookup_endpoints");
    kernel_fill_ulong = cl.get_kernel(prog, "fill_ulong");
    kernel_hash_and_reduce = cl.get_kernel(prog, "hash_and_reduce");
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

  void run(cl::Kernel kernel, uint32_t size) {
    cl.run_kernel(kernel,
        cl::NDRange(utils::round_to_multiple(size, clcfg.local_size)),
        cl::NDRange(clcfg.local_size));
  }
  void run(cl::Kernel kernel, uint64_t size) {
    assert(size <= std::numeric_limits<std::uint32_t>::max());
    run(kernel, (uint32_t)size);
  }

  void sort(
      cl::Buffer buf, int objsize, std::uint32_t bufsize,
      int bits,
      const std::string& type, const std::string& getbit
      )
  {
    assert(bufsize <= std::numeric_limits<std::uint32_t>::max());
    uint32_t scan_count = utils::round_to_multiple(bufsize + clcfg.local_size, clcfg.local_size);
    auto offset_buf = cl.alloc<std::uint32_t>(scan_count);
    auto buf1 = buf;
    auto buf2 = cl.alloc<char>(objsize * bufsize);
    auto defines = std::string()
      + "#define GET_BIT(x, b) (" + getbit + ")\n"
      + "#define T " + type + "\n"
      + "#define LOCAL_SIZE " + std::to_string(clcfg.local_size) + "\n";
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

      for (uint32_t offset = 0; offset < scan_count; offset += clcfg.global_size) {
        kernel_write_bits.setArg(4, (cl_uint)offset);
        uint32_t count = std::min(clcfg.global_size, scan_count - offset);
        run(kernel_write_bits, count);
        cl.finish_queue();
      }

      double t0 = utils::get_time();
      prefix_scan(offset_buf, offset_buf, scan_count);
      t += utils::get_time()-t0;
      //std::cout << "FOO" << std::endl;
      //std::vector<uint32_t> x(scan_count);
      //cl.read_sync(offset_buf, x.data(), scan_count);
      //int i = 0;
      //for (auto a: x) std::cout << (i++)<<":" << a << " "; std::cout << std::endl;

      for (uint32_t offset = 0; offset < bufsize; offset += clcfg.global_size) {
        kernel_partition.setArg(4, (cl_uint)offset);
        uint32_t count = std::min(clcfg.global_size, bufsize - offset);
        run(kernel_partition, count);
        cl.finish_queue();
      }
      std::swap(buf1, buf2);
    }
    std::cout << "scan time " << t << std::endl;
    if (bits & 1)
      cl.copy<char>(buf1, buf, objsize * bufsize);
  }

  void benchmark_hash_and_reduce(uint32_t iters) {
    uint64_t hashes = iters * clcfg.global_size;
    std::cout << "Computing " << hashes << " hashes" << std::endl;
    auto buf = cl.alloc<cl_ulong>(clcfg.global_size);
    //auto dbg = cl.alloc<cl_uint>(32*clcfg.global_size);
    kernel_hash_and_reduce.setArg(0, alphabet_buf);
    kernel_hash_and_reduce.setArg(1, buf);
    //kernel_hash_and_reduce.setArg(3, dbg);
    cl.finish_queue();
    double t0 = utils::get_time();
    for (uint32_t i = 0; i < iters; ++i) {
      kernel_hash_and_reduce.setArg(2, (cl_ulong)i);
      fill_ulong(buf, clcfg.global_size, uint64_t{i}*clcfg.global_size, 1);
      run(kernel_hash_and_reduce, clcfg.global_size);
    }
    cl.finish_queue();
    double t1 = utils::get_time();
    std::cout << "time = " << t1-t0 << std::endl;
    std::cout << "throughput = " << hashes/(t1-t0)*1e-6 << " mhashes/sec" << std::endl;
    std::vector<cl_ulong> results(clcfg.global_size);
    cl.read_sync(buf, results.data(), clcfg.global_size);

    //std::vector<cl_uint> db_results(clcfg.global_size*16);
    //cl.read_sync(dbg, db_results.data(), clcfg.global_size*16);
    //for (int i =  0; i < 64; ++i) {
      //std::printf("%02x", ((unsigned char*)db_results.data())[i]);
    //}
    //std::cout << std::endl;
    //for (int i =  0; i < 64; ++i) {
      //std::printf("%02x", ((unsigned char*)db_results.data())[i+64]);
    //}
    //std::cout << std::endl;

    for (size_t i = 0; i < clcfg.global_size; ++i) {
      uint64_t start = (uint64_t)(iters-1)*clcfg.global_size + i;
      Hash h;
      cpu.compute_hash(start, h);
      //uint32_t buf[] = { start & 0xffffffff, 0x42424242, 0x41414141 };
      //::compute_hash((unsigned char*)buf, 12, h);
      assert(results[i]==cpu.reduce(h, iters - 1));
    }
  }

  void build(RainbowTable& rt) {
    using C = std::pair<cl_ulong,cl_ulong>;
    //auto chain_buf = cl.alloc<cl_ulong>(2 * clcfg.global_size * block_size);
    //auto debug_buf = cl.alloc<cl_ulong>(global_size * block_size);

    uint64_t lo = p.table_index * p.num_start_values;
    uint64_t hi = lo + p.num_start_values;

    kernel_generate_chains.setArg(1, (cl_ulong)hi);
    kernel_generate_chains.setArg(2, alphabet_buf);
    //kernel_generate_chains.setArg(5, debug_buf);

    uint64_t bufsize = 1<<20;
    auto chain_buf = cl.alloc<C>(bufsize);
    uint64_t chunk = block_size * clcfg.global_size;
    uint64_t total = 0, last_compaction = chunk;
    stats.add_timing("time_generate", [&]() {
      utils::Progress progress(hi - lo);
      uint64_t iter = 0;
      for (uint64_t offset = lo; offset < hi; offset += chunk) {
        progress.report(offset - lo);
        uint64_t count = std::min(chunk, hi - offset);
        //std::cout << "count="<< count << " bufsize=" << bufsize << " total=" << total << std::endl;
        while (total + count > bufsize) {
          auto new_chain_buf = cl.alloc<C>(2*bufsize);
          cl.finish_queue();
          stats.add_timing("time_realloc", [&]() {
            cl.copy<C>(chain_buf, new_chain_buf, bufsize);
            cl.finish_queue();
          });
          chain_buf = new_chain_buf;
          bufsize *= 2;
        }
        //std::cout << "new bufsize=" << bufsize << std::endl;
        kernel_generate_chains.setArg(0, (cl_ulong)offset);
        kernel_generate_chains.setArg(3, chain_buf);
        kernel_generate_chains.setArg(4, (cl_ulong)total);

        run(kernel_generate_chains, count);
        total += count;
        if (total > 2*last_compaction || offset + chunk >= hi) {
          //std::cout << "Compacting. Before: " << total << " After: ";
          cl.finish_queue();
          stats.add_timing("time_compaction", [&]() {
            total = ocl_primitives::remove_dups_inplace(
                cl, clcfg, chain_buf, sizeof(C), total, "ulong2", "x.x < y.x");
            cl.finish_queue();
          });
          //std::cout << total << std::endl;
          last_compaction = total;
        }
        //std::vector<uint64_t> debug(clcfg.global_size*block_size);
        //cl.read_sync(debug_buf, debug.data(), count);
        //std::cout << std::endl;
        //for (int i = 0; i < count; ++i)
          //std::cout << debug[i] << " ";
        //std::cout << std::endl;

        // responsiveness
        cl.finish_queue();
        usleep(1000);
      }
      progress.finish();
    });
    rt.table.resize(total);
    cl.read_sync(chain_buf, rt.table.data(), total);

    if (verify) {
      stats.add_timing("time_sort", [&]() {
        cpu.sort_and_uniqify(rt);
        assert(rt.table.size() == total);
      });
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

  void fill_ulong(cl::Buffer buf, std::uint32_t size, std::uint64_t a, uint64_t b=0) {
    kernel_fill_ulong.setArg(0, buf);
    kernel_fill_ulong.setArg(1, (cl_ulong)size);
    kernel_fill_ulong.setArg(2, (cl_ulong)a);
    kernel_fill_ulong.setArg(3, (cl_ulong)b);
    run(kernel_fill_ulong, size);
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
    kernel_compute_endpoints.setArg(2, alphabet_buf);
    kernel_compute_endpoints.setArg(3, query_buf);
    kernel_compute_endpoints.setArg(4, (cl_int)queries.size());
    kernel_compute_endpoints.setArg(5, lookup_buf);

    stats.add_timing("time_compute_endpoints", [&]() {
      utils::Progress progress(hi);
      for (uint64_t offset = 0; offset < hi; offset += clcfg.global_size) {
        progress.report(offset);
        kernel_compute_endpoints.setArg(0, (cl_ulong)offset);
        size_t count = std::min(uint64_t{clcfg.global_size}, hi - offset);
        run(kernel_compute_endpoints, count);
        // responsiveness
        cl.finish_queue();
        usleep(1000);
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
    assert(queries.size() <= std::numeric_limits<uint32_t>::max());
    fill_ulong(result_buf, (uint32_t)queries.size(), NOT_FOUND);
    auto rt_buf = cl.alloc<RainbowTable::Entry>(rt.table.size(), CL_MEM_READ_ONLY);
    cl.write_async(rt_buf, rt.table.data(), rt.table.size());

    kernel_lookup_endpoints.setArg(1, (cl_ulong)hi);
    kernel_lookup_endpoints.setArg(2, alphabet_buf);
    kernel_lookup_endpoints.setArg(3, query_buf);
    kernel_lookup_endpoints.setArg(4, lookup_buf);
    kernel_lookup_endpoints.setArg(5, result_buf);
    kernel_lookup_endpoints.setArg(6, rt_buf);
    kernel_lookup_endpoints.setArg(7, (cl_ulong)0);
    kernel_lookup_endpoints.setArg(8, (cl_ulong)rt.table.size());
    //kernel_lookup_endpoints.setArg(12, debug_buf);

    stats.add_timing("time_lookup_endpoints", [&]() {
      utils::Progress progress(hi);
      for (uint64_t offset = 0; offset < hi; offset += clcfg.global_size) {
        progress.report(offset);
        kernel_lookup_endpoints.setArg(0, (cl_ulong)offset);

        size_t count = std::min(uint64_t{clcfg.global_size}, hi - offset);

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

        run(kernel_lookup_endpoints, count);
        // responsiveness
        cl.finish_queue();
        usleep(1000);
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
