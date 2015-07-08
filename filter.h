#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"
#include "opencl.h"
#include "scan.h"
#include "bitonic_sort.h"

namespace ocl_code {
#include "filter.cl.h"
  std::string filter_cl_str(filter_cl, filter_cl + filter_cl_len);
}

namespace ocl_primitives {
  std::tuple<cl::Buffer, uint32_t> filter(
      OpenCLApp& cl,
      const OpenCLConfig& clcfg,
      cl::Buffer buf, uint32_t element_size, uint32_t size,
      const std::string& type, const std::string& predicate,
      const std::string& additional_defines = "")
  {
    double t0 = utils::get_time();
    static std::map<std::tuple<std::string, std::string>,
      std::tuple<cl::Kernel, cl::Kernel>> memo;
    auto it = memo.find(std::tie(type, predicate));
    cl::Kernel kernel_set_flags, kernel_compact;
    if (it == std::end(memo)) {
      auto defines = std::string()
        + "#define T " + type + "\n"
        + additional_defines + "\n"
        + "bool predicate(__global T* ary, uint i) { return (" + predicate + "); }\n";
      auto prog = cl.build_program(std::vector<std::string> {
        defines,
        ocl_code::filter_cl_str
      });
      kernel_set_flags = cl.get_kernel(prog, "set_flags");
      kernel_compact = cl.get_kernel(prog, "compact");
      memo[std::tie(type, predicate)] = std::tie(kernel_set_flags, kernel_compact);
    } else {
      std::tie(kernel_set_flags, kernel_compact) = it->second;
    }
    cl::Buffer flags = cl.alloc<uint32_t>(size + 1);
    kernel_set_flags.setArg(0, buf);
    kernel_set_flags.setArg(1, flags);
    kernel_set_flags.setArg(2, size);
    cl.run_kernel(kernel_set_flags,
        cl::NDRange(utils::round_to_multiple(size + 1, clcfg.local_size)),
        cl::NDRange(clcfg.local_size));
    scan_naive(cl, clcfg, flags, sizeof(uint32_t), size + 1,
      "uint", "x + y", "0");
    uint32_t total;
    cl.read_sync<uint32_t>(flags, &total, 1, size);
    cl::Buffer res = cl.alloc<char>(element_size * total);
    kernel_compact.setArg(0, buf);
    kernel_compact.setArg(1, res);
    kernel_compact.setArg(2, flags);
    kernel_compact.setArg(3, size);
    cl.run_kernel(kernel_compact,
        cl::NDRange(utils::round_to_multiple(size, clcfg.local_size)),
        cl::NDRange(clcfg.local_size));
    return std::tie(res, total);
  }

  uint32_t filter_inplace(
      OpenCLApp& cl,
      const OpenCLConfig& clcfg,
      cl::Buffer buf, uint32_t element_size, uint32_t size,
      const std::string& type, const std::string& predicate)
  {
    cl::Buffer res;
    uint32_t total;
    std::tie(res, total) = filter(cl, clcfg, buf, element_size, size, type, predicate);
    cl.copy<char>(res, buf, total * element_size);
    return total;
  }

  std::tuple<cl::Buffer, uint32_t> remove_dups(
    OpenCLApp& cl,
    const OpenCLConfig& clcfg,
    cl::Buffer buf, uint32_t element_size, uint32_t size,
    const std::string& type, const std::string& less)
  {
    bitonic_sort(cl, clcfg, buf, size, type, less);
    return filter(cl, clcfg, buf, element_size, size, type,
        "i == 0 || less(ary[i-1], ary[i])",
        "bool less(T x, T y) { return (" + less + "); }\n");
  }

  uint32_t remove_dups_inplace(
    OpenCLApp& cl,
    const OpenCLConfig& clcfg,
    cl::Buffer buf, uint32_t element_size, uint32_t size,
    const std::string& type, const std::string& less)
  {
    cl::Buffer res;
    uint32_t total;
    std::tie(res, total) = remove_dups(cl, clcfg, buf, element_size, size, type, less);
    cl.copy<char>(res, buf, total * element_size);
    return total;
  }

  void test_filter(OpenCLApp& cl, const OpenCLConfig& clcfg) {
    for (int N : { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7 }) {
      std::cout << "testing filter with " << N << " elements" << std::endl;
      cl::Buffer buf = cl.alloc<uint32_t>(N);
      std::vector<uint32_t> z(N), x, y;
      for (int i = 0; i < N; ++i)
        z[i] = std::rand()%1000;
      //for (auto a: x) std::cout << a << " "; std::cout << std::endl;
      cl.write_sync(buf, z.data(), N);

      double t0 = utils::get_time();

      // CPU
      int cnt = 0;
      for (int i = 0; i < N; ++i)
        if (z[i] % 2 == 0)
          cnt++;
      x.resize(cnt);
      int j = 0;
      for (int i = 0; i < N; ++i)
        if (z[i] % 2 == 0)
          x[j++] = z[i];
      double t1 = utils::get_time();

      // GPU
      uint32_t total = filter_inplace(cl, clcfg, buf, sizeof(uint32_t), N, "uint", "ary[i] % 2 == 0");
      cl.finish_queue();
      double t2 = utils::get_time();
      std::cout << (t1-t0) << " " <<(t2-t1) << std::endl;

      if (total != x.size()) {
        std::cout << "total=" << total << " real=" << x.size() << std::endl;
      }
      assert(total == x.size());
      y.resize(total);
      cl.read_sync(buf, y.data(), total);
      if (x != y && N < 100) {
        std::cout << "x="; for (auto a: x) std::cout << a << " "; std::cout << std::endl;
        std::cout << "y="; for (auto a: y) std::cout << a << " "; std::cout << std::endl;
      }
      assert(x == y);
    }
  }

  void test_remove_dups(OpenCLApp& cl, const OpenCLConfig& clcfg) {
    for (int N : { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7 }) {
      std::cout << "testing remove_dups with " << N << " elements" << std::endl;
      cl::Buffer buf = cl.alloc<uint32_t>(N);
      std::vector<uint32_t> x(N), y;
      for (int i = 0; i < N; ++i)
        x[i] = std::rand()%1000;
      //for (auto a: x) std::cout << a << " "; std::cout << std::endl;
      cl.write_sync(buf, x.data(), N);

      double t0 = utils::get_time();
      // CPU
      std::sort(std::begin(x), std::end(x));
      x.erase(std::unique(std::begin(x), std::end(x)), std::end(x));
      double t1 = utils::get_time();

      // GPU
      uint32_t total = remove_dups_inplace(cl, clcfg, buf, sizeof(uint32_t), N, "uint", "x < y");
      cl.finish_queue();
      double t2 = utils::get_time();
      std::cout << (t1-t0) << " " <<(t2-t1) << std::endl;

      if (total != x.size()) {
        std::cout << "total=" << total << " real=" << x.size() << std::endl;
      }
      assert(total == x.size());
      y.resize(total);
      cl.read_sync(buf, y.data(), total);
      if (x != y && N < 100) {
        std::cout << "x="; for (auto a: x) std::cout << a << " "; std::cout << std::endl;
        std::cout << "y="; for (auto a: y) std::cout << a << " "; std::cout << std::endl;
      }
      assert(x == y);
    }
  }
}
