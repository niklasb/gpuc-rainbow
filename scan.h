#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"
#include "opencl.h"

namespace ocl_code {
#include "scan.cl.h"
  std::string scan_cl_str(scan_cl, scan_cl + scan_cl_len);
}

namespace ocl_primitives {
  void scan_naive(
      OpenCLApp& cl,
      const OpenCLConfig& clcfg,
      cl::Buffer buf, uint32_t element_size, uint32_t size,
      const std::string& type, const std::string& combine, const std::string& id)
  {
    double t0 = utils::get_time();
    static std::map<std::tuple<std::string, std::string, std::string>,
      std::tuple<cl::Kernel, cl::Kernel>> memo;
    auto it = memo.find(std::tie(type, combine, id));
    cl::Kernel kernel_naive, kernel_shift;
    if (it == std::end(memo)) {
      auto defines = std::string()
        + "#define T " + type + "\n"
        + "T combine(T x, T y) { return (" + combine + "); }\n"
        + "#define IDENTITY (" + id + ")\n";
      auto prog = cl.build_program(std::vector<std::string> {
        defines,
        ocl_code::scan_cl_str
      });
      kernel_naive = cl.get_kernel(prog, "scan_naive");
      kernel_shift = cl.get_kernel(prog, "shift");
      memo[std::tie(type, combine, id)] = std::tie(kernel_naive, kernel_shift);
    } else {
      std::tie(kernel_naive, kernel_shift) = it->second;
    }
    cl::Buffer ping = buf;
    cl::Buffer pong = cl.alloc<char>(size * element_size);
    int cnt = 0;
    for (uint32_t offset = 1; offset < size; offset *= 2) {
      cnt++;
      kernel_naive.setArg(0, ping);
      kernel_naive.setArg(1, pong);
      kernel_naive.setArg(2, (cl_uint)size);
      kernel_naive.setArg(3, (cl_uint)offset);
      cl.run_kernel(kernel_naive,
          cl::NDRange(utils::round_to_multiple(size, clcfg.local_size)),
          cl::NDRange(clcfg.local_size));
      std::swap(ping, pong);
    }
    kernel_shift.setArg(0, ping);
    kernel_shift.setArg(1, pong);
    kernel_shift.setArg(2, (cl_uint)size);
    cl.run_kernel(kernel_shift,
        cl::NDRange(utils::round_to_multiple(size, clcfg.local_size)),
        cl::NDRange(clcfg.local_size));
    cnt++;
    std::swap(ping, pong);
    if (cnt % 2)
      cl.copy<char>(ping, buf, element_size * size);
  }

  template <typename F>
  void test_scan(OpenCLApp& cl, const OpenCLConfig& clcfg, F f) {
    for (int N : { 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8 }) {
      std::cout << "testing scan with " << N << " elements" << std::endl;
      cl::Buffer buf = cl.alloc<uint32_t>(N);
      std::vector<uint32_t> x(N), y(N);
      for (int i = 0; i < N; ++i)
        x[i] = std::rand()%1000;
      //for (auto a: x) std::cout << a << " "; std::cout << std::endl;
      cl.write_sync(buf, x.data(), N);

      double t0 = utils::get_time();
      uint32_t acc = 0;
      for (int i = 0; i < N; ++i) {
        int val = x[i];
        x[i] = acc;
        acc += val;
      }
      double t1 = utils::get_time();
      f(cl, clcfg, buf, sizeof(uint32_t), N, "uint", "x+y", "0");
      cl.finish_queue();
      double t2 = utils::get_time();
      std::cout << (t1-t0) << " " <<(t2-t1) << std::endl;
      cl.read_sync(buf, y.data(), N);
      if (x != y && N < 100) {
        std::cout << "x="; for (auto a: x) std::cout << a << " "; std::cout << std::endl;
        std::cout << "y="; for (auto a: y) std::cout << a << " "; std::cout << std::endl;
      }
      assert(x == y);
    }
  }
}
