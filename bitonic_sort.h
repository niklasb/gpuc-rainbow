#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "utils.h"
#include "opencl.h"

namespace ocl_code {
#include "bitonic.cl.h"
  std::string bitonic_cl_str(bitonic_cl, bitonic_cl + bitonic_cl_len);
}

namespace ocl_primitives {
  void bitonic_sort(
      OpenCLApp& cl,
      const OpenCLConfig& clcfg,
      cl::Buffer buf, uint32_t size,
      const std::string& type, const std::string& comp)
  {
    //double t0 = utils::get_time();
    static std::map<std::tuple<std::string, std::string>,
      std::tuple<cl::Kernel, cl::Kernel>> memo;
    auto it = memo.find(std::tie(type, comp));
    cl::Kernel kernel_cross, kernel_inc;
    if (it == std::end(memo)) {
      auto defines = std::string()
        + "#define T " + type + "\n"
        + "bool less(T x, T y) { return (" + comp + "); }\n";
      auto prog = cl.build_program(std::vector<std::string> {
        defines,
        ocl_code::bitonic_cl_str
      });
      kernel_cross = cl.get_kernel(prog, "bitonic_cross");
      kernel_inc = cl.get_kernel(prog, "bitonic_inc");
      memo[std::tie(type, comp)] = std::tie(kernel_cross, kernel_inc);
    } else {
      std::tie(kernel_cross, kernel_inc) = it->second;
    }
    kernel_cross.setArg(0, buf);
    kernel_cross.setArg(1, (cl_uint)size);
    kernel_cross.setArg(2, (cl_uint)0);
    kernel_inc.setArg(0, buf);
    kernel_inc.setArg(1, (cl_uint)size);
    kernel_inc.setArg(2, (cl_uint)0);
    //std::cout << "loading time " << utils::get_time()-t0 << std::endl;
    for (uint32_t i = 1; i < size; i <<= 1) {
      //t0 = utils::get_time();
      kernel_cross.setArg(3, (cl_uint)i);
      uint32_t global_size = utils::round_to_multiple(size/2, clcfg.local_size);
      cl.run_kernel(kernel_cross,
          cl::NDRange(global_size), cl::NDRange(clcfg.local_size));
      for (uint32_t j = i/2; j >= 1; j >>= 1) {
        kernel_inc.setArg(3, (cl_uint)j);
        uint32_t global_size = utils::round_to_multiple(size/2, clcfg.local_size);
        cl.run_kernel(kernel_inc,
            cl::NDRange(global_size), cl::NDRange(clcfg.local_size));
      }
      //std::cout << i << " " << utils::get_time()-t0 << std::endl;
    }
  }

  void test_bitonic_sort(OpenCLApp& cl, const OpenCLConfig& clcfg) {
    for (int N : { 1e3, 1e4, 1e5, 1e6, 1e7, 1e8 }) {
      std::cout << "testing bitonic sort with " << N << " elements" << std::endl;
      cl::Buffer buf = cl.alloc<uint32_t>(N);
      std::vector<uint32_t> x(N), y(N);
      for (int i = 0; i < N; ++i)
        x[i] = std::rand()%1000;
      //for (auto a: x) cout << a << " "; cout << endl;
      cl.write_sync(buf, x.data(), N);

      double t0 = utils::get_time();
      sort(begin(x),end(x));
      double t1 = utils::get_time();
      bitonic_sort(cl, clcfg, buf, N, "uint", "x<y");
      bitonic_sort(cl, clcfg, buf, N, "uint", "x<y");
      bitonic_sort(cl, clcfg, buf, N, "uint", "x<y");
      cl.finish_queue();
      double t2 = utils::get_time();
      std::cout << (t1-t0) << " " <<(t2-t1)/3.0 << std::endl;
      cl.read_sync(buf, y.data(), N);
      assert(x == y);
    }
  }
}
