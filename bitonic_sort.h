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
#include "bitonic.cl.h"
  std::string bitonic_cl_str(bitonic_cl, bitonic_cl + bitonic_cl_len);
}

void bitonic_sort(
    OpenCLApp& cl,
    const OpenCLConfig& clcfg,
    cl::Buffer buf, uint32_t size,
    const std::string& type, const std::string& comp)
{
  auto defines = std::string()
    + "#define T " + type + "\n"
    + "bool less(T x, T y) { return (" + comp + "); }\n";
  auto prog = cl.build_program(std::vector<std::string> {
    defines,
    ocl_code::bitonic_cl_str
  });
  auto kernel_cross = cl.get_kernel(prog, "cross");
  auto kernel_inc = cl.get_kernel(prog, "inc");
  for (uint32_t i = 1; i < size; i <<= 1) {
    kernel_cross.setArg(0, buf);
    kernel_cross.setArg(1, (cl_uint)size);
    kernel_cross.setArg(2, (cl_uint)0);
    kernel_cross.setArg(3, (cl_uint)i);
    uint32_t global_size = utils::round_to_multiple(size/2, clcfg.local_size);
    cl.run_kernel(kernel_cross,
        cl::NDRange(global_size), cl::NDRange(clcfg.local_size));
    for (uint32_t j = i/2; j >= 1; j >>= 1) {
      kernel_inc.setArg(0, buf);
      kernel_inc.setArg(1, (cl_uint)size);
      kernel_inc.setArg(2, (cl_uint)0);
      kernel_inc.setArg(3, (cl_uint)j);
      uint32_t global_size = utils::round_to_multiple(size/2, clcfg.local_size);
      cl.run_kernel(kernel_inc,
          cl::NDRange(global_size), cl::NDRange(clcfg.local_size));
    }
  }
}
