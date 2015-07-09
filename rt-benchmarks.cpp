#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_set>

#if !HAVE_OPENCL
#  error "Need OpenCL"
#endif
#include "opencl.h"
#include "hash.h"
#include "rainbow_table.h"
#include "rainbow_cpu.h"
#include "rainbow_gpu.h"
#include "utils.h"

using namespace std;

void usage(char *argv0) {
  cerr << "Usage: " << argv0 << " [FLAGS] [benchmark_name benchmark_opts]"
       << endl
       << "BENCHMARKS" << endl
       << "  hash_and_reduce chain_len:INT (default)" << endl
       << endl
       << "FLAGS" << endl
       << "  -i INT     Benchmark iterations" << endl
       << "  -l INT     OpenCL only: local group size" << endl
       << "  -g INT     OpenCL only: global group size" << endl
       << "  -b INT     OpenCL only: block size" << endl;
  exit(EXIT_FAILURE);
}

uint32_t block_size = 1;
OpenCLConfig clcfg { 1<<17, 1<<8 };
string benchmark = "hash_and_reduce";
uint32_t chain_len = 1;
uint32_t iterations = 1000;

void parse_opts(int argc, char *argv[]) {
  int pos = 0;
  int options = 0;
  for (int i = 1; i < argc; ++i) {
    string o(argv[i]);
    // 0 params
    if (o == "-h") {
      usage(argv[0]);
      continue;
    }
    if (o == "-i") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> iterations) || !iterations) {
          cout << "ERROR: iteration count should be a positive integer" << endl;
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-l") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> clcfg.local_size)) {
          cout << "ERROR: local group size should be an integer" << endl;
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-g") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> clcfg.global_size)) {
          cout << "ERROR: global group size should be an integer" << endl;
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-b") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> block_size)) {
          cout << "ERROR: block size should be an integer" << endl;
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    // positional
    if (pos == 0) {
      if (o.empty() || (o != "hash_and_reduce")) {
        cerr << "ERROR: benchmark name unknown" << endl;
        usage(argv[0]);
      }
      benchmark = o;
    }
    if (pos == 1) {
      if (benchmark == "hash_and_reduce") {
        if (!(stringstream(o) >> chain_len), chain_len == 0) {
          cerr << "ERROR: chain len must be an integer > 0" << endl;
          usage(argv[0]);
        }
      } else {
        usage(argv[0]);
      }
    }
    pos++;
  }
  if (pos > 2)
    usage(argv[0]);
}

int main_(int argc, char* argv[]) {
  parse_opts(argc, argv);
  utils::Stats stats;

  OpenCLApp cl;
  cl.print_cl_info();

  RainbowTableParams params;
  params.chain_len = 1000;
  params.alphabet = "abcdefghijklmnopqrstuvwxyz";
  params.num_strings = 1e9;
  CPUImplementation cpu(params, stats);
  GPUImplementation gpu(params, cl, cpu, stats, false, clcfg, block_size);

  if (benchmark == "hash_and_reduce") {
    cout << "Benchmark hash_and_reduce, " << iterations
      << " iterations, chain len " << chain_len << endl;
    gpu.benchmark_hash_and_reduce(iterations, chain_len);
  } else {
    cerr << "ERROR: No such benchmark: `" << benchmark << "'" << endl;
    usage(argv[0]);
  }

  cout << "STATS" << endl;
  for (auto& it : stats.stats) {
    cout << "  " << it.first << " = " << it.second << endl;
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    return main_(argc, argv);
  } catch (cl::Error err) {
    cerr << "OpenCL exception: " << err.what() << " (" << err.err() << ")" << endl;
    return EXIT_FAILURE;
  }
}
