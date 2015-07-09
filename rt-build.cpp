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
#include "bitonic_sort.h"
#include "scan.h"
#include "filter.h"
#include "utils.h"

using namespace std;

void usage(char *argv0) {
  cerr << "Usage: " << argv0 << " [FLAGS] string_len alphabet outfile" << endl
       << endl
       << "string_len" << endl
       << "  an integer representing the maximum length of strings covered by " << endl
       << "  the generated rainbow table" << endl
       << endl
       << "alphabet" << endl
       << "  the alphabet used to build the strings covered by the rainbow table" << endl
       << endl
       << "FLAGS" << endl
       << "  -h       Show this help" << endl
       << "  -o       Use OpenCL to accelerate the table generation" << endl
       << "  -a FLOAT Floating point value between 0 and 1, indicating the fraction" << endl
       << "           of passwords used as initial values" << endl
       << "  -t INT   Positive integer value specifying the rainbow chain" << endl
       << "           length" << endl
       << "  -s INT   Integer value specifying the number of samples to use" << endl
       << "           for coverage analysis. 0 means no coverage is measured" << endl
       << "  -i INT   Table index in case multiple tables are generated" << endl
       << "  -r INT   Specify random seed (defaults to constant value)" << endl
       << "  -v       OpenCL only: Verify results using CPU implementation" << endl
       << "  -b INT   OpenCL only: block size" << endl
       << "  -l INT   OpenCL only: local group size" << endl
       << "  -g INT   OpenCL only: global group size" << endl
       << endl
       << "EXAMPLES" << endl
       << "  " << argv0 << " -o 7 abcdefghijklmnopqrstuvwxyz0123456789 outputfile" << endl;
  exit(EXIT_FAILURE);
}


uint64_t max_string_len;
bool use_opencl = false, verify = false;
double alpha = 0.01;
uint64_t samples = 0;
uint64_t seed = 0;
RainbowTableParams params;
string outfile;
uint64_t block_size = 1;
OpenCLConfig clcfg { 1<<17, 1<<8 };

const int default_chain_len = 1000;
const int default_table_index = 0;

const double eps = 1e-9;

void parse_opts(int argc, char *argv[]) {
  int pos = 0;
  // defaults
  params.chain_len = default_chain_len;
  params.table_index = default_table_index;

  for (int i = 1; i < argc; ++i) {
    string o(argv[i]);
    // 0 params
    if (o == "-h") {
      usage(argv[0]);
      continue;
    }
    if (o == "-o") {
      use_opencl = true;
      continue;
    }
    if (o == "-v") {
      verify = true;
      continue;
    }
    // 1 params
    if (o == "-a") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> alpha)  || alpha < eps || alpha > 1 + eps) {
          cerr << "ERROR: alpha should be a float in the range (0, 1]" << endl;
          usage(argv[0]);
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-i") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> params.table_index)) {
          cerr << "ERROR: table index should be an integer >= 0" << endl;
          usage(argv[0]);
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-t") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> params.chain_len) || params.chain_len == 0) {
          cerr << "ERROR: t should be an integer > 0" << endl;
          usage(argv[0]);
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-s") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> samples)) {
          cout << "ERROR: samples should be an integer" << endl;
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-r") {
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> seed)) {
          cout << "ERROR: seed should be an integer" << endl;
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
      if (!(stringstream(o) >> max_string_len) || max_string_len <= 0) {
        cerr << "ERROR: string length should be an integer > 0" << endl;
        usage(argv[0]);
      }
    }
    if (pos == 1) {
      params.alphabet = o;
      if (params.alphabet.size() < 1) {
        cerr << "ERROR: alphabet should be a string of length >= 1" << endl;
      }
    }
    if (pos == 2) {
      outfile = o;
      if (outfile.size() < 1) {
        cerr << "ERROR: output filename must be a string of length >= 1" << endl;
      }
    }
    pos++;
  }
  if (pos != 3)
    usage(argv[0]);
}

int main_(int argc, char* argv[]) {
  parse_opts(argc, argv);

  params.num_strings = 0;
  uint64_t cur = 1;
  for (int i = 0; i <= max_string_len; ++i) {
    params.num_strings += cur;
    cur *= params.alphabet.size();
  }

  cout << setprecision(4) << fixed;
  cout << "PARAMETERS" << endl;
  cout << "  string_len  = " << max_string_len << endl;
  cout << "  alphabet    = " << params.alphabet << endl;
  cout << "  num_strings = " << params.num_strings << endl;
  cout << "  alpha       = " << alpha << endl;
  cout << "  t           = " << params.chain_len << endl;
  cout << "  table_index = " << params.table_index << endl;
  cout << "  rand seed   = " << seed << endl;
  if (samples)
    cout << "  cov samples = " << samples << endl;
  cout << "  use OpenCL  = " << (use_opencl?"yes":"no") << endl;
  if (use_opencl) {
    cout << "  verify      = " << (verify?"yes":"no") << endl;
    cout << "  block size  = " << block_size << endl;
    cout << "  local size  = " << clcfg.local_size << endl;
    cout << "  global size = " << clcfg.global_size << endl;
  }

  params.num_start_values =
    min((uint64_t)(alpha * params.num_strings), params.num_strings);

  cout << "Computing " << params.num_start_values << " chains ("
      << (100.*params.num_start_values / params.num_strings)
      << "% of search space)" << endl;

  RainbowTable rt;
  utils::Stats stats;
  CPUImplementation cpu(params, stats);
  OpenCLApp cl;
  GPUImplementation gpu(params, cl, cpu, stats, verify, clcfg, block_size);
  if (use_opencl) {
    cl.print_cl_info();
  }

  if (use_opencl)
    gpu.build(rt);
  else
    cpu.build(rt);

  //ocl_primitives::test_filter(cl, clcfg); return 0;
  cout << setprecision(4);
  cout << "Result: " << rt.table.size() << " unique chains (~"
       << (100. * rt.table.size() / params.num_strings)
       << "% of search space)" << endl;

  if (samples) {
    uint64_t found = 0;
    std::mt19937 gen(seed);
    cout << "Estimating coverage using " << samples << " samples" << endl;
    stats.add_timing("time_coverage_sampling", [&]() {
      vector<Hash> queries;
      Hash h;
      for (uint64_t i = 0; i < samples; ++i) {
        uint64_t sample = (((uint64_t)gen() << 32) | gen()) % params.num_strings;
        cpu.compute_hash(sample, h);
        queries.push_back(h);
      }
      for (auto x: use_opencl ? gpu.lookup(rt, queries) : cpu.lookup(rt, queries))
        found += x != NOT_FOUND;
    });
    cout << setprecision(4);
    cout << "COVERAGE " << (100.*found/samples) << "%" << endl;
  }

  cout << "Writing table to disk" << endl;
  stats.add_timing("time_write_table", [&]() {
    cout << "  " << outfile + ".params" << endl;
    params.save_to_disk(outfile + ".params");
    cout << "  " << outfile << endl;
    rt.save_to_disk(outfile);
  });

  cout << "STATS" << endl;
  for (auto& it : stats.stats) {
    cout << "  " << it.first << " = " << it.second << endl;
  }
  cout << "  throughput_generate = "
    << params.num_start_values * params.chain_len / stats.stats["time_generate"] * 1e-6
    << " mhashes/sec" << endl;
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
