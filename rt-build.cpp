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
       << "  -t INT   Positive integer value specifying the rainbow construct_chain" << endl
       << "           length" << endl
       << "  -s INT   Integer value specifying the number of samples to use" << endl
       << "           for coverage analysis. 0 means no coverage is measured" << endl
       << "  -i INT   Table index in case multiple tables are generated" << endl
       << "  -r       Specify random seed (defaults to constant value)" << endl
       << "  -v       OpenCL only: Verify results using CPU implementation" << endl
       << "  -b INT   OpenCL only: block size" << endl
       << "  -l INT   OpenCL only: local group size" << endl
       << "  -g INT   OpenCL only: global group size" << endl
       << endl
       << "EXAMPLES" << endl
       << "  " << argv0 << " -t 7 abcdefghijklmnopqrstuvwxyz0123456789" << endl;
  exit(EXIT_FAILURE);
}


uint64_t max_string_len;
bool use_opencl = false;
double alpha = 0.01;
uint64_t samples = 0;
uint64_t seed = 0;
RainbowTableParams params;
string outfile;
bool verify = 0;
uint64_t local_size = 1<<8;
uint64_t global_size = 1<<15;
uint64_t block_size = 1<<5;

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
        if (!(stringstream(argv[i+1]) >> local_size)) {
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
        if (!(stringstream(argv[i+1]) >> global_size)) {
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

  cout << setprecision(2) << fixed;
  cout << "PARAMETERS" << endl;
  cout << "  string_len  = " << max_string_len << endl;
  cout << "  alphabet    = " << params.alphabet << endl;
  cout << "  num_strings = " << params.num_strings << endl;
  cout << "  alpha       = " << alpha << endl;
  cout << "  t           = " << params.chain_len << endl;
  cout << "  rand seed   = " << seed << endl;
  if (samples)
    cout << "  cov samples = " << samples << endl;
  cout << "  use OpenCL  = " << (use_opencl?"yes":"no") << endl;
  if (use_opencl) {
    cout << "  verify      = " << (verify?"yes":"no") << endl;
    cout << "  block size  = " << block_size << endl;
    cout << "  local size  = " << local_size << endl;
    cout << "  global size = " << global_size << endl;
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
  GPUImplementation gpu(params, cl, cpu, stats, verify,
      local_size, global_size, block_size);
  if (use_opencl) {
    cl.print_cl_info();
  }

  {
  const int N = 10000101;
  cl::Buffer buf = cl.alloc<uint64_t>(N);
  vector<uint64_t> x(N), y(N);
  for (int i = 0; i < N; ++i)
    x[i] = rand()%1000;
  //for (auto a: x) cout << a << " "; cout << endl;
  cl.write_sync(buf, x.data(), N);

  double t0 = utils::get_time();
  sort(begin(x),end(x));
  double t1 = utils::get_time();
  gpu.sort(buf, sizeof(uint64_t), N, 64/4, "ulong", "((x)>>(b))&1");
  cl.finish_queue();
  double t2 = utils::get_time();
  cout << (t1-t0) << " " <<(t2-t1) << endl;

  cl.read_sync(buf, y.data(), N);
  //for (auto a: x) cout << a << " "; cout << endl;
  //for (auto a: y) cout << a << " "; cout << endl;
  assert(x == y);
  return 0;
  }

  if (use_opencl) {
    gpu.build(rt);
  } else {
    cpu.build(rt);
  }

  cout << "Result: " << rt.table.size() << " unique chains (~"
       << setprecision(2) << fixed << (100. * rt.table.size() / params.num_strings)
       << "% of search space)" << endl;

  if (samples) {
    //cout << "Measuring exact coverage" << endl;
    //unordered_set<uint64_t> covered;
    //stats.add_timing("time_coverage_exact", [&]() {
      //utils::Progress progress(rt.table.size());
      //for (size_t i = 0; i < rt.table.size(); ++i) {
        //auto& entry = rt.table[i];
        //progress.report(i);
        ////cout << "start=" << entry.second << endl;
        //[>
        //construct_chain(entry.second, 0, [&](uint64_t x, const Hash& h) {
          //cout << x << " "; print_hash(h); cout << endl;
        //});
        //*/
        //uint64_t endpoint = cpu.construct_chain(entry.second, 0, [&](uint64_t x, const Hash& h) {
          //covered.insert(x);
          ////assert(is_covered(x, rainbow_table));

          //[>
          //if (!is_covered(x, rainbow_table)) {
            //Hash h;
            //compute_hash(x, h);
            //uint64_t res;
            ////cout << "lookup "; print_hash(h); cout << endl;
            ////cout << "lookup result=" << lookup(h, rainbow_table, res, true) << endl;
            //cout << res<< endl;
          //}
          //*/
        //});
        //assert(endpoint == entry.first);
      //}
      //progress.finish();
    //});
    //cout << "Coverage: " << setprecision(4)
        //<< fixed << (100.*covered.size()/params.num_strings) << "%" << endl;
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
      if (use_opencl) {
        for (auto x: gpu.lookup(rt, queries))
          found += x != NOT_FOUND;
      } else {
        for (auto x: cpu.lookup(rt, queries))
          found += x != NOT_FOUND;
      }
    });
    cout << "Coverage: " << setprecision(4) << fixed
        << (100.*found/samples) << "%" << endl;
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
