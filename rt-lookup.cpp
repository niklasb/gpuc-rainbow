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
  cerr << "Usage: " << argv0 << " [FLAGS] [-f infile | -H hash | -s samples] "
       << "table_file1 [table_file2 ...]" << endl
       << endl
       << "MODES" << endl
       << "  -f STRING  Read hashes from file" << endl
       << "  -l STRING  Look up given hash" << endl
       << "  -s INT     Use random sampling to estimate coverage" << endl
       << endl
       << "FLAGS" << endl
       << "  -o         Use OpenCL to accelerate the lookup" << endl
       << "  -v         Verify results with CPU" << endl
       << "  -r INT     Specify random seed (defaults to constant value)" << endl
       << "  -l INT     OpenCL only: local group size" << endl
       << "  -g INT     OpenCL only: global group size" << endl
       << "  -b INT     OpenCL only: block size" << endl
       << endl
       << "EXAMPLES" << endl
       << "  " << argv0 << " -H a4d80eac9ab26a4a2da04125bc2c096a alphalow_num_6" << endl
       ;
  exit(EXIT_FAILURE);
}

bool use_opencl = false, verify = false;
string infile;
vector<string> table_files;
Hash hash_value;
uint32_t block_size = 1;
OpenCLConfig clcfg { 1<<17, 1<<8 };
uint64_t seed = 0;
uint32_t samples = 0;

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
    if (o == "-o") {
      use_opencl = true;
      continue;
    }
    if (o == "-v") {
      verify = true;
      continue;
    }
    // 1 params
    if (o == "-f") {
      options++;
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> infile) || infile.empty()) {
          cerr << "ERROR: file name should be a non-empty string" << endl;
          usage(argv[0]);
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    if (o == "-s") {
      options++;
      if (i + 1 < argc) {
        if (!(stringstream(argv[i+1]) >> samples) || !samples) {
          cerr << "ERROR: samples must be an integer > 0" << endl;
          usage(argv[0]);
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
          cerr << "ERROR: seed must be an integer" << endl;
          usage(argv[0]);
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
    if (o == "-H") {
      options++;
      if (i + 1 < argc) {
        if (!parse_hash(argv[i+1], hash_value)) {
          cerr << "ERROR: argument `" << argv[i+1] << "' is not a valid hash value" << endl;
        }
      } else {
        usage(argv[0]);
      }
      ++i;
      continue;
    }
    // positional
    if (o.empty()) {
      cerr << "ERROR: table file names must be non-empty" << endl;
      usage(argv[0]);
    }
    table_files.push_back(o);
    pos++;
  }
  if (pos < 1 || options != 1)
    usage(argv[0]);
}

vector<uint64_t> lookup_any(
    RainbowTableParams& first_params,
    OpenCLApp& cl, utils::Stats& stats,
    vector<Hash> queries) {
  vector<uint64_t> all_results(queries.size(), NOT_FOUND);
  vector<size_t> indices;
  for (size_t i = 0; i < queries.size(); ++i)
    indices.push_back(i);
  for (auto table_file: table_files) {
    if (queries.empty())
      break;
    cout << "TABLE " << table_file << endl;
    RainbowTableParams params;
    params.read_from_disk(table_file + ".params");
    if (tie(params.num_strings, params.alphabet) !=
        tie(first_params.num_strings, first_params.alphabet))
    {
      cerr << "ERROR: Inconsistent alphabets between tables" << endl;
      exit(EXIT_FAILURE);
    }

    cout << setprecision(4) << fixed;
    cout << "PARAMETERS" << endl;
    cout << "  alphabet    = " << params.alphabet << endl;
    cout << "  num_strings = " << params.num_strings << endl;
    cout << "  t           = " << params.chain_len << endl;
    cout << "  table_index = " << params.table_index << endl;
    cout << "  use OpenCL  = " << (use_opencl?"yes":"no") << endl;
    if (use_opencl) {
      cout << "  verify      = " << (verify?"yes":"no") << endl;
      cout << "  block size  = " << block_size << endl;
      cout << "  local size  = " << clcfg.local_size << endl;
      cout << "  global size = " << clcfg.global_size << endl;
    }

    RainbowTable rt;
    stats.add_timing("time_read_table", [&]() {
      cout << "Reading table from file " << table_file << endl;
      rt.read_from_disk(table_file);
    });

    CPUImplementation cpu(params, stats);
    GPUImplementation gpu(params, cl, cpu, stats, verify, clcfg, block_size);

    vector<uint64_t> results;
    stats.add_timing("time_lookup", [&]() {
      results = use_opencl ? gpu.lookup(rt, queries) : cpu.lookup(rt, queries);
    });
    assert(queries.size() == results.size());
    vector<Hash> new_queries;
    vector<size_t> new_indices;
    for (size_t i = 0; i < queries.size(); ++i) {
      uint64_t r = results[i];
      size_t index = indices[i];
      Hash h = queries[i];
      if (r == NOT_FOUND) {
        new_queries.push_back(h);
        new_indices.push_back(index);
      } else {
        all_results[index] = r;
      }
    }
    queries = new_queries;
    indices = new_indices;
    cout << setprecision(4);
    cout << "COVERAGE " << (all_results.size() - queries.size()) * 100. / all_results.size()
        << "%" << endl;
  }
  return all_results;
}

int main_(int argc, char* argv[]) {
  parse_opts(argc, argv);
  utils::Stats stats;

  OpenCLApp cl;
  if (use_opencl)
    cl.print_cl_info();

  RainbowTableParams params;
  params.read_from_disk(table_files[0] + ".params");
  CPUImplementation cpu(params, stats);

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
      for (auto x: lookup_any(params, cl, stats, queries))
        found += x != NOT_FOUND;
    });
  } else {
    std::vector<Hash> queries;
    if (!infile.empty()) {
      cout << "Reading queries" << endl;
      std::ifstream in(infile);
      string s;
      while(in >> s) {
        Hash h;
        if (!parse_hash(s, h)) {
          cout << "Invalid hash `" << s << "' in input file" << endl;
          exit(EXIT_FAILURE);
        }
        queries.push_back(h);
      }
    } else {
      queries.push_back(hash_value);
    }
    vector<uint64_t> results = lookup_any(params, cl, stats, queries);
    assert(results.size() == queries.size());
    for (size_t i = 0; i < results.size(); ++i) {
      print_hash(queries[i]);
      cout << " ";
      auto r = results[i];
      if (r == NOT_FOUND)
        cout << "-" << endl;
      else
        cout << cpu.string_from_index(r) << endl;
    }
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
