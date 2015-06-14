#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <unordered_set>

#include "io.h"
#include "timing.h"
#if !HAVE_OPENCL
#  error "Need OpenCL"
#endif
#include "opencl.h"
#include "hash.h"
#include "rainbow_table.h"
#include "rainbow_cpu.h"
#include "utils.h"

using namespace std;

void usage(char *argv0) {
  cerr << "Usage: " << argv0 << " [FLAGS] string_len alphabet outfile" << endl
       << endl
       << "string_len" << endl
       << "  an integer representing the maximum length of strings covered by "
       << "  generated the rainbow table" << endl
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
       << "  -c       Check coverage of generated table" << endl
       << "  -s INT   Integer value specifying the number of samples to use" << endl
       << "           for coverage analysis. 0 means coverage is measured" << endl
       << "           exactly by reconstructing all chains in memory (needs more " << endl
       << "           memory, but much faster)" << endl
       << "  -i INT   Table index in case multiple tables are generated" << endl
       << "  -r       Specify random seed (defaults to constant value)" << endl
       << endl
       << "EXAMPLES" << endl
       << "  " << argv0 << " -t 7 abcdefghijklmnopqrstuvwxyz0123456789" << endl;
  exit(EXIT_FAILURE);
}


uint64_t max_string_len;
bool use_opencl = false;
bool check_coverage = false;
double alpha = 0.01;
uint64_t samples = 0;
uint64_t seed = 0;
RainbowTableParams params;
string outfile;

const double eps = 1e-9;

void parse_opts(int argc, char *argv[]) {
  int pos = 0;
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
    if (o == "-c") {
      check_coverage = true;
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

int main(int argc, char* argv[]) {
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
  if (check_coverage)
    cout << "  cov samples = " << samples << endl;

  params.num_start_values =
    min((uint64_t)(alpha * params.num_strings), params.num_strings);

  cout << "Computing " << params.num_start_values << " chains ("
      << (100.*params.num_start_values / params.num_strings)
      << "% of search space)" << endl;

  RainbowTable rt;
  CPUImplementation cpu(params);
  if (use_opencl) {
    cerr << "OpenCL not implemented!" << endl;
    exit(1);
  } else {
    cpu.build(rt);
  }

  cout << "Result: " << rt.table.size() << " unique chains (~"
       << setprecision(2) << fixed << (100. * rt.table.size() / params.num_strings)
       << "% of search space)" << endl;

  if (check_coverage) {
    if (samples == 0) {
      cout << "Measuring exact coverage" << endl;
      unordered_set<uint64_t> covered;
      for (size_t i = 0; i < rt.table.size(); ++i) {
        auto& entry = rt.table[i];
        utils::print_progress(i, rt.table.size());
        //cout << "start=" << entry.second << endl;
        /*
        construct_chain(entry.second, 0, [&](uint64_t x, const Hash& h) {
          cout << x << " "; print_hash(h); cout << endl;
        });
        */
        uint64_t endpoint = cpu.construct_chain(entry.second, 0, [&](uint64_t x, const Hash& h) {
          covered.insert(x);
          //assert(is_covered(x, rainbow_table));

          /*
          if (!is_covered(x, rainbow_table)) {
            Hash h;
            compute_hash(x, h);
            uint64_t res;
            //cout << "lookup "; print_hash(h); cout << endl;
            //cout << "lookup result=" << lookup(h, rainbow_table, res, true) << endl;
            cout << res<< endl;
          }
          */
        });
        assert(endpoint == entry.first);
      }
      utils::finish_progress();
      cout << "Coverage: " << setprecision(4)
          << fixed << (100.*covered.size()/params.num_strings) << "%" << endl;
    } else {
      uint64_t found = 0;
      std::mt19937 gen(seed);
      cout << "Estimating coverage using " << samples << " samples" << endl;
      for (uint64_t i = 0; i < samples; ++i) {
        utils::print_progress(i, samples);
        uint64_t sample = (((uint64_t)gen() << 32) | gen()) % params.num_strings;
        if (cpu.is_covered(sample, rt))
          found++;
      }
      utils::finish_progress();
      cout << "Coverage: " << setprecision(4) << fixed
          << (100.*found/samples) << "%" << endl;
    }
  }
}
