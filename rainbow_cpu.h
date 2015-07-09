#pragma once

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "hash.h"
#include "rainbow_table.h"
#include "utils.h"

template <typename T, typename P>
static void uniqify(std::vector<T>& vec, P eq) {
  vec.resize(unique(std::begin(vec), std::end(vec), eq) - std::begin(vec));
}

template <typename T>
static void uniqify(std::vector<T>& vec) {
  vec.resize(unique(std::begin(vec), std::end(vec)) - std::begin(vec));
}

struct CPUImplementation {
  RainbowTableParams p;
  utils::Stats& stats;

  CPUImplementation(const RainbowTableParams& p, utils::Stats& stats)
    : p(p), stats(stats)
  { }

  void string_from_index(std::uint64_t n, unsigned char* buf, std::uint64_t& len) {
    std::uint64_t base = p.alphabet.size();
    std::uint64_t offset = 0;
    std::uint64_t num = 1;
    len = 0;
    while (offset + num <= n) {
      offset += num;
      num *= base;
      len++;
    }
    n -= offset;
    for (int i = 0; i < len; ++i) {
      buf[len - i - 1] = p.alphabet[n%base];
      n/=base;
    }
  }

  std::string string_from_index(std::uint64_t n) {
    // TODO overflow?
    unsigned char buf[20];
    uint64_t len;
    string_from_index(n, buf, len);
    return std::string((char*)buf, (char*)buf + len);
  }

  std::uint64_t reduce(const Hash& h, std::uint64_t round) {
    std::uint64_t x = 0;
    for (std::uint64_t i = 0; i < h.size(); ++i) {
      x = ((std::uint64_t)x * 0x100 + h[i]) % p.num_strings;
    }
    return (x + round + MAGIC_MANGLER*p.table_index) % p.num_strings;
  }

  void compute_hash(std::uint64_t x, Hash& h) {
    unsigned char buf[16];
    std::uint64_t len;
    string_from_index(x, buf, len);
    ::compute_hash(buf, len, h);
  }

  std::pair<std::uint64_t,Hash> construct_chain(Hash h, std::uint64_t start_iteration, std::uint64_t end_iteration) {
    assert(start_iteration < end_iteration);
    std::uint64_t x;
    auto t = p.chain_len;
    for (std::uint64_t i = start_iteration; i < end_iteration; ++i) {
      x = reduce(h, i);
      compute_hash(x, h);
    }
    return {x, h};
  }

  std::pair<std::uint64_t,Hash> construct_chain(std::uint64_t x, std::uint64_t start_iteration, std::uint64_t end_iteration) {
    Hash h;
    compute_hash(x, h);
    if (start_iteration == end_iteration)
      return {x, h};
    return construct_chain(h, start_iteration, end_iteration);
  }

  void sort_and_uniqify(RainbowTable& rt) {
    std::sort(std::begin(rt.table), std::end(rt.table));
    uniqify(rt.table, [&](const RainbowTable::Entry& a, const RainbowTable::Entry& b) {
      return a.first == b.first;
    });
  }

  void build(RainbowTable& rt) {
    std::uint64_t offset = p.table_index * p.num_start_values;
    if (offset + p.num_start_values > p.num_strings) {
      std::cerr << "ERROR: Cannot generate table with this index" << std::endl;
      exit(1);
    }
    rt.table.resize(p.num_start_values);
    utils::Progress progress(p.num_start_values);
    stats.add_timing("time_generate", [&]() {
      for (std::uint64_t i = 0; i < p.num_start_values; ++i) {
        progress.report(i);
        std::uint64_t start = offset + i;
        rt.table[i] = {construct_chain(start, 0, p.chain_len).first, start};
      }
    });
    progress.finish();
    stats.add_timing("time_sort", [&]() {
      sort_and_uniqify(rt);
    });
  }

  std::uint64_t lookup_single(const RainbowTable& rt, const Hash& h) {
    for (int i = 0; i < p.chain_len; ++i) {
      std::uint64_t endpoint = construct_chain(h, i, p.chain_len).first;
      auto l = std::lower_bound(std::begin(rt.table), std::end(rt.table),
          std::make_pair(endpoint, std::uint64_t{0}));
      auto r = std::upper_bound(std::begin(rt.table), std::end(rt.table),
          std::make_pair(endpoint, std::numeric_limits<std::uint64_t>::max()));
      for (auto it = l; it != r; ++it) {
        std::uint64_t start = it->second;
        auto candidate = construct_chain(start, 0, i);
        if (candidate.second == h)
          return candidate.first;
      }
    }
    return NOT_FOUND;
  }

  std::vector<std::uint64_t> lookup(
      const RainbowTable& table,
      const std::vector<Hash>& queries)
  {
    std::vector<std::uint64_t> res;
    utils::Progress prog(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
      prog.report(i);
      res.push_back(lookup_single(table, queries[i]));
    }
    prog.finish();
    return res;
  }
};
