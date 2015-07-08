#pragma once

#include <map>
#include <string>

namespace utils {

double get_time();

struct Stats {
  std::map<std::string, double> stats;
  void add(std::string name, double x) {
    stats[name] += x;
  }
  template <typename F>
  void add_timing(std::string name, F f) {
    double t0 = get_time();
    f();
    add(name, get_time() - t0);
  }
};

struct Progress {
  static constexpr double DEFAULT_INTERVAL = 0.5;
  uint64_t total;
  double interval, last;
  Progress(std::uint64_t total, double interval=DEFAULT_INTERVAL);
  void report(std::uint64_t cur);
  void finish();
};

template <typename T>
T round_to_multiple(T a, T b) {
  return (a + b - 1) / b * b;
}

}
