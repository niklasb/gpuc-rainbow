#ifndef _UTILS_H
#define _UTILS_H

#include <cstdint>

namespace utils {
  void print_progress(std::uint64_t cur, std::uint64_t total);
  void finish_progress();
}

#endif
