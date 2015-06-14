#include <algorithm>
#include <iomanip>
#include <iostream>
#include "utils.h"

namespace utils {
  void print_progress(std::uint64_t cur, std::uint64_t total) {
    if (cur >= total || cur % std::max(std::uint64_t{1}, total/100) == 0) {
      std::cout << "\rProgress: " << std::setprecision(2) << std::fixed
        << (100. * cur / total) << "%    " << std::flush;
    }
  }

  void finish_progress() {
    std::cout << "\rProgress: 100%    " << std::endl;
  }
}
