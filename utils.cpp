#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
#include "utils.h"

using namespace std;

namespace utils {

double get_time() {
  struct timezone tzp;
  timeval now;
  gettimeofday(&now, &tzp);
  return now.tv_sec + now.tv_usec / 1000000.;
}

Progress::Progress(uint64_t total, double interval)
  : total(total), interval(interval), last(-(1./0.))
{
  report(0);
}

void Progress::report(uint64_t cur) {
  double t = get_time();
  if (t < last + interval)
    return;
  last = t;
  cout << "\rProgress: " << setprecision(2) << fixed
      << (100. * cur / total) << "%    " << flush;
}

void Progress::finish() {
  cout << "\rProgress: 100%    " << endl;
}

}
