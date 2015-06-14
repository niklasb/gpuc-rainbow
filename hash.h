#ifndef _HASH_H
#define _HASH_H

#include <array>
#include <cassert>
#include "md5.h"

const size_t hash_size = 16;

using Hash = std::array<unsigned char, hash_size>;

void compute_hash(unsigned char buf[], size_t buf_len, Hash& h) {
  assert(buf_len <= 0xffffffff);
  md5_hash(buf, (uint32_t)buf_len, (uint32_t*)&h[0]);
}

void print_hash(const Hash& h) {
  for (size_t i = 0; i < hash_size; ++i)
    printf("%02x", h[i]);
}

#endif
