#pragma once

#include <array>
#include <string>
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

bool parse_hash(std::string s, Hash& h) {
  if (s.size() != hash_size * 2)
    return false;
  unsigned char byte = 0;
  for (int i = 0; i < s.size(); ++i) {
    byte <<= 4;
    int c = s[i];
    if ('0' <= c && c <= '9')
      byte |= c-'0';
    else if ('a' <= c && c <= 'f')
      byte |= c-'a'+10;
    else if ('A' <= c && c <= 'F')
      byte |= c-'A'+10;
    else
      return false;
    if (i & 1) {
      h[i/2] = byte;
      byte = 0;
    }
  }
  return true;
}
