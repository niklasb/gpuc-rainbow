#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct RainbowTableParams {
  std::string alphabet;
  std::uint64_t num_strings, chain_len, table_index, num_start_values;

  void save_to_disk(std::string filename) {
    std::ofstream pf(filename);
    pf << alphabet.size() << " ";
    pf.write(alphabet.c_str(), alphabet.size());
    pf << " " << num_strings << " " << chain_len << " " << table_index
      << " " << num_start_values;
  }

  void read_from_disk(std::string filename) {
    std::ifstream pf(filename);
    std::size_t alphabet_size;
    pf >> alphabet_size;
    assert(pf);
    assert(pf.peek() == ' ');
    pf.ignore();
    alphabet.resize(alphabet_size);
    pf.read((char*)&alphabet[0], alphabet_size);
    assert(pf);
    assert(pf.peek() == ' ');
    pf.ignore();
    pf >> num_strings >> chain_len >> table_index >> num_start_values;
    assert(pf);
  }
};

struct RainbowTable {
  using Entry = std::pair<std::uint64_t, std::uint64_t>;
  std::vector<Entry> table;

  void save_to_disk(std::string filename) {
    std::ofstream f(filename);
    f.write((char*)&table[0], table.size() * sizeof table[0]);
  }

  void read_from_disk(std::string filename) {
    std::ifstream f(filename);
    f.seekg(0, std::ios::end);
    std::size_t num = f.tellg() / sizeof(Entry);
    table.resize(num);
    f.seekg(0, std::ios::beg);
    f.read((char*)&table[0], num * sizeof table[0]);
  }
};

const std::uint64_t NOT_FOUND = -1;
