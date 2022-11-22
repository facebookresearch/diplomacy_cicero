/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include <future>
#include <random>
#include <string>
#include <vector>

#include "tensor_dict.h"

namespace rela {

static constexpr int kMagicNumber = 575757;

int readInt(FILE *file) {
  int tmp;
  fread(&tmp, sizeof(int), 1, file);
  return tmp;
}

void writeInt(int tmp, FILE *file) { fwrite(&tmp, sizeof(int), 1, file); }

void write(const std::vector<TensorDict> &elements, FILE *f) {
  assert(!elements.empty());

  writeInt(kMagicNumber, f);
  std::vector<std::string> header;
  for (const auto &p : elements[0])
    header.push_back(p.first);

  writeInt(header.size(), f);
  for (size_t i = 0; i < header.size(); ++i) {
    writeInt(header[i].size(), f);
    fwrite(header[i].c_str(), sizeof(char), header[i].size(), f);
  }

  writeInt(elements.size(), f);
  for (const auto &datum : elements) {
    std::vector<torch::Tensor> all_tensors;
    for (const auto &name : header) {
      all_tensors.push_back(datum.at(name));
    }
    std::ostringstream stream;
    torch::save(all_tensors, stream);
    const std::string buffer = stream.str();
    writeInt(buffer.size(), f);
    fwrite(buffer.c_str(), sizeof(char), buffer.size(), f);
  }
}

std::vector<TensorDict> read(FILE *f) {
  std::vector<std::string> header;
  const int magic_number = readInt(f);
  assert(magic_number == kMagicNumber && "bad buffer");
  const int header_size = readInt(f);
  std::vector<char> buffer;
  buffer.reserve(1 << 25);
  for (int i = 0; i < header_size; ++i) {
    int sz = readInt(f);
    buffer.resize(sz);
    fread(buffer.data(), sizeof(char), sz, f);
    header.push_back(std::string(buffer.data(), sz));
  }

  std::vector<TensorDict> elements(readInt(f));
  for (size_t i = 0; i < elements.size(); ++i) {
    auto &tdict = elements[i];
    int sz = readInt(f);
    buffer.resize(sz);
    fread(buffer.data(), sizeof(char), sz, f);
    std::vector<torch::Tensor> tensor_vec;
    torch::load(tensor_vec, static_cast<const char *>(buffer.data()), sz);
    for (size_t j = 0; j < header.size(); ++j) {
      tdict[header[j]] = tensor_vec[j];
    }
  }
  return elements;
}
} // namespace rela