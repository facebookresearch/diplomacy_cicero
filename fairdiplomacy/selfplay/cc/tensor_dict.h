/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once
#include <ostream>

#include <torch/extension.h>
#include <unordered_map>

namespace rela {

using TensorDict = std::unordered_map<std::string, torch::Tensor>;

namespace tensor_dict {

inline void compareShape(const TensorDict &src, const TensorDict &dest) {
  if (src.size() != dest.size()) {
    std::cout << "src.size()[" << src.size() << "] != dest.size()["
              << dest.size() << "]" << std::endl;
    std::cout << "src keys: ";
    for (const auto &p : src)
      std::cout << p.first << " ";
    std::cout << "dest keys: ";
    for (const auto &p : dest)
      std::cout << p.first << " ";
    std::cout << std::endl;
    assert(false);
  }

  for (const auto &name2tensor : src) {
    const auto &name = name2tensor.first;
    const auto &srcTensor = name2tensor.second;
    // std::cout << "in copy: trying to get: " << name << std::endl;
    // std::cout << "dest map keys" << std::endl;
    // printMapKey(dest);
    const auto &destTensor = dest.at(name);
    // if (destTensor.sizes() != srcTensor.sizes()) {
    //   std::cout << "copy size-mismatch: "
    //             << destTensor.sizes() << ", " << srcTensor.sizes() <<
    //             std::endl;
    // }
    if (destTensor.sizes() != srcTensor.sizes()) {
      std::cout << name << ", dstSize: " << destTensor.sizes()
                << ", srcSize: " << srcTensor.sizes() << std::endl;
      assert(false);
    }

    // if (destTensor.dtype() != srcTensor.dtype()) {
    //   std::cout << name << ", dstType: " << destTensor.dtype()
    //             << ", srcType: " << srcTensor.dtype() << std::endl;
    //   assert(false);
    // }
  }
}

inline void copy(const TensorDict &src, TensorDict &dest) {
  compareShape(src, dest);
  for (const auto &name2tensor : src) {
    const auto &name = name2tensor.first;
    const auto &srcTensor = name2tensor.second;
    // std::cout << "in copy: trying to get: " << name << std::endl;
    // std::cout << "dest map keys" << std::endl;
    // printMapKey(dest);
    auto &destTensor = dest.at(name);
    // if (destTensor.sizes() != srcTensor.sizes()) {
    //   std::cout << "copy size-mismatch: "
    //             << destTensor.sizes() << ", " << srcTensor.sizes() <<
    //             std::endl;
    // }
    destTensor.copy_(srcTensor);
  }
}

// // TODO: maybe merge these two functions?
// inline void copyTensors(
//     const std::unordered_map<std::string, torch::Tensor>& src,
//     std::unordered_map<std::string, torch::Tensor>& dest,
//     std::vector<int64_t>& index) {
//   assert(src.size() == dest.size());
//   assert(!index.empty());
//   torch::Tensor indexTensor =
//       torch::from_blob(index.data(), {(int64_t)index.size()}, torch::kInt64);

//   for (const auto& name2tensor : src) {
//     const auto& name = name2tensor.first;
//     const auto& srcTensor = name2tensor.second;
//     auto& destTensor = dest.at(name);
//     // assert(destTensor.sizes() == srcTensor.sizes());
//     assert(destTensor.dtype() == srcTensor.dtype());
//     assert(indexTensor.size(0) == srcTensor.size(0));
//     destTensor.index_copy_(0, indexTensor, srcTensor);
//   }
// }

inline void copy(const TensorDict &src, TensorDict &dest,
                 const torch::Tensor &index) {
  assert(src.size() == dest.size());
  assert(index.size(0) > 0);
  for (const auto &name2tensor : src) {
    const auto &name = name2tensor.first;
    const auto &srcTensor = name2tensor.second;
    auto &destTensor = dest.at(name);
    assert(destTensor.dtype() == srcTensor.dtype());
    assert(index.size(0) == srcTensor.size(0));
    destTensor.index_copy_(0, index, srcTensor);
  }
}

inline bool eq(const TensorDict &d0, const TensorDict &d1) {
  if (d0.size() != d1.size()) {
    return false;
  }

  for (const auto &name2tensor : d0) {
    auto key = name2tensor.first;
    if ((d1.at(key) != name2tensor.second).all().item<bool>()) {
      return false;
    }
  }
  return true;
}

/*
 * indexes into a TensorDict
 */
inline TensorDict index(const TensorDict &batch, size_t i) {
  TensorDict result;
  for (const auto &name2tensor : batch) {
    result.insert({name2tensor.first, name2tensor.second[i]});
  }
  return result;
}

inline TensorDict narrow(const TensorDict &batch, size_t dim, size_t i,
                         size_t len, bool squeeze) {
  TensorDict result;
  for (auto &name2tensor : batch) {
    auto t = name2tensor.second.narrow(dim, i, len);
    if (squeeze) {
      assert(len == 1);
      t = t.squeeze(dim);
    }
    result.insert({name2tensor.first, std::move(t)});
  }
  return result;
}

inline TensorDict clone(const TensorDict &input) {
  TensorDict output;
  for (auto &name2tensor : input) {
    output.insert({name2tensor.first, name2tensor.second.clone()});
  }
  return output;
}

inline TensorDict zerosLike(const TensorDict &input) {
  TensorDict output;
  for (auto &name2tensor : input) {
    output.insert({name2tensor.first, torch::zeros_like(name2tensor.second)});
  }
  return output;
}

template <typename Func> inline void for_each(const TensorDict &dict, Func f) {
  for (const auto &name2tensor : dict) {
    f(name2tensor.second);
  }
}

// TODO: rewrite the above functions with this template
template <typename Func> inline TensorDict apply(TensorDict &dict, Func f) {
  TensorDict output;
  for (const auto &name2tensor : dict) {
    auto tensor = f(name2tensor.second);
    output.insert({name2tensor.first, tensor});
  }
  return output;
}

inline TensorDict stack(const std::vector<TensorDict> &vec, int stackdim) {
  assert(vec.size() >= 1);
  TensorDict ret;
  for (auto &name2tensor : vec[0]) {
    std::vector<torch::Tensor> buffer(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
      buffer[i] = vec[i].at(name2tensor.first);
    }
    ret[name2tensor.first] = torch::stack(buffer, stackdim);
  }
  return ret;
}

inline TensorDict fromIValue(const torch::jit::IValue &value,
                             torch::DeviceType device, bool detach) {
  std::unordered_map<std::string, torch::Tensor> map;
  auto dict = value.toGenericDict();
  // auto ivalMap = dict->elements();
  for (auto &name2tensor : dict) {
    auto name = name2tensor.key().toString();
    torch::Tensor tensor = name2tensor.value().toTensor();
    tensor = tensor.to(device);
    if (detach) {
      tensor = tensor.detach();
    }
    map.insert({name->string(), tensor});
  }
  return map;
}

// TODO: this may be simplified with constructor in the future version
inline torch::jit::IValue toIValue(const TensorDict &tensorDict,
                                   const torch::Device &device) {
  torch::Dict<std::string, torch::Tensor> dict;
  for (const auto &name2tensor : tensorDict) {
    dict.insert(name2tensor.first, name2tensor.second.to(device));
  }
  return torch::jit::IValue(dict);
}
} // namespace tensor_dict
} // namespace rela

inline std::ostream &operator<<(std::ostream &stream,
                                const rela::TensorDict &dict) {
  for (auto name2tensor : dict) {
    stream << name2tensor.first << name2tensor.second.sizes() << " ";
  }
  return stream;
}