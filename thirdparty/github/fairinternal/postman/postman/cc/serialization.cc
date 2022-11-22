/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#include "postman/serialization.h"

namespace postman {
namespace detail {

// Generic function to fill an ArrayNest proto from a Nest given
// a function for the leaf values.
template <typename T, typename Function>
void fill_proto_from_nest(postman::ArrayNest* nest_pb,
                          const nest::Nest<T>& nest,
                          Function fill_array_proto) {
  using Nest = nest::Nest<T>;
  std::visit(
      nest::overloaded{
          [&](const T& t) { fill_array_proto(nest_pb->mutable_array(), t); },
          [&](const std::vector<Nest>& v) {
            for (const Nest& n : v) {
              postman::ArrayNest* subnest = nest_pb->add_vector();
              fill_proto_from_nest(subnest, n, fill_array_proto);
            }
          },
          [&](const std::map<std::string, Nest>& m) {
            auto* map_pb = nest_pb->mutable_map();
            for (const auto& p : m) {
              postman::ArrayNest& subnest_pb = (*map_pb)[p.first];
              fill_proto_from_nest(&subnest_pb, p.second, fill_array_proto);
            }
          }},
      nest.value);
}

// Generic function to create a nest from an ArrayNest proto given
// a function for the leaf values.
template <typename Function>
nest::Nest<std::invoke_result_t<Function, postman::NDArray*>>
nest_proto_to_nest(postman::ArrayNest* nest_pb, Function array_proto_to_T) {
  using T = std::invoke_result_t<Function, postman::NDArray*>;
  if (nest_pb->has_array()) {
    return nest::Nest<T>(array_proto_to_T(nest_pb->mutable_array()));
  }
  if (nest_pb->map_size() > 0) {
    std::map<std::string, nest::Nest<T>> m;
    for (auto& p : *nest_pb->mutable_map()) {
      m[p.first] = nest_proto_to_nest(&p.second, array_proto_to_T);
    }
    return nest::Nest<T>(std::move(m));
  }

  // Default to (possibly empty) vector.
  std::vector<nest::Nest<T>> v;
  for (int i = 0, length = nest_pb->vector_size(); i < length; ++i) {
    v.push_back(
        nest_proto_to_nest(nest_pb->mutable_vector(i), array_proto_to_T));
  }
  return nest::Nest<T>(std::move(v));
}

// Fill a NDArray proto with data from an aten tensor. One too many copies :/
void fill_array_proto_from_tensor(NDArray* array, const at::Tensor& tensor) {
  if (!tensor.is_contiguous())
    // TODO(heiner): Fix this non-contiguous case.
    throw std::runtime_error("Cannot convert non-contiguous tensor.");
  array->set_scalar_type(static_cast<int8_t>(tensor.scalar_type()));

  at::IntArrayRef shape = tensor.sizes();

  for (size_t i = 0, ndim = shape.size(); i < ndim; ++i) {
    array->add_shape(shape[i]);
  }

  // TODO: Consider set_allocated_data.
  // TODO: Consider [ctype = STRING_VIEW] in proto file.
  array->set_data(tensor.data_ptr(), tensor.nbytes());
}

// Convert a NDArray proto to an aten tensor.
at::Tensor tensor_from_proto(postman::NDArray* array_pb) {
  std::vector<int64_t> shape;
  for (int i = 0, length = array_pb->shape_size(); i < length; ++i) {
    shape.push_back(array_pb->shape(i));
  }
  std::string* data = array_pb->release_data();
  at::ScalarType scalar_type =
      static_cast<at::ScalarType>(array_pb->scalar_type());

  return at::from_blob(
      data->data(), shape,
      /*deleter=*/[data](void*) { delete data; }, scalar_type);
}

// Fill an ArrayNest proto from a TensorNest.
void fill_proto_from_tensornest(postman::ArrayNest* nest_pb,
                                const TensorNest& nest) {
  return fill_proto_from_nest(nest_pb, nest, fill_array_proto_from_tensor);
}

// Create a TensorNest from an ArrayNest proto.
TensorNest nest_proto_to_tensornest(postman::ArrayNest* nest_pb) {
  return nest_proto_to_nest(nest_pb, tensor_from_proto);
}

}  // namespace detail
}  // namespace postman
