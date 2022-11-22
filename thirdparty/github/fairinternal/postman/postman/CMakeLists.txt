cmake_minimum_required(VERSION 3.10)
project(postman VERSION 0.2.2 LANGUAGES CXX)

###########################################
# Global flags

# For pybind11 to work on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#############################################
### LIBS ####################################
#############################################
# Declare dependencies
find_package(Threads REQUIRED)
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

# Make sure we are compatible with torch ABI. Must go before any add_subdirectory.
execute_process(
  COMMAND python -c "import torch; import os; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
  OUTPUT_VARIABLE TorchAbi
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TorchAbi})

#### Pybind11 LIB  #####
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 third_party/pybind11
)

#### GRPC LIB  #####
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/grpc
  ${CMAKE_CURRENT_BINARY_DIR}/grpc EXCLUDE_FROM_ALL
)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
set(_GRPC_GRPCPP_UNSECURE grpc++_unsecure)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

#### TORCH LIB  #####
if (NOT TARGET torch_library)
  execute_process(
    COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
    OUTPUT_VARIABLE TORCH_PATH
  )
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
  # set cuda arch list so that the built binary can be run on both pascal and volta
  set(TORCH_CUDA_ARCH_LIST "6.0;7.0")
  find_package(Torch REQUIRED)

  # Fix for PyTorch 1.5, see https://github.com/pytorch/pytorch/issues/38122
  # TODO: Remove this fix once PyTorch 1.6 is released.
  set(TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so")
endif()

# Make more dependencies available (but don't build by default)
add_subdirectory(${CMAKE_SOURCE_DIR}/../third_party/grpc/third_party/googletest ${CMAKE_CURRENT_BINARY_DIR}/googletest EXCLUDE_FROM_ALL)

#############################################
### POSTMAN ACTUAL ##########################
#############################################

### POSTMAN PROTO FILES ###
# Proto file
get_filename_component(hw_proto "proto/rpc.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(gen_proto_dir "${CMAKE_CURRENT_BINARY_DIR}/proto")
set(hw_proto_srcs "${gen_proto_dir}/rpc.pb.cc")
set(hw_grpc_srcs "${gen_proto_dir}/rpc.grpc.pb.cc")
file(MAKE_DIRECTORY ${gen_proto_dir})
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${gen_proto_dir}"
        --cpp_out "${gen_proto_dir}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

### Nest ###
add_library(nest INTERFACE)
target_include_directories(nest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../nest/nest)

### Python-free Postman library ###
add_library(postman STATIC
  cc/server.cc cc/client.cc cc/asyncclient.cc cc/serialization.cc
  "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}")
target_include_directories(postman PUBLIC ${gen_proto_dir})
target_include_directories(postman PUBLIC include)
target_link_libraries(postman PUBLIC grpc libprotobuf grpc++ ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} nest)
set_target_properties(postman PROPERTIES CXX_STANDARD 17)

### Python module ###
pybind11_add_module(
  rpc
  cc/rpc_python.cc
  "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}")
target_include_directories(rpc PRIVATE ${PYTHON_INCLUDE_DIRS} ${gen_proto_dir})
target_link_libraries(rpc PUBLIC postman)
set_target_properties(rpc PROPERTIES CXX_STANDARD 17)

enable_testing()

#if(BUILD_TESTING)
add_subdirectory(tests)
#endif()
