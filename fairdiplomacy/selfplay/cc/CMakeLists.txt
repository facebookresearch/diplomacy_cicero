cmake_minimum_required(VERSION 3.10)
project(postman VERSION 0.0.1 LANGUAGES CXX)

if(NOT DEFINED POSTMAN_DIR)
  message(FATAL_ERROR "Must set 'POSTMAN_DIR' variable")
endif(NOT DEFINED POSTMAN_DIR)

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
  ${POSTMAN_DIR}/third_party/pybind11 third_party/pybind11
)

#### GTest LIB  #####
add_subdirectory(
  ${POSTMAN_DIR}/third_party/grpc/third_party/googletest/ third_party/googletest
)

#### TORCH LIB  #####
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
# set cuda arch list so that the built binary can be run on both pascal and volta
set(TORCH_CUDA_ARCH_LIST "6.0;7.0")
find_package(Torch REQUIRED)

# Temp fix for PyTorch 1.5.
set(TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so")

### Nest ###
add_library(nest INTERFACE)
target_include_directories(nest INTERFACE ${POSTMAN_DIR}/nest/nest)

### Python module ###
pybind11_add_module(
  rela
  pybind.cc
)
target_include_directories(rela PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(rela
  PUBLIC
  ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} nest
)
set_target_properties(rela PROPERTIES CXX_STANDARD 17)


include(GoogleTest)
enable_testing()

add_executable(prioritized_replay_test prioritized_replay_test.cc)
target_include_directories(prioritized_replay_test PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(prioritized_replay_test
  PUBLIC
  gtest_main
  ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES} nest
)
set_target_properties(prioritized_replay_test PROPERTIES CXX_STANDARD 17)
add_test(NAME prioritized_replay_test COMMAND prioritized_replay_test)