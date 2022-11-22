cmake_minimum_required(VERSION 3.10)
project(dipcc VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# For pybind11 to work on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/out)

# Declare dependencies
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(pybind11 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIR})
link_directories($ENV{CONDA_PREFIX}/lib/)

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -pg")

# Include torch
# ABI stuff must go before any add_subdirectory.
execute_process(
  COMMAND python -c "import torch; import os; print(int(torch._C._GLIBCXX_USE_CXX11_ABI), end='')"
  OUTPUT_VARIABLE TorchAbi
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${TorchAbi})
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
# set cuda arch list so that the built binary can be run on both pascal and volta
set(TORCH_CUDA_ARCH_LIST "6.0;7.0")
find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIBRARIES "${TORCH_PATH}/lib/libtorch_python.so")

# Compile dipcc
file(GLOB SOURCES "dipcc/cc/*.cc")
add_library(dipcc ${SOURCES})
target_compile_options(dipcc PRIVATE -Wall -Wextra -pedantic -Wno-sign-compare)
target_link_libraries(dipcc glog pthread ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

# Compile python interface
file(GLOB PY_SOURCES "dipcc/pybind/*.cc")
pybind11_add_module(pydipcc ${PY_SOURCES} ${SOURCES})
target_link_libraries(pydipcc PRIVATE glog pthread ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})
IF (DEFINED ENV{PYDIPCC_OUT_DIR})
    set_target_properties(pydipcc PROPERTIES LIBRARY_OUTPUT_DIRECTORY $ENV{PYDIPCC_OUT_DIR})
ELSE()
    set_target_properties(pydipcc PROPERTIES LIBRARY_OUTPUT_DIRECTORY "dipcc/python/")
ENDIF()

# Compile profiling
IF(CMAKE_BUILD_TYPE MATCHES Release)
    add_executable(profile_dipcc dipcc/profiling/profile.cc ${SOURCES})
    target_link_libraries(profile_dipcc dipcc glog pthread)

    add_executable(profile_dipcc_late dipcc/profiling/profile_late.cc ${SOURCES})
    target_link_libraries(profile_dipcc_late dipcc glog pthread)
ENDIF()
