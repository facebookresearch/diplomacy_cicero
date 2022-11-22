#!/bin/bash -e
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

MODE=${MODE:-Release}

pushd $(dirname $0)

# Remove some lingering files that might be around from prior to building in build
rm -f CMakeCache.txt
rm -f cmake_install.cmake
rm -f MakeFile
rm -rf CMakeFiles/

mkdir -p build
pushd build
cmake -DCMAKE_BUILD_TYPE=$MODE .. && make -j ${N_DIPCC_JOBS:-}
popd >/dev/null

popd >/dev/null
