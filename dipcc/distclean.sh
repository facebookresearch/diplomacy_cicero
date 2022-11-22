#!/bin/bash -e
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

pushd $(dirname $0)

# Removes all cmake cached stuff
rm -rf build

# Also remove some lingering files that might be around from prior to building in build
rm -f CMakeCache.txt
rm -f cmake_install.cmake
rm -f MakeFile
rm -rf CMakeFiles/

popd >/dev/null
