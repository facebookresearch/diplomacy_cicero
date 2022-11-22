# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Defines `gtest_discover_tests()`.
include(GoogleTest)
enable_testing()

add_executable(test_postman cc/simple_test.cc cc/async_client_test.cc)
target_link_libraries(test_postman postman gtest_main)
add_test(NAME test_postman COMMAND test_postman)
set_target_properties(test_postman PROPERTIES CXX_STANDARD 17)
gtest_add_tests(TARGET test_postman AUTO)
