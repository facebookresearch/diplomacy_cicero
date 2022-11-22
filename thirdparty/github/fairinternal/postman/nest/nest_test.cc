/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
/*
On mac:
c++ nest_test.cc -lgtest -lgtest_main -std=c++17 -stdlib=libc++
-mmacosx-version-min=10.14 -o nest_test

On linux:
conda install gtest
g++ nest_test.cc -lgtest -lgtest_main -std=c++17 -L$CONDA_PREFIX/lib -pthread -o nest_test
./nest_test
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "nest/nest.h"

namespace {
using namespace nest;

TEST(NestTest, TestConstructDestroy) { Nest<int> n(3); }

TEST(NestTest, TestEmpty) {
  std::vector<Nest<int>> v;
  std::map<std::string, Nest<int>> m;

  Nest<int> n1(v);

  Nest<int> n2(
      std::vector<Nest<int>>({Nest<int>(v), Nest<int>(v), Nest<int>(m)}));

  Nest<int> n3(42);

  Nest<int> n4(
      std::vector<Nest<int>>({Nest<int>(v), Nest<int>(m), Nest<int>(666)}));

  std::vector<Nest<int>> v1({Nest<int>(32)});
  std::map<std::string, Nest<int>> m1;
  m1["integer"] = Nest<int>(69);

  Nest<int> n5(std::vector<Nest<int>>({Nest<int>(v), Nest<int>(v1)}));
  Nest<int> n6(std::vector<Nest<int>>({Nest<int>(v), Nest<int>(m1)}));

  ASSERT_TRUE(n1.empty());
  ASSERT_TRUE(n2.empty());

  ASSERT_FALSE(n3.empty());
  ASSERT_FALSE(n4.empty());

  ASSERT_FALSE(n5.empty());
  ASSERT_FALSE(n6.empty());
}

TEST(NestTest, TestMapDifferentTypes1) {
  Nest<int> n1(1);
  Nest<double> res = n1.map([](int i) -> double { return i / 2.0; });
  ASSERT_EQ(res.front(), 0.5);
}

TEST(NestTest, TestMapDifferentTypes2) {
  Nest<int> n1(1);
  Nest<double> n2(2.0);
  Nest<std::string> res = Nest<int>::map2(
      [](int i, double d) -> std::string {
        return std::to_string(i) + " " + std::to_string(d);
      },
      n1, n2);
  ASSERT_EQ(res.front(), "1 2.000000");
}

TEST(NestTest, TestInvalidZip1) {
  std::vector<Nest<int>> v(
      {Nest<int>(32), Nest<int>(std::vector({Nest<int>(32), Nest<int>(34)}))});
  std::string expected_error = "nests don't match";
  try {
    Nest<int>::zip(v);
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  } catch (const std::invalid_argument& err) {
    EXPECT_EQ(std::string(err.what()), expected_error);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  }
}

TEST(NestTest, TestInvalidZip2) {
  std::vector<Nest<int>> v(
      {Nest<int>(std::vector({Nest<int>(32)})),
       Nest<int>(std::vector({Nest<int>(32), Nest<int>(34)}))});
  std::string expected_error = "Expected vectors of same length but got 1 vs 2";
  try {
    Nest<int>::zip(v);
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  } catch (const std::invalid_argument& err) {
    EXPECT_EQ(std::string(err.what()), expected_error);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  }
}

TEST(NestTest, TestInvalidZip3) {
  std::vector<Nest<int>> v({
      Nest<int>(std::map<std::string, Nest<int>>({{"one", Nest<int>(1)}})),
      Nest<int>(std::map<std::string, Nest<int>>(
          {{"one", Nest<int>(2)}, {"two", Nest<int>(3)}})),
  });
  std::string expected_error = "Expected maps of same length but got 1 vs 2";
  try {
    Nest<int>::zip(v);
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  } catch (const std::invalid_argument& err) {
    EXPECT_EQ(std::string(err.what()), expected_error);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  }
}

TEST(NestTest, TestInvalidZip4) {
  std::vector<Nest<int>> v({
      Nest<int>(std::map<std::string, Nest<int>>({{"one", Nest<int>(1)}})),
      Nest<int>(std::map<std::string, Nest<int>>({{"two", Nest<int>(2)}})),
  });
  std::string expected_error =
      "Expected maps to have same keys but found 'one' vs 'two'";
  try {
    Nest<int>::zip(v);
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  } catch (const std::invalid_argument& err) {
    EXPECT_EQ(std::string(err.what()), expected_error);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  }
}

TEST(NestTest, TestEmptyZip) {
  std::vector<Nest<int>> empty;
  std::string expected_error = "Expected at least one nest.";
  try {
    Nest<int>::zip(empty);
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  } catch (const std::invalid_argument& err) {
    EXPECT_EQ(std::string(err.what()), expected_error);
  } catch (...) {
    FAIL() << "Expected std::invalid_argument: " + expected_error;
  }
}

TEST(NestTest, TestZip1) {
  std::vector<Nest<int>> v(
      {Nest<int>(std::vector({Nest<int>(1), Nest<int>(2)})),
       Nest<int>(std::vector({Nest<int>(3), Nest<int>(4)}))});
  Nest<std::vector<int>> res = Nest<int>::zip(v);
  ASSERT_TRUE(res.is_vector());
  std::vector<Nest<std::vector<int>>> vecs = res.get_vector();
  EXPECT_EQ(vecs.size(), 2);
  EXPECT_THAT(vecs[0].front(), testing::ElementsAre(1, 3));
  EXPECT_THAT(vecs[1].front(), testing::ElementsAre(2, 4));
}

TEST(NestTest, TestZip2) {
  std::vector<Nest<int>> v(
      {Nest<int>(std::vector({Nest<int>(1), Nest<int>(2)})),
       Nest<int>(std::vector({Nest<int>(3), Nest<int>(4)}))});
  Nest<std::vector<int>> zipped = Nest<int>::zip(v);
  Nest<int> res =
      zipped.map([](std::vector<int> ints) { return ints[0]; });
  ASSERT_TRUE(res.is_vector());
  std::vector<Nest<int>> vecs = res.get_vector();
  EXPECT_EQ(vecs.size(), 2);
  EXPECT_EQ(vecs[0].front(), 1);
  EXPECT_EQ(vecs[1].front(), 2);
}

TEST(NestTest, TestMap3) {
  std::vector<Nest<int>> v(
      {Nest<int>(std::vector({Nest<int>(1), Nest<int>(2)})),
       Nest<int>(std::vector({Nest<int>(3), Nest<int>(4)}))});
  Nest<std::vector<int>> zipped = Nest<int>::zip(v);
  Nest<std::vector<int>> res = zipped.map(
      [](std::vector<int> ints) {
        return std::vector<int>({ints[1], ints[0]});
      });
  ASSERT_TRUE(res.is_vector());
  std::vector<Nest<std::vector<int>>> vecs = res.get_vector();
  EXPECT_EQ(vecs.size(), 2);
  EXPECT_THAT(vecs[0].front(), testing::ElementsAre(3, 1));
  EXPECT_THAT(vecs[1].front(), testing::ElementsAre(4, 2));
}

}  // namespace
