#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include <mptensor/index.hpp>

namespace {

using SIndex = mptensor::BasicIndex<std::ptrdiff_t>;
using UIndex = mptensor::BasicIndex<std::size_t>;

TEST(BasicIndex, VariadicAcceptsMixedIntegerTypes) {
  const int a = 1;
  const long b = -2;
  const std::size_t c = 3;
  const std::ptrdiff_t d = -4;
  const SIndex idx(a, b, c, d);
  ASSERT_EQ(idx.size(), 4u);
  EXPECT_EQ(idx[0], 1);
  EXPECT_EQ(idx[1], -2);
  EXPECT_EQ(idx[2], 3);
  EXPECT_EQ(idx[3], -4);
}

TEST(BasicIndex, ParenthesesAndBracesAgree) {
  EXPECT_EQ(SIndex(0, 1, -1), (SIndex{0, 1, -1}));
  EXPECT_EQ(UIndex(0, 1, 2), (UIndex{0, 1, 2}));
}

TEST(BasicIndex, SingleArgumentIsLengthOne) {
  const SIndex a(3);
  const SIndex b{3};
  ASSERT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0], 3);
  EXPECT_EQ(a, b);
}

TEST(BasicIndex, ImplicitConversionFromInteger) {
  const SIndex a = 2;
  ASSERT_EQ(a.size(), 1u);
  EXPECT_EQ(a[0], 2);
}

TEST(BasicIndex, MoreThanSixteenArguments) {
  const SIndex idx(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                   17);
  ASSERT_EQ(idx.size(), 18u);
  EXPECT_EQ(idx[17], 17);
}

TEST(BasicIndex, NegativeValueIntoUnsignedThrows) {
  EXPECT_THROW(UIndex(0, -1), std::out_of_range);
}

TEST(BasicIndex, HugeUnsignedValueIntoSignedThrows) {
  const std::size_t huge = std::numeric_limits<std::size_t>::max();
  EXPECT_THROW((void)SIndex(huge), std::out_of_range);
  EXPECT_THROW(SIndex(0, huge), std::out_of_range);
}

TEST(BasicIndex, FromVector) {
  const std::vector<std::ptrdiff_t> v{5, -6};
  const SIndex idx(v);
  ASSERT_EQ(idx.size(), 2u);
  EXPECT_EQ(idx[1], -6);
}

TEST(BasicIndex, PushResizeAssign) {
  UIndex idx;
  idx.push(4);
  idx.push(7);
  EXPECT_EQ(idx, UIndex(4, 7));
  idx.resize(3);
  EXPECT_EQ(idx, UIndex(4, 7, 0));
  const std::size_t raw[] = {9, 8};
  idx.assign(2, raw);
  EXPECT_EQ(idx, UIndex(9, 8));
}

TEST(BasicIndex, SortInverseConcatenate) {
  UIndex p(2, 0, 1);
  EXPECT_EQ(p.inverse(), UIndex(1, 2, 0));
  p.sort();
  EXPECT_EQ(p, UIndex(0, 1, 2));
  EXPECT_EQ(UIndex(0, 1) + UIndex(2, 3), UIndex(0, 1, 2, 3));
  SIndex s(-1, 0);
  s += SIndex(5);
  EXPECT_EQ(s, SIndex(-1, 0, 5));
}

TEST(BasicIndex, StreamFormat) {
  std::ostringstream os;
  os << SIndex(0, -1, 2) << " " << UIndex();
  EXPECT_EQ(os.str(), "[0, -1, 2] []");
}

TEST(BasicIndex, Range) {
  EXPECT_EQ(mptensor::range(3), mptensor::Index(0, 1, 2));
  EXPECT_EQ(mptensor::range(2, 4), mptensor::Index(2, 3));
  EXPECT_EQ(mptensor::range(2, 2).size(), 0u);
}

}  // namespace
