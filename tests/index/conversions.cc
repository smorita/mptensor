#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>
#include <mptensor/index.hpp>

namespace {

using SIndex = mptensor::BasicIndex<std::ptrdiff_t>;
using mptensor::internal::UIndex;

TEST(NormalizeAxis, Boundaries) {
  EXPECT_EQ(mptensor::normalize_axis(-3, 3), 0u);
  EXPECT_EQ(mptensor::normalize_axis(-1, 3), 2u);
  EXPECT_EQ(mptensor::normalize_axis(0, 3), 0u);
  EXPECT_EQ(mptensor::normalize_axis(2, 3), 2u);
  EXPECT_THROW(mptensor::normalize_axis(-4, 3), std::out_of_range);
  EXPECT_THROW(mptensor::normalize_axis(3, 3), std::out_of_range);
}

TEST(NormalizeAxis, MessageNamesValueAndRange) {
  try {
    mptensor::normalize_axis(5, 3);
    FAIL() << "no exception";
  } catch (const std::out_of_range& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("axis 5"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[-3, 3)"), std::string::npos) << msg;
  }
}

TEST(NormalizeAxes, ConvertsEachElementAndReportsPosition) {
  EXPECT_EQ(mptensor::normalize_axes(SIndex(-1, 0, -3), 4), UIndex(3, 0, 1));
  EXPECT_EQ(mptensor::normalize_axes(SIndex(), 4).size(), 0u);
  try {
    mptensor::normalize_axes(SIndex(0, 7), 4);
    FAIL() << "no exception";
  } catch (const std::out_of_range& e) {
    EXPECT_NE(std::string(e.what()).find("position 1"), std::string::npos)
        << e.what();
  }
}

TEST(NormalizeIndex, ScalarBoundaries) {
  EXPECT_EQ(mptensor::normalize_index(-5, 5), 0u);
  EXPECT_EQ(mptensor::normalize_index(-1, 5), 4u);
  EXPECT_EQ(mptensor::normalize_index(4, 5), 4u);
  EXPECT_THROW(mptensor::normalize_index(-6, 5), std::out_of_range);
  EXPECT_THROW(mptensor::normalize_index(5, 5), std::out_of_range);
}

TEST(NormalizeIndex, IndexVersion) {
  EXPECT_EQ(mptensor::normalize_index(SIndex(-1, 0), UIndex(3, 4)),
            UIndex(2, 0));
  EXPECT_THROW(mptensor::normalize_index(SIndex(0, 4), UIndex(3, 4)),
               std::out_of_range);
  EXPECT_THROW(mptensor::normalize_index(SIndex(0), UIndex(3, 4)),
               std::invalid_argument);
}

TEST(NormalizeSliceEnd, AcceptsSize) {
  EXPECT_EQ(mptensor::normalize_slice_end(5, 5), 5u);
  EXPECT_EQ(mptensor::normalize_slice_end(-5, 5), 0u);
  EXPECT_EQ(mptensor::normalize_slice_end(-1, 5), 4u);
  EXPECT_THROW(mptensor::normalize_slice_end(6, 5), std::out_of_range);
  EXPECT_THROW(mptensor::normalize_slice_end(-6, 5), std::out_of_range);
}

TEST(ToInternalShape, RejectsNegative) {
  EXPECT_EQ(mptensor::to_internal_shape(SIndex(2, 0, 3)), UIndex(2, 0, 3));
  EXPECT_THROW(mptensor::to_internal_shape(SIndex(2, -1)),
               std::invalid_argument);
}

TEST(ToPublic, RoundTrip) {
  const UIndex u(0, 7, 42);
  EXPECT_EQ(mptensor::to_internal_shape(mptensor::to_public(u)), u);
  UIndex huge;
  huge.push(std::numeric_limits<std::size_t>::max());
  EXPECT_THROW(mptensor::to_public(huge), std::out_of_range);
}

TEST(IdentityAxes, Sequence) {
  EXPECT_EQ(mptensor::internal::identity_axes(3), UIndex(0, 1, 2));
  EXPECT_EQ(mptensor::internal::identity_axes(0).size(), 0u);
}

TEST(NormalizeSliceRange, ScalarSlice) {
  using mptensor::internal::normalize_slice_range;
  using Range = std::pair<std::size_t, std::size_t>;
  EXPECT_EQ(normalize_slice_range(1, -1, 5, 0), Range(1, 4));
  EXPECT_EQ(normalize_slice_range(-2, 5, 5, 0), Range(3, 5));
  EXPECT_THROW(normalize_slice_range(2, -3, 5, 0), std::out_of_range);  // empty
  EXPECT_THROW(normalize_slice_range(3, 3, 5, 0), std::out_of_range);   // empty
  EXPECT_THROW(normalize_slice_range(5, 5, 5, 0), std::out_of_range);   // begin == n
}

TEST(NormalizeSliceRanges, IndexSlice) {
  UIndex b, e;
  mptensor::internal::normalize_slice_ranges(SIndex(0, 1, -2, 3), SIndex(0, -1, 5, 3),
                                           UIndex(5, 5, 5, 5), b, e);
  EXPECT_EQ(b, UIndex(0, 1, 3, 0));  // raw equal -> full axis
  EXPECT_EQ(e, UIndex(5, 4, 5, 5));
  EXPECT_THROW(mptensor::internal::normalize_slice_ranges(
                   SIndex(2), SIndex(-3), UIndex(5), b, e),
               std::out_of_range);
  EXPECT_THROW(mptensor::internal::normalize_slice_ranges(
                   SIndex(0, 0), SIndex(1), UIndex(5, 5), b, e),
               std::invalid_argument);
}

}  // namespace
