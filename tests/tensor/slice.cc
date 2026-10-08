#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Slice : public ::testing::Test {
 protected:
  // A.shape = (L, max(L+1,6), L+2, max(L+3,5)) filled with func4_1.
  static TensorType make_source() {
    using T = typename TensorType::value_type;
    TensorType A(Shape(kL, std::max(kL + 1, 6), kL + 2, std::max(kL + 3, 5)));
    fill_tensor(A, func4_1<T>);
    return A;
  }
};
TYPED_TEST_SUITE(Slice, TensorTypes, TensorTypeNames);

// B = A[:, 3:6, :, :]
TYPED_TEST(Slice, ScalarRange) {
  using T = typename TypeParam::value_type;
  const TypeParam A = TestFixture::make_source();
  const Shape shape_A = A.shape();

  TypeParam B = slice(A, 1, 3, 6);

  const double error = max_error(B, [&](Index b) {
    b[1] += 3;
    return func4_1<T>(b, shape_A);
  });
  EXPECT_LT(error, kEps);
}

// C = A[:, 2:4, :, 1:5]; begin == end (0, 0) keeps the whole axis.
TYPED_TEST(Slice, IndexRange) {
  using T = typename TypeParam::value_type;
  const TypeParam A = TestFixture::make_source();
  const Shape shape_A = A.shape();

  TypeParam C = slice(A, Index(0, 2, 0, 1), Index(0, 4, 0, 5));

  const double error = max_error(C, [&](Index c) {
    c[1] += 2;
    c[3] += 1;
    return func4_1<T>(c, shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
