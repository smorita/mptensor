#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Transpose : public ::testing::Test {};
TYPED_TEST_SUITE(Transpose, TensorTypes, TensorTypeNames);

// B = transpose(A, (2,0,3,1)), so B[a,b,c,d] = A[b,d,a,c].
TYPED_TEST(Transpose, PermutesAxes) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam B = transpose(A, Axes(2, 0, 3, 1));

  const double error = max_error(B, [&](const Index& b) {
    return func4_1<T>(Index(b[1], b[3], b[0], b[2]), shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
