#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Reshape : public ::testing::Test {};
TYPED_TEST_SUITE(Reshape, TensorTypes, TensorTypeNames);

// B = reshape(A, (N0*N1, N2*N3)); the first index runs fastest.
TYPED_TEST(Reshape, MergesAxes) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2, N3 = kL + 3;
  TypeParam A(Shape(N0, N1, N2, N3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam B = reshape(A, Shape(N0 * N1, N2 * N3));

  const double error = max_error(B, [&](const Index& b) {
    return func4_1<T>(Index(b[0] % N0, b[0] / N0, b[1] % N2, b[1] / N2),
                      shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
