#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Kron : public ::testing::Test {};
TYPED_TEST_SUITE(Kron, TensorTypes, TensorTypeNames);

// C = kron(A, B): C[i,j] = A[i % nA0, j % nA1] * B[i / nA0, j / nA1]
TYPED_TEST(Kron, KroneckerProduct) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1));
  TypeParam B(Shape(kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  const Shape shape_B = B.shape();
  fill_tensor(A, func2_1<T>);
  fill_tensor(B, func2_1<T>);

  TypeParam C = kron(A, B);

  const double error = max_error(C, [&](const Index& c) {
    return func2_1<T>(Index(c[0] % shape_A[0], c[1] % shape_A[1]), shape_A) *
           func2_1<T>(Index(c[0] / shape_A[0], c[1] / shape_A[1]), shape_B);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
