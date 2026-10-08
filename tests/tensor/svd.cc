#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Svd : public ::testing::Test {};
TYPED_TEST_SUITE(Svd, TensorTypes, TensorTypeNames);

// A[i,j,k,l] = sum_a U[k,i,a] S[a] V[a,j,l]
TYPED_TEST(Svd, Reconstructs) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam U, V;
  std::vector<double> S;
  svd(A, Axes(2, 0), Axes(1, 3), U, S, V);

  U.multiply_vector(S, 2);  // U[k,i,a] <= U[k,i,a] * S[a]
  TypeParam B = tensordot(U, V, Axes(2), Axes(0));  // B[k,i,j,l] = A[i,j,k,l]

  const double error = max_error(B, [&](const Index& b) {
    return func4_1<T>(Index(b[1], b[2], b[0], b[3]), shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
