#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Tensordot : public ::testing::Test {};
TYPED_TEST_SUITE(Tensordot, TensorTypes, TensorTypeNames);

// C = tensordot(A, B, axes=([1,3],[2,0])):
// C[a,b,c,d] = sum_{m,k} A[a,m,b,k] B[k,c,m,d]
TYPED_TEST(Tensordot, ContractsTwoAxes) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2, N3 = kL + 3;
  const int M = kL + 4, K = kL + 5;
  TypeParam A(Shape(N0, M, N1, K));
  TypeParam B(Shape(K, N2, M, N3));
  const Shape shape_A = A.shape();
  const Shape shape_B = B.shape();
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);

  TypeParam C = tensordot(A, B, Axes(1, 3), Axes(2, 0));

  const double error = max_error(C, [&](const Index& c) {
    T exact = 0.0;
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        exact += func4_1<T>(Index(c[0], m, c[1], k), shape_A) *
                 func4_2<T>(Index(k, c[2], m, c[3]), shape_B);
      }
    }
    return exact;
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
