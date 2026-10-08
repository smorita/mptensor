#include <cmath>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Trace : public ::testing::Test {};
TYPED_TEST_SUITE(Trace, TensorTypes, TensorTypeNames);

// trace(A, Axes(0,1), Axes(3,2)) = sum_{i,j} A[i,j,j,i]
TYPED_TEST(Trace, FullTrace) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1;
  TypeParam A(Shape(N0, N1, N1, N0));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  T exact = 0.0;
  for (int i = 0; i < N0; ++i) {
    for (int j = 0; j < N1; ++j) {
      exact += func4_1<T>(Index(i, j, j, i), shape_A);
    }
  }

  {
    SCOPED_TRACE("trace of rank-4 tensor");
    const T result = trace(A, Axes(0, 1), Axes(3, 2));
    EXPECT_LT(std::abs(result - exact) / std::abs(exact), kEps);
  }
  {
    SCOPED_TRACE("trace of reshaped matrix");
    TypeParam M = transpose(A, Axes(0, 1, 3, 2));
    const Shape s = M.shape();
    M = reshape(M, Shape(s[0] * s[1], s[2] * s[3]));
    const T result = trace(M);
    EXPECT_LT(std::abs(result - exact) / std::abs(exact), kEps);
  }
}

template <typename TensorType>
class Trace2 : public ::testing::Test {};
TYPED_TEST_SUITE(Trace2, TensorTypes, TensorTypeNames);

// trace(A, B, Axes(0,3,2,1), Axes(3,2,0,1)) = sum_{i,j,k,l} A[i,j,k,l] B[k,j,l,i]
TYPED_TEST(Trace2, FullContraction) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2, N3 = kL + 3;
  TypeParam A(Shape(N0, N1, N2, N3));
  TypeParam B(Shape(N2, N1, N3, N0));
  const Shape shape_A = A.shape();
  const Shape shape_B = B.shape();
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);

  const T result = trace(A, B, Axes(0, 3, 2, 1), Axes(3, 2, 0, 1));

  T exact = 0.0;
  for (int i = 0; i < N0; ++i) {
    for (int j = 0; j < N1; ++j) {
      for (int k = 0; k < N2; ++k) {
        for (int l = 0; l < N3; ++l) {
          exact += func4_1<T>(Index(i, j, k, l), shape_A) *
                   func4_2<T>(Index(k, j, l, i), shape_B);
        }
      }
    }
  }
  EXPECT_LT(std::abs(result - exact) / std::abs(exact), kEps);
}

}  // namespace
}  // namespace mptensor_test
