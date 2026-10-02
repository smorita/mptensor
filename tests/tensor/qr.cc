#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Qr : public ::testing::Test {};
TYPED_TEST_SUITE(Qr, TensorTypes, TensorTypeNames);

// A[i,j,k,l] = sum_a Q[k,i,a] R[a,j,l]
TYPED_TEST(Qr, Reconstructs) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam Q, R;
  qr(A, Axes(2, 0), Axes(1, 3), Q, R);

  TypeParam B =
      transpose(tensordot(Q, R, Axes(2), Axes(0)), Axes(1, 2, 0, 3));

  const double error = max_error(
      B, [&](const Index& b) { return func4_1<T>(b, shape_A); });
  EXPECT_LT(error, kEps);
}

template <typename TensorType>
class QrRank2 : public ::testing::Test {};
TYPED_TEST_SUITE(QrRank2, TensorTypes, TensorTypeNames);

// Q, R = np.linalg.qr(A, mode='reduced')
TYPED_TEST(QrRank2, Reconstructs) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL * (kL + 1), kL * kL));
  const Shape shape = A.shape();
  fill_tensor(A, func2_1<T>);

  TypeParam Q, R;
  qr(A, Q, R);

  TypeParam B = tensordot(Q, R, Axes(1), Axes(0));

  const double error =
      max_error(B, [&](const Index& b) { return func2_1<T>(b, shape); });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
