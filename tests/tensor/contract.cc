#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Contract : public ::testing::Test {};
TYPED_TEST_SUITE(Contract, TensorTypes, TensorTypeNames);

// B = contract(A, Axes(0), Axes(2)): B[a,b] = sum_i A[i,a,i,b]
TYPED_TEST(Contract, PartialTrace) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2;
  TypeParam A(Shape(N0, N1, N0, N2));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam B = contract(A, 0, 2);

  const double error = max_error(B, [&](const Index& b) {
    T exact = 0.0;
    for (int m = 0; m < N0; ++m) {
      exact += func4_1<T>(Index(m, b[0], m, b[1]), shape_A);
    }
    return exact;
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
