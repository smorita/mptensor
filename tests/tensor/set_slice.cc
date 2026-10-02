#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class SetSlice : public ::testing::Test {};
TYPED_TEST_SUITE(SetSlice, TensorTypes, TensorTypeNames);

// A[:, 3:6, :, :] = B and A[1:4, 0:2, :, 2:5] = C; other elements stay 1.
TYPED_TEST(SetSlice, OverwritesRanges) {
  using T = typename TypeParam::value_type;
  const int N0 = std::max(kL, 4), N1 = std::max(kL + 1, 6), N2 = kL + 2,
            N3 = std::max(kL + 3, 5);
  TypeParam A(Shape(N0, N1, N2, N3));
  TypeParam B(Shape(N0, 3, N2, N3));
  TypeParam C(Shape(3, 2, N2, 3));
  A = 1.0;
  fill_tensor(B, func4_1<T>);
  fill_tensor(C, func4_2<T>);
  const Shape shape_B = B.shape();
  const Shape shape_C = C.shape();

  A.set_slice(B, 1, 3, 6);
  A.set_slice(C, Index(1, 0, 0, 2), Index(4, 2, 0, 5));

  const double error = max_error(A, [&](Index a) -> T {
    if (a[1] >= 3 && a[1] < 6) {
      a[1] -= 3;
      return func4_1<T>(a, shape_B);
    }
    if (a[0] >= 1 && a[0] < 4 && a[1] < 2 && a[3] >= 2 && a[3] < 5) {
      a[0] -= 1;
      a[3] -= 2;
      return func4_2<T>(a, shape_C);
    }
    return T(1.0);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
