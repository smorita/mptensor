#include <type_traits>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Arithmetic : public ::testing::Test {};
TYPED_TEST_SUITE(Arithmetic, TensorTypes, TensorTypeNames);

// A chain of +=, *=, /=, -=, binary +, -, *, / and unary minus.
TYPED_TEST(Arithmetic, OperatorChain) {
  using T = typename TypeParam::value_type;
  const Shape shape(kL, kL + 1, kL + 2, kL + 3);
  TypeParam A(shape);
  TypeParam B(shape);
  TypeParam C;
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);

  T coef_1, coef_2;  // expected A = coef_1 * func4_1 + coef_2 * func4_2
  if constexpr (std::is_same_v<T, double>) {
    A += B;
    A *= 2.0;
    B /= 3.0;
    A -= B;
    A = A * 3.0;
    C = A + B;
    B = A / 2.0;
    C = C - B;
    B = -C;
    A = 0.6 * B;
    coef_1 = -1.8;
    coef_2 = -1.7;
  } else {
    A += B;
    A *= complex(0.0, 2.0);
    B /= 3.0;
    A -= B;
    A = A * 3.0;
    C = A + B;
    B = A / complex(1.0, 2.0);
    C = C - B;
    B = -C;
    A = 0.75 * B;
    coef_1 = complex(1.8, -3.6);
    coef_2 = complex(2.15, -3.3);
  }

  const double error = max_error(A, [&](const Index& a) {
    return coef_1 * func4_1<T>(a, shape) + coef_2 * func4_2<T>(a, shape);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
