#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

// Hermitian matrix A[i,j] = f(i,j) + conj(f(j,i)) with f = func2_1.
template <typename TensorType>
TensorType make_hermitian_matrix(int n) {
  using T = typename TensorType::value_type;
  TensorType A(Shape(n, n));
  fill_tensor(A, [](const Index& idx, const Shape& shape) {
    return func2_1<T>(idx, shape) +
           conj_value(func2_1<T>(Index(idx[1], idx[0]), shape));
  });
  return A;
}

template <typename TensorType>
class Eigh : public ::testing::Test {};
TYPED_TEST_SUITE(Eigh, TensorTypes, TensorTypeNames);

// A[i,j,k,l] = sum_a Z[i,k,a] W[a] conj(Z[l,j,a])
TYPED_TEST(Eigh, Reconstructs) {
  TypeParam A = make_hermitian_matrix<TypeParam>(kL * kL);
  A = transpose(reshape(A, Shape(kL, kL, kL, kL)), Axes(0, 3, 1, 2));

  TypeParam Z;
  std::vector<double> W;
  eigh(A, Axes(0, 2), Axes(3, 1), W, Z);

  TypeParam B = conj(Z);
  Z.multiply_vector(W, 2);  // Z[i,k,a] <= Z[i,k,a] * W[a]
  B = transpose(tensordot(Z, B, Axes(2), Axes(2)), Axes(0, 3, 1, 2));

  EXPECT_LT(max_abs(B - A), kEps);
}

template <typename TensorType>
class EighRank2 : public ::testing::Test {};
TYPED_TEST_SUITE(EighRank2, TensorTypes, TensorTypeNames);

// A[i,j] = sum_a Z[i,a] W[a] conj(Z[j,a])
TYPED_TEST(EighRank2, Reconstructs) {
  const TypeParam A = make_hermitian_matrix<TypeParam>(kL * kL);

  TypeParam Z;
  std::vector<double> W;
  eigh(A, W, Z);

  TypeParam B = conj(Z);
  Z.multiply_vector(W, 1);  // Z[i,a] <= Z[i,a] * W[a]
  B = tensordot(Z, B, Axes(1), Axes(1));

  EXPECT_LT(max_abs(B - A), kEps);
}

template <typename TensorType>
class EighGeneral : public ::testing::Test {};
TYPED_TEST_SUITE(EighGeneral, TensorTypes, TensorTypeNames);

// A[i,j,k,l] Z[l,k,a] = B[j,l,i,k] Z[l,k,a] W[a]
TYPED_TEST(EighGeneral, Solves) {
  using T = typename TypeParam::value_type;
  constexpr double kEpsGeneral = 1.0e-8;
  constexpr bool is_double = std::is_same_v<T, double>;

  // Hermitian A: double uses func4_1, complex uses func4_2 (as in legacy).
  TypeParam A(Shape(kL, kL, kL, kL));
  fill_tensor(A, [](const Index& idx, const Shape& shape) {
    const Index idx2(idx[2], idx[3], idx[0], idx[1]);
    if constexpr (is_double) {
      return func4_1<T>(idx, shape) + func4_1<T>(idx2, shape);
    } else {
      return func4_2<T>(idx, shape) + conj_value(func4_2<T>(idx2, shape));
    }
  });

  // Positive-definite B = U S U^dagger, with small singular values lifted.
  TypeParam B(Shape(kL, kL, kL, kL));
  if constexpr (is_double) {
    fill_tensor(B, func4_2<T>);
  } else {
    fill_tensor(B, func4_1<T>);
  }
  {
    TypeParam u, vt;
    std::vector<double> s;
    svd(B, Axes(0, 1), Axes(2, 3), u, s, vt);
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] < 1.0e-3) s[i] = 1.0e-3 * (i + 1);
    }
    TypeParam us = u;
    us.multiply_vector(s, 2);
    B = transpose(tensordot(us, conj(u), Axes(2), Axes(2)), Axes(1, 3, 0, 2));
  }

  TypeParam Z;
  std::vector<double> W;
  eigh(A, Axes(1, 0), Axes(3, 2), B, Axes(0, 2), Axes(1, 3), W, Z);

  TypeParam lhs =
      tensordot(A.transpose(Axes(1, 0, 3, 2)), Z, Axes(2, 3), Axes(0, 1));
  TypeParam rhs =
      tensordot(B.transpose(Axes(0, 2, 1, 3)), Z, Axes(2, 3), Axes(0, 1));
  rhs.multiply_vector(W, 2);

  EXPECT_LT(max_abs(lhs - rhs), kEpsGeneral);
}

}  // namespace
}  // namespace mptensor_test
