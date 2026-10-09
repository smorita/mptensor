#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

// A[L, L+1, L+2, L+3] filled with func4_1.
template <typename TensorType>
TensorType make_rank4() {
  using T = typename TensorType::value_type;
  TensorType A(Shape(kL, kL + 1, kL + 2, kL + 3));
  fill_tensor(A, func4_1<T>);
  return A;
}

template <typename TensorType>
double max_diff(const TensorType& x, const TensorType& y) {
  return max_abs(x - y);
}

template <typename TensorType>
class NegativeIndex : public ::testing::Test {};
TYPED_TEST_SUITE(NegativeIndex, TensorTypes, TensorTypeNames);

TYPED_TEST(NegativeIndex, TransposeFreeAndMember) {
  const TypeParam A = make_rank4<TypeParam>();
  const TypeParam expected = transpose(A, Axes(2, 0, 3, 1));
  EXPECT_EQ(max_diff(transpose(A, Axes(-2, 0, -1, 1)), expected), 0.0);
  EXPECT_EQ(max_diff(transpose(A, Axes(-2, -4, -1, -3), 2),
                     transpose(A, Axes(2, 0, 3, 1), 2)),
            0.0);
  TypeParam B = A;
  B.transpose(Axes(2, -4, 3, -3));
  EXPECT_EQ(max_diff(B, expected), 0.0);
  EXPECT_EQ(max_diff(transpose(A, range(-4, 0)), A), 0.0);
  EXPECT_EQ(max_diff(transpose(A, range(-1, -5, -1)),
                     transpose(A, Axes(3, 2, 1, 0))),
            0.0);
}

TYPED_TEST(NegativeIndex, Tensordot) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 4, kL + 1, kL + 5));
  TypeParam B(Shape(kL + 5, kL + 2, kL + 4, kL + 3));
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);
  EXPECT_EQ(max_diff(tensordot(A, B, Axes(-3, -1), Axes(2, -4)),
                     tensordot(A, B, Axes(1, 3), Axes(2, 0))),
            0.0);
}

TYPED_TEST(NegativeIndex, ContractAndTrace) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL, kL + 2));
  fill_tensor(A, func4_1<T>);
  EXPECT_EQ(max_diff(contract(A, -4, -2), contract(A, 0, 2)), 0.0);

  TypeParam C(Shape(kL, kL + 1, kL + 1, kL));
  fill_tensor(C, func4_1<T>);
  EXPECT_EQ(std::abs(trace(C, Axes(0, -3), Axes(-1, 2)) -
                     trace(C, Axes(0, 1), Axes(3, 2))),
            0.0);

  TypeParam D(Shape(kL + 2, kL + 1, kL + 3, kL));
  fill_tensor(D, func4_2<T>);
  TypeParam E(Shape(kL, kL + 1, kL + 2, kL + 3));
  fill_tensor(E, func4_1<T>);
  EXPECT_EQ(std::abs(trace(E, D, Axes(0, -1, 2, -3), Axes(-1, 2, 0, -3)) -
                     trace(E, D, Axes(0, 3, 2, 1), Axes(3, 2, 0, 1))),
            0.0);
}

// Decompositions are not bitwise reproducible between calls with some process
// counts, so they are checked by reconstruction and by comparing the spectra.
template <typename T>
double max_vector_diff(const std::vector<T>& x, const std::vector<T>& y) {
  EXPECT_EQ(x.size(), y.size());
  double d = 0.0;
  for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) {
    d = std::max(d, std::abs(x[i] - y[i]));
  }
  return d;
}

TYPED_TEST(NegativeIndex, Svd) {
  const TypeParam A = make_rank4<TypeParam>();
  TypeParam U, V, U2, V2;
  std::vector<double> S, S2;
  svd(A, Axes(-2, 0), Axes(1, -1), U, S, V);  // == Axes(2, 0), Axes(1, 3)
  svd(A, Axes(2, 0), Axes(1, 3), U2, S2, V2);
  EXPECT_LT(max_vector_diff(S, S2), kEps);
  EXPECT_EQ(U.shape(), U2.shape());
  EXPECT_EQ(V.shape(), V2.shape());
  U.multiply_vector(S, 2);
  // tensordot gives B[k,i,j,l] = A[i,j,k,l]
  EXPECT_LT(max_diff(tensordot(U, V, Axes(2), Axes(0)),
                     transpose(A, Axes(2, 0, 1, 3))),
            kEps);
}

TYPED_TEST(NegativeIndex, Qr) {
  const TypeParam A = make_rank4<TypeParam>();
  TypeParam Q, R;
  qr(A, Axes(-2, -4), Axes(-3, -1), Q, R);  // == Axes(2, 0), Axes(1, 3)
  EXPECT_LT(max_diff(transpose(tensordot(Q, R, Axes(2), Axes(0)),
                               Axes(1, 2, 0, 3)),
                     A),
            kEps);
}

TYPED_TEST(NegativeIndex, Eigh) {
  using T = typename TypeParam::value_type;
  TypeParam H(Shape(kL, kL, kL, kL));
  fill_tensor(H, [](const Index& idx, const Shape& shape) {
    return func4_1<T>(idx, shape) +
           conj_value(func4_1<T>(Index(idx[2], idx[3], idx[0], idx[1]), shape));
  });
  TypeParam Z, Z2;
  std::vector<double> W, W2;
  eigh(H, Axes(-4, -3), Axes(2, -1), W, Z);  // == Axes(0, 1), Axes(2, 3)
  eigh(H, Axes(0, 1), Axes(2, 3), W2, Z2);
  EXPECT_LT(max_vector_diff(W, W2), kEps);
  // H[i,j,k,l] = sum_a Z[i,j,a] W[a] conj(Z[k,l,a])
  TypeParam B = conj(Z);
  Z.multiply_vector(W, 2);
  EXPECT_LT(max_diff(tensordot(Z, B, Axes(2), Axes(2)), H), kEps);
}

TYPED_TEST(NegativeIndex, MultiplyVector) {
  using T = typename TypeParam::value_type;
  TypeParam X = make_rank4<TypeParam>();
  TypeParam Y = X;
  std::vector<T> v(kL + 2);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = T(1.0 + 0.5 * i);
  X.multiply_vector(v, 2);
  Y.multiply_vector(v, -2);
  EXPECT_EQ(max_diff(X, Y), 0.0);
}

TYPED_TEST(NegativeIndex, SliceAndSetSlice) {
  using T = typename TypeParam::value_type;
  const TypeParam A = make_rank4<TypeParam>();  // (10, 11, 12, 13)
  EXPECT_EQ(max_diff(slice(A, -3, 3, -5), slice(A, 1, 3, 6)), 0.0);
  EXPECT_EQ(max_diff(slice(A, Index(0, 2, 0, -12), Index(0, -7, 0, 5)),
                     slice(A, Index(0, 2, 0, 1), Index(0, 4, 0, 5))),
            0.0);
  // Raw begin == end keeps the whole axis, also for a nonzero value.
  EXPECT_EQ(max_diff(slice(A, Index(3, 2, -1, 0), Index(3, 4, -1, 0)),
                     slice(A, Index(0, 2, 0, 0), Index(0, 4, 0, 0))),
            0.0);

  TypeParam X = A;
  TypeParam Y = A;
  TypeParam B(Shape(kL, 3, kL + 2, kL + 3));
  TypeParam C(Shape(3, 2, kL + 2, 3));
  fill_tensor(B, func4_2<T>);
  fill_tensor(C, func4_2<T>);
  X.set_slice(B, -3, 3, -5);
  Y.set_slice(B, 1, 3, 6);
  X.set_slice(C, Index(-9, 0, 0, -11), Index(4, -9, 0, -8));
  Y.set_slice(C, Index(1, 0, 0, 2), Index(4, 2, 0, 5));
  EXPECT_EQ(max_diff(X, Y), 0.0);
}

TYPED_TEST(NegativeIndex, GetAndSetValue) {
  using T = typename TypeParam::value_type;
  TypeParam X(Shape(kL, kL + 1, kL + 2, kL + 3));
  TypeParam Y(Shape(kL, kL + 1, kL + 2, kL + 3));
  X = 0.0;
  Y = 0.0;
  X.set_value(Index(-1, -2, 0, -1), T(5.0));
  Y.set_value(Index(kL - 1, kL - 1, 0, kL + 2), T(5.0));
  EXPECT_EQ(max_diff(X, Y), 0.0);

  T value = 0.0;
  const bool mine = X.get_value(Index(-1, -2, 0, -1), value);
  if (mine) EXPECT_EQ(value, T(5.0));
  EXPECT_EQ(max_over_ranks(mine ? 1.0 : 0.0, X.get_comm()), 1.0);
}

template <typename TensorType>
class NegativeIndexExceptions : public ::testing::Test {};
TYPED_TEST_SUITE(NegativeIndexExceptions, TensorTypes, TensorTypeNames);

TYPED_TEST(NegativeIndexExceptions, OutOfRangeAxes) {
  const TypeParam A = make_rank4<TypeParam>();
  EXPECT_THROW(transpose(A, Axes(0, 1, 2, 4)), std::out_of_range);
  EXPECT_THROW(transpose(A, Axes(-5, 1, 2, 3), 2), std::out_of_range);
  EXPECT_THROW(tensordot(A, A, Axes(1, 4), Axes(1, 3)), std::out_of_range);
  TypeParam U, V;
  std::vector<double> S;
  EXPECT_THROW(svd(A, Axes(0, 1), Axes(2, -5), U, S, V), std::out_of_range);
  EXPECT_THROW(slice(A, 4, 0, 1), std::out_of_range);
}

TYPED_TEST(NegativeIndexExceptions, OutOfRangeIndices) {
  using T = typename TypeParam::value_type;
  TypeParam A = make_rank4<TypeParam>();  // (10, 11, 12, 13)
  T value;
  EXPECT_THROW(A.get_value(Index(0, 0, 0, 13), value), std::out_of_range);
  EXPECT_THROW(A.set_value(Index(-11, 0, 0, 0), T(1.0)), std::out_of_range);
  EXPECT_THROW(A.get_value(Index(0, 0, 0), value), std::invalid_argument);
  EXPECT_THROW(slice(A, 1, 0, 12), std::out_of_range);  // end > n
  EXPECT_THROW(slice(A, 1, 3, 3), std::out_of_range);   // empty
  EXPECT_THROW(slice(A, Index(0, 2, 0, 0), Index(0, -9, 0, 0)),
               std::out_of_range);  // [2, 2) is empty
}

TYPED_TEST(NegativeIndexExceptions, NegativeShape) {
  const TypeParam A = make_rank4<TypeParam>();
  EXPECT_THROW(TypeParam(Shape(2, -1)), std::invalid_argument);
  EXPECT_THROW(reshape(A, Shape(-1, (kL + 2) * (kL + 3))),
               std::invalid_argument);
}

TYPED_TEST(NegativeIndexExceptions, TensorUnchangedAfterThrow) {
  using T = typename TypeParam::value_type;
  TypeParam A = make_rank4<TypeParam>();
  const TypeParam A0 = A;
  const Shape shape0 = A.shape();

  EXPECT_THROW(A.transpose(Axes(0, 1, 2, 9)), std::out_of_range);
  TypeParam B(Shape(kL, 3, kL + 2, kL + 3));
  EXPECT_THROW(A.set_slice(B, 1, 3, 99), std::out_of_range);
  std::vector<double> v(kL + 3, 2.0);
  EXPECT_THROW(A.multiply_vector(v, 4), std::out_of_range);
  EXPECT_THROW(A.set_value(Index(0, 0, 0, 99), T(1.0)), std::out_of_range);

  EXPECT_EQ(A.shape(), shape0);
  EXPECT_EQ(max_diff(A, A0), 0.0);
}

TYPED_TEST(NegativeIndexExceptions, CollectiveContinuesAfterThrow) {
  const TypeParam A = make_rank4<TypeParam>();
  EXPECT_THROW(tensordot(A, A, Axes(0, 9), Axes(0, 1)), std::out_of_range);
  // Every rank threw at the same point, so the next collective call matches.
  const TypeParam C = tensordot(A, A, Axes(0, 1), Axes(0, 1));
  EXPECT_EQ(C.shape(), Shape(kL + 2, kL + 3, kL + 2, kL + 3));
  EXPECT_GT(max_abs(C), 0.0);
}

}  // namespace
}  // namespace mptensor_test
