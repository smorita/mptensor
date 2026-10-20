#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

// Value of a non-distributed (LAPACK) tensor; every process holds all elements.
template <typename LocalTensor>
typename LocalTensor::value_type at(const LocalTensor& t, const Index& idx) {
  typename LocalTensor::value_type v = 0.0;
  t.get_value(idx, v);
  return v;
}

template <typename TensorType>
class Gather : public ::testing::Test {};
TYPED_TEST_SUITE(Gather, TensorTypes, TensorTypeNames);

TYPED_TEST(Gather, CollectsAllElements) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2));
  const Shape shape = A.shape();
  fill_tensor(A, [](const Index& idx, const Shape& s) {
    return T(idx[0] + 100.0 * idx[1] + 10000.0 * idx[2]) / T(s[0] + 1.0);
  });
  A.transpose(Axes(2, 0, 1));  // a non-trivial layout to gather from

  const auto G = A.gather();
  ASSERT_EQ(G.shape(), A.shape());
  double error = 0.0;
  for (std::ptrdiff_t i = 0; i < shape[2]; ++i)
    for (std::ptrdiff_t j = 0; j < shape[0]; ++j)
      for (std::ptrdiff_t k = 0; k < shape[1]; ++k) {
        const T exact = T(j + 100.0 * k + 10000.0 * i) / T(shape[0] + 1.0);
        error = std::max(error, std::abs(at(G, Index(i, j, k)) - exact));
      }
  EXPECT_EQ(error, 0.0);
}

template <typename TensorType>
class Eig : public ::testing::Test {};
TYPED_TEST_SUITE(Eig, TensorTypes, TensorTypeNames);

// Max over a of |sum_k A[i,k] Z[k,a] - W[a] Z[i,a]| for gathered n x n A and Z.
template <typename LocalA, typename LocalZ>
double right_eigen_residual(const LocalA& a, const LocalZ& z,
                            const std::vector<complex>& w, std::ptrdiff_t n) {
  double r = 0.0;
  for (std::ptrdiff_t c = 0; c < n; ++c)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
      complex lhs = 0.0;
      for (std::ptrdiff_t k = 0; k < n; ++k) lhs += at(a, Index(i, k)) * at(z, Index(k, c));
      r = std::max(r, std::abs(lhs - w[c] * at(z, Index(i, c))));
    }
  return r;
}

TYPED_TEST(Eig, Rank2Eigenvectors) {
  using T = typename TypeParam::value_type;
  const std::ptrdiff_t n = kL;
  TypeParam A(Shape(n, n));
  fill_tensor(A, func2_1<T>);

  std::vector<complex> W;
  typename TypeParam::template rebind<complex> Z;
  eig(A, W, Z);
  ASSERT_EQ(W.size(), static_cast<size_t>(n));

  TypeParam A_copy = A;
  EXPECT_LT(right_eigen_residual(A_copy.gather(), Z.gather(), W, n), 1.0e-10);
}

TYPED_TEST(Eig, TensorEigenvectors) {
  using T = typename TypeParam::value_type;
  // A[i,j,k,l] as an (i,j) x (k,l) matrix of size kL^2.
  TypeParam A(Shape(kL, kL, kL, kL));
  fill_tensor(A, func4_1<T>);

  std::vector<complex> W;
  typename TypeParam::template rebind<complex> Z;
  eig(A, Axes(0, 1), Axes(2, 3), W, Z);
  const std::ptrdiff_t n = kL * kL;
  ASSERT_EQ(W.size(), static_cast<size_t>(n));
  ASSERT_EQ(Z.shape(), Shape(kL, kL, n));

  // Compare as matrices: the first index runs fastest in reshape.
  TypeParam A_mat = reshape(A, Shape(n, n));
  auto Z_mat = reshape(Z, Shape(n, n));
  double scale = 1.0;
  for (const complex& w : W) scale = std::max(scale, std::abs(w));
  EXPECT_LT(right_eigen_residual(A_mat.gather(), Z_mat.gather(), W, n),
            1.0e-10 * scale);
}

}  // namespace
}  // namespace mptensor_test
