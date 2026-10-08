#ifndef MPTENSOR_TEST_TEST_FUNCTIONS_HPP_
#define MPTENSOR_TEST_TEST_FUNCTIONS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mptensor/mptensor.hpp>

#include "mpi_helpers.hpp"

namespace mptensor_test {

using namespace mptensor;

//! Base size of test tensors (legacy default N).
constexpr int kL = 10;
//! Default tolerance.
constexpr double kEps = 1.0e-10;

namespace detail {
inline double coord(const Index& idx, const Shape& shape, std::size_t k) {
  return double(idx[k]) / double(shape[k]);
}
}  // namespace detail

//! Analytic matrix element (legacy func2_1 / cfunc2_1).
template <typename T>
T func2_1(const Index& idx, const Shape& shape) {
  const double x0 = detail::coord(idx, shape, 0);
  const double x1 = detail::coord(idx, shape, 1);
  if constexpr (std::is_same_v<T, double>) {
    return 1.0 / (1.0 + std::abs(x0 - std::sqrt(x1)));
  } else {
    return 1.0 / (1.0 + complex(x0, std::sqrt(x1)));
  }
}

//! Analytic rank-4 element (legacy func4_1 / cfunc4_1).
template <typename T>
T func4_1(const Index& idx, const Shape& shape) {
  const double x0 = detail::coord(idx, shape, 0);
  const double x1 = detail::coord(idx, shape, 1);
  const double x2 = detail::coord(idx, shape, 2);
  const double x3 = detail::coord(idx, shape, 3);
  if constexpr (std::is_same_v<T, double>) {
    return (x0 + x1) / (1.0 + std::abs(x2 - x3));
  } else {
    return complex(x0, x1) / complex(1.0 + x2, -x3);
  }
}

//! Analytic rank-4 element (legacy func4_2 / cfunc4_2).
template <typename T>
T func4_2(const Index& idx, const Shape& shape) {
  const double x0 = detail::coord(idx, shape, 0);
  const double x1 = detail::coord(idx, shape, 1);
  const double x2 = detail::coord(idx, shape, 2);
  const double x3 = detail::coord(idx, shape, 3);
  if constexpr (std::is_same_v<T, double>) {
    return std::cos(x0 * x3) + std::sin(x1 / (x2 + 1.0));
  } else {
    return complex(std::cos(x0 * x3), std::sin(x1 / (x2 + 1.0)));
  }
}

//! Complex conjugate that keeps \c double as \c double.
inline double conj_value(double x) { return x; }
inline complex conj_value(const complex& x) { return std::conj(x); }

//! Set every local element to f(global_index, shape).
template <typename TensorType, typename F>
void fill_tensor(TensorType& t, F f) {
  const Shape shape = t.shape();
  for (std::size_t i = 0; i < t.local_size(); ++i) {
    t[i] = f(t.global_index(i), shape);
  }
}

//! Maximum over all ranks of |t[idx] - exact(idx)|. Collective.
template <typename TensorType, typename F>
double max_error(const TensorType& t, F exact) {
  double error = 0.0;
  for (std::size_t i = 0; i < t.local_size(); ++i) {
    error = std::max(error, std::abs(t[i] - exact(t.global_index(i))));
  }
  return max_over_ranks(error, t.get_comm());
}

}  // namespace mptensor_test

#endif  // MPTENSOR_TEST_TEST_FUNCTIONS_HPP_
