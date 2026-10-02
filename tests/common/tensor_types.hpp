#ifndef MPTENSOR_TEST_TENSOR_TYPES_HPP_
#define MPTENSOR_TEST_TENSOR_TYPES_HPP_

#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include <mptensor/mptensor.hpp>

namespace mptensor_test {

using namespace mptensor;

#ifdef _NO_MPI
template <typename T>
using DistMatrix = lapack::Matrix<T>;
#else
template <typename T>
using DistMatrix = scalapack::Matrix<T>;
#endif

using TensorD = Tensor<DistMatrix<double>>;
using TensorC = Tensor<DistMatrix<complex>>;

//! Tensor types every TYPED_TEST runs with.
using TensorTypes = ::testing::Types<TensorD, TensorC>;

//! Readable names ("double" / "complex") in gtest output.
struct TensorTypeNames {
  template <typename TensorType>
  static std::string GetName(int) {
    return std::is_same_v<typename TensorType::value_type, double> ? "double"
                                                                   : "complex";
  }
};

}  // namespace mptensor_test

#endif  // MPTENSOR_TEST_TENSOR_TYPES_HPP_
