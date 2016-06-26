/*!
  \file   typedef.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>

  \brief  Define TensorD and TensorC.

  If you want to use other Matrix classes in TESTS, please modify this file.
*/

#ifndef _TEST_TYPEDEF_HPP_
#define _TEST_TYPEDEF_HPP_

#include <mptensor.hpp>
namespace tests {
using namespace mptensor;

// ScaLAPACK
typedef Tensor<scalapack::Matrix,double> TensorD;
typedef Tensor<scalapack::Matrix,complex> TensorC;

// // LAPACK
// typedef Tensor<lapack::Matrix,double> TensorD;
// typedef Tensor<lapack::Matrix,complex> TensorC;

} // namespace tests
#endif // _TEST_TYPEDEF_HPP_

