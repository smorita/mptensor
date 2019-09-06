/*
  mptensor - Parallel Library for Tensor Network Methods

  Copyright 2016 Satoshi Morita

  mptensor is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  mptensor is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with mptensor.  If not, see
  <https://www.gnu.org/licenses/>.
*/

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

#ifdef _NO_MPI
// LAPACK
typedef Tensor<lapack::Matrix,double> TensorD;
typedef Tensor<lapack::Matrix,complex> TensorC;

#else
// ScaLAPACK
typedef Tensor<scalapack::Matrix,double> TensorD;
typedef Tensor<scalapack::Matrix,complex> TensorC;

#endif

} // namespace tests
#endif // _TEST_TYPEDEF_HPP_
