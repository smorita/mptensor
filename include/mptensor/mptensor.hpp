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
  \file   mptensor.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jun 26 2016

  \brief  Top header file of mptensor.
*/

#ifndef _MPTENSOR_HPP_
#define _MPTENSOR_HPP_

#include "mptensor/version.hpp"
#include "mptensor/mpi/mpi.hpp"

#include "mptensor/tensor.hpp"

#include "mptensor/file_io/file_io.hpp"
#include "mptensor/rsvd.hpp"

namespace mptensor {

#ifdef _NO_MPI
using DTensor = Tensor<lapack::Matrix<double>>;
using ZTensor = Tensor<lapack::Matrix<complex>>;
#else
using DTensor = Tensor<scalapack::Matrix<double>>;
using ZTensor = Tensor<scalapack::Matrix<complex>>;
#endif

}  // namespace mptensor

#endif  // _MPTENSOR_HPP_
