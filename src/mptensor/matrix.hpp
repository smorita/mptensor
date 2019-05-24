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
  \file   matrix.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Thu Oct  8 15:04:31 2015
  \brief  List of header files for matrix classes

  This file will be included from tensor.hpp.
*/

#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#ifndef _NO_MPI
  #include "scalapack/matrix_scalapack.hpp"
#endif // _NO_MPI

#include "lapack/matrix_lapack.hpp"

#endif // _MATRIX_HPP_
