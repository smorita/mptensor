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
  \file   functions.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Functions for tensor elements
*/

#ifndef _FUNCTIONS_HPP_
#define _FUNCTIONS_HPP_

#include <cmath>
#include <mptensor.hpp>

using mptensor::Index;
using mptensor::Shape;
using mptensor::complex;


namespace tests {
//! \name For tensor elements
//! \{

/* Example of matrix element */
inline double func2_1(Index idx, Shape shape) {
  double x0 = double(idx[0])/double(shape[0]);
  double x1 = double(idx[1])/double(shape[1]);
  return 1.0/(1.0+std::abs(x0-std::sqrt(x1)));
}

inline complex cfunc2_1(Index idx, Shape shape) {
  double x0 = double(idx[0])/double(shape[0]);
  double x1 = double(idx[1])/double(shape[1]);
  return 1.0/(1.0+complex(x0,std::sqrt(x1)));
}

inline double func4_1(Index idx, Shape shape) {
  double x0 = double(idx[0])/double(shape[0]);
  double x1 = double(idx[1])/double(shape[1]);
  double x2 = double(idx[2])/double(shape[2]);
  double x3 = double(idx[3])/double(shape[3]);
  return (x0+x1)/(1.0+std::abs(x2-x3));
}

inline double func4_2(Index idx, Shape shape) {
  double x0 = double(idx[0])/double(shape[0]);
  double x1 = double(idx[1])/double(shape[1]);
  double x2 = double(idx[2])/double(shape[2]);
  double x3 = double(idx[3])/double(shape[3]);
  return std::cos(x0*x3) + std::sin(x1/(x2+1.0));
}

inline complex cfunc4_1(Index idx, Shape shape) {
  double x0 = double(idx[0])/double(shape[0]);
  double x1 = double(idx[1])/double(shape[1]);
  double x2 = double(idx[2])/double(shape[2]);
  double x3 = double(idx[3])/double(shape[3]);
  return complex(x0, x1)/complex(1.0+x2,-x3);
}

inline complex cfunc4_2(Index idx, Shape shape) {
  double x0 = double(idx[0])/double(shape[0]);
  double x1 = double(idx[1])/double(shape[1]);
  double x2 = double(idx[2])/double(shape[2]);
  double x3 = double(idx[3])/double(shape[3]);
  return complex(std::cos(x0*x3), std::sin(x1/(x2+1.0)) );
}
//! \}

} // namespace

#endif // _FUNCTIONS_HPP_
