/*
  Apr. 24, 2015
  Copyright (C) 2015 Satoshi Morita
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
