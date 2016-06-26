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
