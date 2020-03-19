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
  \file   complex.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jan 13 2015

  \brief  Define the type of a complex number.
*/

#ifndef _COMPLEX_HPP_
#define _COMPLEX_HPP_

#include <complex>

namespace mptensor {

//! Alias for the value type of complex numbers
//! \ingroup Complex
typedef std::complex<double> complex;

template <typename C>
constexpr size_t value_type_tag();
template <>
constexpr size_t value_type_tag<double>() {
  return 0;
};
template <>
constexpr size_t value_type_tag<complex>() {
  return 1;
};

template <typename C>
constexpr char* value_type_name();
template <>
constexpr char* value_type_name<double>() {
  return (char*)"double";
};
template <>
constexpr char* value_type_name<complex>() {
  return (char*)"complex";
};

}  // namespace mptensor

#endif  // _COMPLEX_HPP_
