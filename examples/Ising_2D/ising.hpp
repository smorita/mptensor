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
  \file   ising.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Thu Nov 5 2015
  \brief  Two-dimensional Ising model
*/

#ifndef _ISING_HPP_
#define _ISING_HPP_

#include <cmath>

namespace examples {
namespace Ising_2D {

//! Tc of the Ising model on the square lattice
const double Ising_Tc = 2.0 / log(1.0 + sqrt(2.0));
double exact_free_energy(double temp);

}  // namespace Ising_2D
}  // namespace examples

#endif  // _ISING_HPP_
