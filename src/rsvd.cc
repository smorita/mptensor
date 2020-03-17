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
  \file   mptensor/rsvd.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Feb 3 2016

  \brief  Randomized algorithm for singular value decomposition.
*/
//! @cond

#include <random>

#include "mptensor/rsvd.hpp"
#include "mptensor/complex.hpp"

typedef typename std::mt19937 gen_t;
typedef typename std::uniform_real_distribution<double> dist_t;

namespace mptensor {
namespace random_tensor {

gen_t gen;
dist_t dist(-1.0, 1.0);
template <>
double uniform_dist() {
  return dist(gen);
};
template <>
complex uniform_dist() {
  return complex(dist(gen), dist(gen));
};
void set_seed(unsigned int seed) { gen.seed(seed); };

}  // namespace random_tensor
}  // namespace mptensor

//! @endcond
