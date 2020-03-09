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

#include "mptensor/rsvd.hpp"
#include "mptensor/complex.hpp"

#if defined(_USE_RANDOM_CPP11)
#include <random>
typedef typename std::mt19937 gen_t;
typedef typename std::uniform_real_distribution<double> dist_t;

#elif defined(_USE_RANDOM_BOOST)
#include <boost/random.hpp>
typedef typename boost::random::mt19937 gen_t;
typedef typename boost::random::uniform_real_distribution<double> dist_t;

#elif defined(_USE_RANDOM_DSFMT)
#define DSFMT_MEXP 19937
#include <dSFMT.h>

#endif

namespace mptensor {
namespace random_tensor {

#if defined(_USE_RANDOM_CPP11) || defined(_USE_RANDOM_BOOST)
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

#elif defined(_USE_RANDOM_DSFMT)
dsfmt_t dsfmt;
double dist() { return 2.0 * dsfmt_genrand_close_open(&dsfmt) - 1.0; };
template <>
double uniform_dist() {
  return dist();
};
template <>
complex uniform_dist() {
  return complex(dist(), dist());
};
void set_seed(unsigned int seed) { dsfmt_init_gen_rand(&dsfmt, seed); };

#endif

}  // namespace random_tensor
}  // namespace mptensor

//! @endcond
