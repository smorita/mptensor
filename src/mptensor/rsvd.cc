/*!
  \file   rsvd.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Feb 3 2016

  \brief  Randomized algorithm for singular value decomposition.
*/
//! @cond

#include "complex.hpp"
#include "rsvd.hpp"

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
dist_t dist(-1.0,1.0);
template <> double uniform_dist() {return dist(gen);};
template <> complex uniform_dist() {return complex(dist(gen),dist(gen));};
void set_seed(unsigned int seed) {gen.seed(seed);};

#elif defined(_USE_RANDOM_DSFMT)
dsfmt_t dsfmt;
double dist() {return 2.0*dsfmt_genrand_close_open(&dsfmt)-1.0;};
template <> double uniform_dist() {return dist();};
template <> complex uniform_dist() {return complex(dist(),dist());};
void set_seed(unsigned int seed) {dsfmt_init_gen_rand(&dsfmt, seed);};

#endif


} // namespace random_tensor
} // namespace mptensor

//! @endcond
