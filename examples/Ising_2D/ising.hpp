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
const double Ising_Tc = 2.0/log(1.0 + sqrt(2.0));
double exact_free_energy(double temp);

} // namespace Ising_2D
} // namespace examples

#endif // _ISING_HPP_
