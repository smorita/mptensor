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
  \file   trg.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Aug 25 2016

  \brief  Two-dimensional Ising model by TRG

  This sample program calculates the free energy of the two-dimensional Ising
  model by using a tensor renormalization group (TRG).

  \par Reference
  M. Levin and C. P. Nave: Phys. Rev. Lett. \b 99, 120601 (2007)
*/

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <mptensor/mptensor.hpp>

#include "ising.hpp"

namespace examples {
namespace Ising_2D {

using namespace mptensor;

//! class for TRG
class Trg {
 public:
  Trg(double temp);
  double free_energy() const;
  double n_spin() const;
  void update(size_t chi);

  double temp;
  DTensor a;  // 4-leg tensor. [Top][Right][Down][Left]
  double log_factor;
  double log_n_spin;
};

Trg::Trg(double t) : temp(t) {
  a = DTensor(Shape(2, 2, 2, 2));
  const double c = cosh(1.0 / temp);
  const double s = sinh(1.0 / temp);
  Index idx;
  for (size_t i = 0; i < a.local_size(); ++i) {
    idx = a.global_index(i);
    size_t sum = idx[0] + idx[1] + idx[2] + idx[3];
    if (sum == 0) {
      a[i] = 2 * c * c;
    } else if (sum == 2) {
      a[i] = 2 * c * s;
    } else if (sum == 4) {
      a[i] = 2 * s * s;
    }
  }
  double val = trace(a, Axes(0, 1), Axes(2, 3));
  a /= val;
  log_factor = log(val);
  log_n_spin = log(1.0);
}

inline double Trg::free_energy() const {
  return -temp * (log_factor + log(trace(a, Axes(0, 1), Axes(2, 3)))) /
         exp(log_n_spin);
}

inline double Trg::n_spin() const { return exp(log_n_spin); }

void Trg::update(size_t chi) {
  Shape shape = a.shape();
  size_t size = std::min(chi, shape[0] * shape[1]);
  DTensor c0, c1, c2, c3;
  DTensor u, v;
  std::vector<double> s, sqrt_s(size);

  // SVD (top,right) - (bottom,left)
  svd(a, Axes(0, 1), Axes(2, 3), u, s, v);
  for (size_t i = 0; i < size; ++i) sqrt_s[i] = sqrt(s[i]);
  c3 = slice(u, 2, 0, size).multiply_vector(sqrt_s, 2);
  c1 = slice(v, 0, 0, size).multiply_vector(sqrt_s, 0);

  // SVD (top,left) - (right,bottom)
  svd(a, Axes(0, 3), Axes(1, 2), u, s, v);
  for (size_t i = 0; i < size; ++i) sqrt_s[i] = sqrt(s[i]);
  c2 = slice(u, 2, 0, size).multiply_vector(sqrt_s, 2);
  c0 = slice(v, 0, 0, size).multiply_vector(sqrt_s, 0);

  a = tensordot(tensordot(c0, c1, 1, 2), tensordot(c2, c3, 1, 1), Axes(1, 3),
                Axes(2, 0));

  log_factor *= 2.0;       // factor_new = factor_old^2
  log_n_spin += log(2.0);  // n_new = 2*n_old

  double val = trace(a, Axes(0, 1), Axes(2, 3));
  a /= val;
  log_factor += log(val);
  return;
}

}  // namespace Ising_2D
}  // namespace examples

namespace {

void output(int step, double n_spin, double f, double f_exact) {
  using namespace mptensor;
  // if (mpiroot) {
  if (mpi::is_root) {
    std::cout << step << "\t" << std::scientific << std::setprecision(6)
              << n_spin << "\t" << std::scientific << std::setprecision(10) << f
              << "\t" << (f - f_exact) / std::abs(f_exact) << std::endl;
  }
}

}  // namespace

/* Main function */
int main(int argc, char **argv) {
  using namespace examples::Ising_2D;
  using namespace mptensor;

  /* Start */
  mpi::initialize(argc, argv);

  /* Get arguments */
  if (mpi::is_root) {
    if (argc < 4) std::cerr << "Usage: trg.out chi step T\n";
    if (argc < 2) std::cerr << "Warning: Assuming chi = 8\n";
    if (argc < 3) std::cerr << "Warning: Assuming step = 16\n";
    if (argc < 4) std::cerr << "Warning: Assuming T = T_c\n";
  }
  const int chi = (argc < 2) ? 8 : atoi(argv[1]);
  const int step = (argc < 3) ? 16 : atoi(argv[2]);
  const double temp = (argc < 4) ? Ising_Tc : atof(argv[3]);
  const double f_exact = exact_free_energy(temp);

  if (mpi::is_root) {
    std::cout << "##### parameters #####\n"
              << "# T= " << std::setprecision(10) << temp << "\n"
              << "# chi= " << chi << "\n"
              << "# f_exact= " << std::setprecision(10) << f_exact << "\n"
              << "##### keys #####\n"
              << "# 1: step"
              << "\n"
              << "# 2: N_spin"
              << "\n"
              << "# 3: free energy (f)"
              << "\n"
              << "# 4: relative error ((f-f_exact)/|f_exact|)"
              << "\n"
              << "##### output #####\n";
  }

  Trg trg(temp);
  output(0, trg.n_spin(), trg.free_energy(), f_exact);

  for (int i = 0; i < step; ++i) {
    trg.update(chi);
    output(i + 1, trg.n_spin(), trg.free_energy(), f_exact);
  }

  /* End */
}
