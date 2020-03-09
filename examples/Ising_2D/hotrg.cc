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
  \file   hotrg.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Aug 25 2016

  \brief  Two-dimensional Ising model by HOTRG

  This sample program calculates the free energy of the two-dimensional Ising
  model by using a tensor renormalization group method based on higher-order
  singular value decomposition (HOTRG).

  \par Reference
  Z. Y. Xie, et al.: Phys. Rev. B \b 86, 045139 (2012)
*/

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <mpi.h>
#include <mptensor/mptensor.hpp>

#include "ising.hpp"

namespace examples {
namespace Ising_2D {

using namespace mptensor;
typedef Tensor<scalapack::Matrix, double> tensor;

//! class for HOTRG
class Hotrg {
 public:
  Hotrg(double temp);
  double free_energy() const;
  double n_spin() const;
  void update(int chi, int direction);

  double temp;
  tensor a;  // 4-leg tensor. [Top][Right][Down][Left]
  double log_factor;
  double log_n_spin;

 private:
  void update_child(int chi);
  tensor a4_top(const tensor &a) const;
  tensor a2u_top_block(const tensor &a, const tensor &u);
  void a2u_top_block_child(const tensor &a, const tensor &u, tensor &a_new,
                           size_t j0, size_t j1);
};

Hotrg::Hotrg(double t) : temp(t) {
  a = tensor(Shape(2, 2, 2, 2));
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

inline double Hotrg::free_energy() const {
  return -temp * (log_factor + log(trace(a, Axes(0, 1), Axes(2, 3)))) /
         exp(log_n_spin);
}

inline double Hotrg::n_spin() const { return exp(log_n_spin); }

void Hotrg::update(int chi, int direction) {
  if (direction == 0) {
    // top direction
    update_child(chi);
  } else if (direction == 1) {
    // right direction
    a.transpose(Axes(1, 2, 3, 0));
    update_child(chi);
    a.transpose(Axes(3, 0, 1, 2));
  }
  return;
}

void Hotrg::update_child(int chi) {
  Shape shape = a.shape();
  size_t size = std::min(size_t(chi), shape[0] * shape[0]);
  tensor u, vt;
  std::vector<double> s;

  svd(a4_top(a), Axes(0, 2), Axes(1, 3), u, s, vt);
  if (s.size() > size) {
    u = slice(u, 2, 0, size);
    vt = slice(vt, 0, 0, size);
  }

  a = a2u_top_block(a, u);

  log_factor *= 2.0;       // factor_new = factor_old^2
  log_n_spin += log(2.0);  // n_new = 2*n_old

  double val = trace(a, Axes(0, 1), Axes(2, 3));
  a /= val;
  log_factor += log(val);
}

inline tensor Hotrg::a4_top(const tensor &a) const {
  ////////////////////////////////////////////////////////////
  // A4_top.tdt
  ////////////////////////////////////////////////////////////
  // ((A0*A2)*(A1*A3))
  // cpu_cost= 3e+06  memory= 50000
  // final_bond_order (a, c, b, d)
  ////////////////////////////////////////////////////////////
  return tensordot(tensordot(a, a, Axes(2, 3), Axes(2, 3)),
                   tensordot(a, a, Axes(1, 2), Axes(1, 2)), Axes(1, 3),
                   Axes(1, 3));
}

tensor Hotrg::a2u_top_block(const tensor &a, const tensor &u) {
  Shape shape = a.shape();
  size_t n = u.shape()[2];
  tensor a_new(Shape(n, shape[1], n, shape[3]));

  const size_t block_size = 8;
  size_t block_num_j = shape[1] / block_size;
  size_t j0, j1;
  for (size_t bj = 0; bj < block_num_j; bj++) {
    j0 = bj * block_size;
    j1 = j0 + block_size;
    a2u_top_block_child(a, u, a_new, j0, j1);
  }
  if (shape[1] % block_size != 0) {
    j0 = block_num_j * block_size;
    j1 = shape[1];
    a2u_top_block_child(a, u, a_new, j0, j1);
  }

  return a_new;
}

inline void Hotrg::a2u_top_block_child(const tensor &a, const tensor &u,
                                       tensor &a_new, size_t j0, size_t j1) {
  ////////////////////////////////////////////////////////////
  // A2U_top_block_v2.tdt
  ////////////////////////////////////////////////////////////
  // ((u0*A1[:,:,:,b])*(u1*A0[:,b,:,:]))
  // cpu_cost= 2.4e+06  memory= 50000
  // final_bond_order (a, b, c, d)
  ////////////////////////////////////////////////////////////
  a_new += tensordot(tensordot(u, slice(a, 3, j0, j1), Axes(1), Axes(0)),
                     tensordot(u, slice(a, 1, j0, j1), Axes(0), Axes(2)),
                     Axes(0, 3, 4), Axes(2, 0, 3));
}

}  // namespace Ising_2D
}  // namespace examples

namespace {

MPI_Comm comm;
int mpirank;
int mpisize;
bool mpiroot;

void output(int step, double n_spin, double f, double f_exact) {
  if (mpiroot) {
    std::cout << step << "\t" << std::scientific << std::setprecision(6)
              << n_spin << "\t" << std::scientific << std::setprecision(10) << f
              << "\t" << (f - f_exact) / std::abs(f_exact) << std::endl;
  }
}

}  // namespace

/* Main function */
int main(int argc, char **argv) {
  using namespace examples::Ising_2D;
  /* Start */
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);
  mpiroot = (mpirank == 0);

  /* Get arguments */
  if (mpiroot) {
    if (argc < 4) std::cerr << "Usage: hotrg.out chi step T\n";
    if (argc < 2) std::cerr << "Warning: Assuming chi = 8\n";
    if (argc < 3) std::cerr << "Warning: Assuming step = 16\n";
    if (argc < 4) std::cerr << "Warning: Assuming T = T_c\n";
  }
  const int chi = (argc < 2) ? 8 : atoi(argv[1]);
  const int step = (argc < 3) ? 16 : atoi(argv[2]);
  const double temp = (argc < 4) ? Ising_Tc : atof(argv[3]);
  const double f_exact = exact_free_energy(temp);

  if (mpiroot) {
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

  Hotrg hotrg(temp);
  output(0, hotrg.n_spin(), hotrg.free_energy(), f_exact);

  for (int i = 0; i < step; ++i) {
    hotrg.update(chi, i % 2);
    output(i + 1, hotrg.n_spin(), hotrg.free_energy(), f_exact);
  }

  /* End */
  MPI_Finalize();
}
