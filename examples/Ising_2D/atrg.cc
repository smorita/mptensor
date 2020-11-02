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
  \file   atrg.cc
  \author Daiki Adachi <daiki.adachi@phys.s.u-tokyo.ac.jp>
  \date   Oct 29, 2020

  \brief  Two-dimensional Ising model by ATRG

  This sample program calculates the free energy of the two-dimensional Ising
  model by using an anisotropic tensor renormalization group(ATRG) method.

  \par Reference
  D. Adachi, T. Okubo, S. Todo: Phys. Rev. B \b 102, 054432 (2020)
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

//! class for ATRG
class Atrg {
 public:
  Atrg(double temp);
  double free_energy() const;
  double n_spin() const;
  void update(int chi, int direction);

  double temp;
  tensor a_up;    // 3-leg tensor. [Top][Right][Middle]
  tensor a_down;  // 3-leg tensor. [Middle][Left][Down]
  double log_factor;
  double log_n_spin;

 private:
  void update_child(int chi);
  void initialize_up(tensor& A, tensor& B);
  void initialize_down(tensor& C, tensor& D);
  void swap(int chi, tensor& B, tensor& C);
  void update_from_ABCD(int chi, tensor& A, tensor& B, tensor& C, tensor& D);
};

Atrg::Atrg(double t) : temp(t) {
  a_up = tensor(Shape(2, 2, 2));
  a_down = tensor(Shape(2, 2, 2));
  const double c = sqrt(cosh(1.0 / temp));
  const double s = sqrt(sinh(1.0 / temp));
  Index idx;
  for (size_t i = 0; i < a_up.local_size(); ++i) {
    idx = a_up.global_index(i);
    a_up[i] = 1.0;
    idx[0] == 0 ? a_up[i] *= c : a_up[i] *= s;
    idx[1] == 0 ? a_up[i] *= c : a_up[i] *= s;
    if (idx[0] * idx[2] + idx[1] * idx[2] == 1) a_up[i] *= -1;
  }
  for (size_t i = 0; i < a_down.local_size(); ++i) {
    idx = a_down.global_index(i);
    a_down[i] = 1.0;
    idx[1] == 0 ? a_down[i] *= c : a_down[i] *= s;
    idx[2] == 0 ? a_down[i] *= c : a_down[i] *= s;
    if (idx[0] * idx[1] + idx[0] * idx[2] == 1) a_down[i] *= -1;
  }

  double val = trace(a_up, a_down, Axes(0, 1, 2), Axes(2, 1, 0));
  a_up /= sqrt(val);
  a_down /= sqrt(val);
  log_factor = log(val);
  log_n_spin = log(1.0);
  return;
}

inline double Atrg::free_energy() const {
  return -temp *
         (log_factor + log(trace(a_up, a_down, Axes(0, 1, 2), Axes(2, 1, 0)))) /
         exp(log_n_spin);
}

inline double Atrg::n_spin() const { return exp(log_n_spin); }

void Atrg::update(int chi, int direction) {
  if (direction == 0) {
    // top direction
    update_child(chi);
  } else if (direction == 1) {
    // right direction
    a_up.transpose(Axes(1, 0, 2));
    a_down.transpose(Axes(0, 2, 1));
    update_child(chi);
    a_up.transpose(Axes(1, 0, 2));
    a_down.transpose(Axes(0, 2, 1));
  }
  return;
}

void Atrg::initialize_up(tensor& A, tensor& B) {
  tensor u, vt;
  std::vector<double> s;
  svd(a_up, Axes(0, 1), Axes(2), u, s, vt);
  A = u;
  B = tensordot(vt.multiply_vector(s, 0), a_down, Axes(1), Axes(0));
  return;
}

void Atrg::initialize_down(tensor& C, tensor& D) {
  tensor u, vt;
  std::vector<double> s;
  svd(a_down, Axes(0), Axes(1, 2), u, s, vt);
  C = tensordot(a_up, u.multiply_vector(s, 1), Axes(2), Axes(0));
  D = vt;
  return;
}

void Atrg::swap(int chi, tensor& B, tensor& C) {
  Shape shape = B.shape();
  size_t size = std::min(size_t(chi), shape[0] * shape[0]);
  tensor u, vt;
  std::vector<double> s, sqrt_s(size);

  svd(tensordot(B, C, Axes(2), Axes(0)), Axes(0, 2), Axes(1, 3), u, s, vt);
  if (s.size() > size) {
    u = slice(u, 2, 0, size);
    vt = slice(vt, 0, 0, size);
  }
  for (size_t i = 0; i < size; ++i) sqrt_s[i] = sqrt(s[i]);
  B = u.multiply_vector(sqrt_s, 2);
  C = vt.multiply_vector(sqrt_s, 0);
  return;
}

void Atrg::update_from_ABCD(int chi, tensor& A, tensor& B, tensor& C,
                            tensor& D) {
  Shape shape = A.shape();
  size_t size = std::min(size_t(chi), shape[0] * shape[0]);
  tensor u, vt;
  std::vector<double> s, sqrt_s(size);

  // truncated svd without outer tensordot should be implemented for O(chi^5)
  // algorithm
  svd(tensordot(tensordot(A, B, Axes(2), Axes(0)),
                tensordot(C, D, Axes(2), Axes(0)), Axes(1, 2), Axes(1, 2)),
      Axes(0, 1), Axes(2, 3), u, s, vt);
  if (s.size() > size) {
    u = slice(u, 2, 0, size);
    vt = slice(vt, 0, 0, size);
  }
  for (size_t i = 0; i < size; ++i) sqrt_s[i] = sqrt(s[i]);
  a_up = u.multiply_vector(sqrt_s, 2);
  a_down = vt.multiply_vector(sqrt_s, 0);
  a_up.transpose(Axes(0, 2, 1));
  a_down.transpose(Axes(1, 0, 2));
  return;
}

void Atrg::update_child(int chi) {
  Shape shape = a_up.shape();
  tensor A, B, C, D;
  initialize_up(A,
                B);  // create A[Top][Right][Middle] & B[Middle][Left][Bottom]
  initialize_down(C,
                  D);  // create C[Top][Right][Middle] & D[Middle][Left][Bottom]
  swap(chi, B,
       C);  // swap tensors: B[Middle][Left][Bottom] & C[Top][Right][Middle] ->
            // B[Middle][Right][Bottom] & C[Middle][Left][Top]
  update_from_ABCD(chi, A, B, C, D);

  log_factor *= 2.0;       // factor_new = factor_old^2
  log_n_spin += log(2.0);  // n_new = 2*n_old

  double val = trace(a_up, a_down, Axes(0, 1, 2), Axes(2, 1, 0));
  a_up /= sqrt(val);
  a_down /= sqrt(val);
  log_factor += log(val);
  return;
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
int main(int argc, char** argv) {
  using namespace examples::Ising_2D;
  /* Start */
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);
  mpiroot = (mpirank == 0);

  /* Get arguments */
  if (mpiroot) {
    if (argc < 4) std::cerr << "Usage: atrg.out chi step T\n";
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

  Atrg atrg(temp);
  output(0, atrg.n_spin(), atrg.free_energy(), f_exact);

  for (int i = 0; i < step; ++i) {
    atrg.update(chi, i % 2);
    output(i + 1, atrg.n_spin(), atrg.free_energy(), f_exact);
  }

  /* End */
  MPI_Finalize();
}
