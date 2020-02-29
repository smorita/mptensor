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
  \file   eigh.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code for eigenvalue decomposition
*/

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mptensor.hpp>
#include "functions.hpp"
#include "mpi_tool.hpp"
#include "timer.hpp"
#include "typedef.hpp"

namespace tests {

//! Test for TensorD::eigh
/*! A[i,j,k,l] => Contract( Z[i,k,a] * W[a] * (Z[a,l,j])^t )

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L,L,L,L)
  \param ostrm output stream for results
*/
void test_eigh(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t N = L * L;

  TensorD A(Shape(N, N));

  Shape shape = A.shape();
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    Index index_r(index[1], index[0]);
    double val = func2_1(index, shape);
    val += func2_1(index_r, shape);
    A.set_value(index, val);
  }
  A = transpose(reshape(A, Shape(L, L, L, L)), Axes(0, 3, 1, 2));

  TensorD Z;
  std::vector<double> W;

  time0.now();

  // A[i,j,k,l] => Contract( Z[i,k,a] * W[a] * (Z[a,l,j])^t )
  // Z^t A Z = diag(W)
  eigh(A, Axes(0, 2), Axes(3, 1), W, Z);

  time1.now();

  /* Check */
  TensorD B = Z;
  Z.multiply_vector(W, 2);  // Z[i,k,a] <= Z[i,k,a] * W[a]
  B = transpose(tensordot(Z, B, Axes(2), Axes(2)), Axes(0, 3, 1, 2));
  // now B[i,j,k,l] = A[i,j,k,l]

  double error = 0.0;
  Index index;
  index.resize(4);
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val;
    B.get_value(index, val);

    double exact;
    A.get_value(index, exact);

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);

  if (mpiroot) {
    ostrm << "========================================\n"
          << "eigh <double> ( A[L,L,L,L], Axes(0,2), Axes(3,1), W, Z )\n"
          << "[L,L,L,L] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "Z: ";
    Z.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    int n = (W.size() < 5) ? W.size() : 5;
    for (int i = 0; i < n; ++i) {
      ostrm << "S[" << i << "]= " << W[i] << "\n";
    }
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

//! Test for TensorC::eigh
/*! A[i,j,k,l] => Contract( Z[i,k,a] * W[a] * conj(Z[a,l,j]^t) )

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L,L,L,L)
  \param ostrm output stream for results
*/
void test_eigh_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t N = L * L;

  TensorC A(Shape(N, N));

  Shape shape = A.shape();
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    Index index_r(index[1], index[0]);
    complex val = cfunc2_1(index, shape);
    val += conj(cfunc2_1(index_r, shape));
    A.set_value(index, val);
  }
  A = transpose(reshape(A, Shape(L, L, L, L)), Axes(0, 3, 1, 2));

  TensorC Z;
  std::vector<double> W;

  time0.now();

  // A[i,j,k,l] => Contract( Z[i,k,a] * W[a] * (Z[a,l,j])^t )
  // Z^t A Z = diag(W)
  eigh(A, Axes(0, 2), Axes(3, 1), W, Z);

  time1.now();

  /* Check */
  TensorC B = conj(Z);
  Z.multiply_vector(W, 2);  // Z[i,k,a] <= Z[i,k,a] * W[a]
  B = transpose(tensordot(Z, B, Axes(2), Axes(2)), Axes(0, 3, 1, 2));
  // now B[i,j,k,l] = A[i,j,k,l]

  double error = 0.0;
  Index index;
  index.resize(4);
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val;
    B.get_value(index, val);

    complex exact;
    A.get_value(index, exact);

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);

  if (mpiroot) {
    ostrm << "========================================\n"
          << "eigh <complex> ( A[L,L,L,L], Axes(0,2), Axes(3,1), W, Z )\n"
          << "[L,L,L,L] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "Z: ";
    Z.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    int n = (W.size() < 5) ? W.size() : 5;
    for (int i = 0; i < n; ++i) {
      ostrm << "S[" << i << "]= " << W[i] << "\n";
    }
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
