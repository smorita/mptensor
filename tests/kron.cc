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
  \file   tensordot.cc
  \author Satoshi Morita <smorita@keio.jp>
  \date   May 09 2023
  \brief  Test code for kron
*/

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <mptensor/mptensor.hpp>

#include "functions.hpp"
#include "mpi_tool.hpp"
#include "timer.hpp"
#include "typedef.hpp"

namespace tests {

//! Test for TensorD::kron
/*! C = kron(A, B)

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1), B.shape = (L+2, L+3)
  \param ostrm output stream for results
*/
void test_kron(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L + 1;
  int N2 = L + 2;
  int N3 = L + 3;

  TensorD A(Shape(N0, N1));
  TensorD B(Shape(N2, N3));

  Shape shape_A = A.shape();
  Shape shape_B = B.shape();

  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val = func2_1(index, shape_A);
    A.set_value(index, val);
  }

  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val = func2_1(index, shape_B);
    B.set_value(index, val);
  }

  time0.now();

  TensorD C = kron(A, B);

  time1.now();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(2);
  index_B.resize(2);
  for (size_t i = 0; i < C.local_size(); ++i) {
    Index index = C.global_index(i);
    double val;
    C.get_value(index, val);

    index_A[0] = index[0] % shape_A[0];
    index_A[1] = index[1] % shape_A[1];

    index_B[0] = index[0] / shape_A[0];
    index_B[1] = index[1] / shape_A[1];
    double exact = func2_1(index_A, shape_A) * func2_1(index_B, shape_B);

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "kron <double> ( C[M, K] = A[N0, N1] * B[N2, N3] )\n"
          << "[N0, N1, N2, N3] = [" << N0 << ", " << N1 << ", " << N2 << ", "
          << N3 << "] "
          << "M= " << N0 * N2 << " K= " << N1 * N3 << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "C: ";
    C.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  // assert(error < EPS);
  mpi_barrier(comm);
}

//! Test for TensorC::kron
/*! C = kron(A, B)

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1), B.shape = (L+2, L+3)
  \param ostrm output stream for results
*/
void test_kron_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L + 1;
  int N2 = L + 2;
  int N3 = L + 3;

  TensorC A(Shape(N0, N1));
  TensorC B(Shape(N2, N3));

  Shape shape_A = A.shape();
  Shape shape_B = B.shape();

  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val = cfunc2_1(index, shape_A);
    A.set_value(index, val);
  }

  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val = cfunc2_1(index, shape_B);
    B.set_value(index, val);
  }

  time0.now();

  TensorC C = kron(A, B);

  time1.now();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(2);
  index_B.resize(2);
  for (size_t i = 0; i < C.local_size(); ++i) {
    Index index = C.global_index(i);
    complex val;
    C.get_value(index, val);

    index_A[0] = index[0] % shape_A[0];
    index_A[1] = index[1] % shape_A[1];

    index_B[0] = index[0] / shape_A[0];
    index_B[1] = index[1] / shape_A[1];
    complex exact = cfunc2_1(index_A, shape_A) * cfunc2_1(index_B, shape_B);

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "kron <complex> ( C[M, K] = A[N0, N1] * B[N2, N3] )\n"
          << "[N0, N1, N2, N3] = [" << N0 << ", " << N1 << ", " << N2 << ", "
          << N3 << "] "
          << "M= " << N0 * N2 << " K= " << N1 * N3 << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "C: ";
    C.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
