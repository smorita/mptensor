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
  \file   tests/transpose.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code for transpose
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

//! Test for TensorD::transpose
/*! B = transpose(A, (2,0,3,1))

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_transpose(const mpi_comm &comm, int L, std::ostream &ostrm) {
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

  TensorD A(Shape(N0, N1, N2, N3));

  Shape shape_A = A.shape();
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val = func4_1(index, shape_A);
    A.set_value(index, val);
  }

  time0.now();

  // B = transpose(A, (2,0,3,1))
  TensorD B = transpose(A, Axes(2, 0, 3, 1));

  time1.now();

  double error = 0.0;
  Index index;
  index.resize(4);
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index_B = B.global_index(i);
    double val;
    B.get_value(index_B, val);

    index[0] = index_B[1];
    index[1] = index_B[3];
    index[2] = index_B[0];
    index[3] = index_B[2];

    double exact = func4_1(index, shape_A);

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "transpose <double> (A[N0,N1,N2,N3], Axes(2,0,3,1)) = "
             "B[N2,N0,N3,N1]\n"
          << "[N0, N1, N2, N3] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

//! Test for TensorC::transpose
/*! B = transpose(A, (2,0,3,1))

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_transpose_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
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

  TensorC A(Shape(N0, N1, N2, N3));

  Shape shape_A = A.shape();
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val = cfunc4_1(index, shape_A);
    A.set_value(index, val);
  }

  time0.now();

  // B = transpose(A, (2,0,3,1))
  TensorC B = transpose(A, Axes(2, 0, 3, 1));

  time1.now();

  double error = 0.0;
  Index index;
  index.resize(4);
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index_B = B.global_index(i);
    complex val;
    B.get_value(index_B, val);

    index[0] = index_B[1];
    index[1] = index_B[3];
    index[2] = index_B[0];
    index[3] = index_B[2];

    complex exact = cfunc4_1(index, shape_A);

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "transpose <complex> (A[N0,N1,N2,N3], Axes(2,0,3,1)) = "
             "B[N2,N0,N3,N1]\n"
          << "[N0, N1, N2, N3] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
