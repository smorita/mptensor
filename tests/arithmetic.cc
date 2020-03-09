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
  \file   arithmetic.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015

  \brief  Test code for arithmetic operators
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

//! Test for arithmetic operators
/*!
  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_arithmetic(const mpi_comm &comm, int L, std::ostream &ostrm) {
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
  TensorD B(Shape(N0, N1, N2, N3));
  TensorD C;

  Shape shape_A = A.shape();
  Shape shape_B = B.shape();

  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val = func4_1(index, shape_A);
    A.set_value(index, val);
  }

  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val = func4_2(index, shape_B);
    B.set_value(index, val);
  }

  time0.now();

  A += B;
  A *= 2.0;
  B /= 3.0;
  A -= B;

  A = A * 3.0;
  C = A + B;
  B = A / 2.0;
  C = C - B;
  B = -C;
  A = 0.6 * B;

  time1.now();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(4);
  index_B.resize(4);
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val;
    A.get_value(index, val);

    double exact =
        -1.8 * func4_1(index, shape_A) - 1.7 * func4_2(index, shape_B);

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "Arithmetic Operators <double> ( A[N0, N1, N2, N3] )\n"
          << "[N0, N1, N2, N3] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

//! Test for arithmetic operators (complex version)
/*!
  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_arithmetic_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
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
  TensorC B(Shape(N0, N1, N2, N3));
  TensorC C;

  Shape shape_A = A.shape();
  Shape shape_B = B.shape();

  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val = cfunc4_1(index, shape_A);
    A.set_value(index, val);
  }

  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val = cfunc4_2(index, shape_B);
    B.set_value(index, val);
  }

  time0.now();

  A += B;
  A *= complex(0.0, 2.0);
  B /= 3.0;
  A -= B;

  A = A * 3.0;
  C = A + B;
  B = A / complex(1.0, 2.0);
  C = C - B;
  B = -C;
  A = 0.75 * B;

  time1.now();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(4);
  index_B.resize(4);
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val;
    A.get_value(index, val);

    complex exact = complex(1.8, -3.6) * cfunc4_1(index, shape_A) +
                    complex(2.15, -3.3) * cfunc4_2(index, shape_B);

    if (error < abs(val - exact)) error = abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);

  if (mpiroot) {
    ostrm << "========================================\n"
          << "Arithmetic Operators <complex> ( A[N0, N1, N2, N3] )\n"
          << "[N0, N1, N2, N3] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
