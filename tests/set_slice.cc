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
  \file   set_slice.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code for set_slice
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

//! Test for TensorD::set_slice
/*! A[:, 3:6, :, :] = B and A[1:4, 0:2, :, 2:5] = C

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_set_slice(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2, time3;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L + 1;
  int N2 = L + 2;
  int N3 = L + 3;

  if (N0 < 4) N0 = 4;
  if (N1 < 6) N1 = 6;
  if (N3 < 5) N3 = 5;

  TensorD A(Shape(N0, N1, N2, N3));
  TensorD B(Shape(N0, 3, N2, N3));
  TensorD C(Shape(3, 2, N2, 3));

  A = 1.0;

  Shape shape_B = B.shape();
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val = func4_1(index, shape_B);
    B[i] = val;
  }

  Shape shape_C = C.shape();
  for (size_t i = 0; i < C.local_size(); ++i) {
    Index index = C.global_index(i);
    double val = func4_2(index, shape_C);
    C[i] = val;
  }

  time0.now();

  // A[:, 3:6, :, :] = B
  A.set_slice(B, 1, 3, 6);

  time1.now();

  // A[1:4, 0:2, :, 2:5] = C
  A.set_slice(C, Index(1, 0, 0, 2), Index(4, 2, 0, 5));

  time2.now();

  double error = 0.0;
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val, exact;
    A.get_value(index, val);

    if (index[1] >= 3 && index[1] < 6) {
      index[1] -= 3;
      exact = func4_1(index, shape_B);
    } else if (index[0] >= 1 && index[0] < 4 && index[1] >= 0 && index[1] < 2 &&
               index[3] >= 2 && index[3] < 5) {
      index[0] -= 1;
      index[3] -= 2;
      exact = func4_2(index, shape_C);
    } else {
      exact = 1.0;
    }

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time3.now();

  double max_error = mpi_reduce_max(error, comm);

  if (mpiroot) {
    ostrm
        << "========================================\n"
        << "set_slice <double> A.set_slice(B, 1, 3, 6)\n"
        << "set_slice <double> A.set_slice(C, Index(1,0,0,2), Index(4,2,0,5))\n"
        << "A[N0, N1, N2, N3] = " << A.shape() << "\n"
        << "Error= " << max_error << "\n"
        << "Time(B)= " << time1 - time0 << " [sec]\n"
        << "Time(C)= " << time2 - time1 << " [sec]\n"
        << "Time(check)= " << time3 - time2 << " [sec]\n"
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

//! Test for TensorC::slice
/*! A[:, 3:6, :, :] = B and A[1:4, 0:2, :, 2:5] = C

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_set_slice_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2, time3;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L + 1;
  int N2 = L + 2;
  int N3 = L + 3;

  if (N0 < 4) N0 = 4;
  if (N1 < 6) N1 = 6;
  if (N3 < 5) N3 = 5;

  TensorC A(Shape(N0, N1, N2, N3));
  TensorC B(Shape(N0, 3, N2, N3));
  TensorC C(Shape(3, 2, N2, 3));

  A = 1.0;

  Shape shape_B = B.shape();
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val = cfunc4_1(index, shape_B);
    B[i] = val;
  }

  Shape shape_C = C.shape();
  for (size_t i = 0; i < C.local_size(); ++i) {
    Index index = C.global_index(i);
    complex val = cfunc4_2(index, shape_C);
    C[i] = val;
  }

  time0.now();

  // A[:, 3:6, :, :] = B
  A.set_slice(B, 1, 3, 6);

  time1.now();

  // A[1:4, 0:2, :, 2:5] = C
  A.set_slice(C, Index(1, 0, 0, 2), Index(4, 2, 0, 5));

  time2.now();

  double error = 0.0;
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val, exact;
    A.get_value(index, val);

    if (index[1] >= 3 && index[1] < 6) {
      index[1] -= 3;
      exact = cfunc4_1(index, shape_B);
    } else if (index[0] >= 1 && index[0] < 4 && index[1] >= 0 && index[1] < 2 &&
               index[3] >= 2 && index[3] < 5) {
      index[0] -= 1;
      index[3] -= 2;
      exact = cfunc4_2(index, shape_C);
    } else {
      exact = 1.0;
    }

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time3.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "set_slice <complex> A.set_slice(B, 1, 3, 6)\n"
          << "set_slice <complex> A.set_slice(C, Index(1,0,0,2), "
             "Index(4,2,0,5))\n"
          << "A[N0, N1, N2, N3] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time(B)= " << time1 - time0 << " [sec]\n"
          << "Time(C)= " << time2 - time1 << " [sec]\n"
          << "Time(check)= " << time3 - time2 << " [sec]\n"
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
