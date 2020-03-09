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
  \file   qr_rank2.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015

  \brief  Test code for QR decomposition of matrix
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

//! Test for TensorD::qr (Matrix version)
/*! \code Q, R = np.linalg.qr(A,mode='reduced') \endcode

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L*(L+1), L*L)
  \param ostrm output stream for results
*/
void test_qr_rank2(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L * (L + 1);
  int N1 = L * L;

  TensorD A(Shape(N0, N1));

  Shape shape = A.shape();
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val = func2_1(index, shape);
    A.set_value(index, val);
  }

  TensorD Q, R;

  time0.now();

  // Q, R = np.linalg.qr(A,mode='reduced')
  qr(A, Q, R);

  time1.now();

  /* Check */
  TensorD B = tensordot(Q, R, Axes(1), Axes(0));
  // now B[i,j] = A[i,j]

  double error = 0.0;
  Index index;
  index.resize(4);
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val;
    B.get_value(index, val);

    double exact = func2_1(index, shape);

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "QR <double> ( A[N0,N1], Q, R )\n"
          << "[N0, N1] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "Q: ";
    Q.print_info(ostrm);
    ostrm << "R: ";
    R.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

//! Test for TensorC::qr (Matrix version)
/*! \code Q, R = np.linalg.qr(A,mode='reduced') \endcode

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L*(L+1), L*L)
  \param ostrm output stream for results
*/
void test_qr_rank2_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L * (L + 1);
  int N1 = L * L;

  TensorC A(Shape(N0, N1));

  Shape shape = A.shape();
  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val = cfunc2_1(index, shape);
    A.set_value(index, val);
  }

  TensorC Q, R;

  time0.now();

  // Q, R = np.linalg.qr(A,mode='reduced')
  qr(A, Q, R);

  time1.now();

  /* Check */
  TensorC B = tensordot(Q, R, Axes(1), Axes(0));
  // now B[i,j] = A[i,j]

  double error = 0.0;
  Index index;
  index.resize(4);
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val;
    B.get_value(index, val);

    complex exact = cfunc2_1(index, shape);

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "QR <complex> ( A[N0,N1], Q, R )\n"
          << "[N0, N1] = " << A.shape() << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "Q: ";
    Q.print_info(ostrm);
    ostrm << "R: ";
    R.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
