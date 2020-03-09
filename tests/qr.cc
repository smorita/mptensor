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
  \file   qr.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code for QR decomposition
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

//! Test for TensorD::qr
/*! A[i,j,k,l] => Contract( Q[k,i,a] * R[a,j,l])

  This is equivalent to the following numpy code.

  \code
  AT = np.transpose(AT, (2,0,1,3))
  shape = AT.shape
  AT.reshape( (shape[0]*shape[1], shape[2]*shape[3]) )
  Q, R = linalg.qr(AT, mode='reduced')
  Q.reshape( (shape[0], shape[1], Q.shape[2]) )
  R.reshape( (R.shape[0], shape[2], shape[3]) )
  \endcode

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_qr(const mpi_comm &comm, int L, std::ostream &ostrm) {
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

  TensorD Q, R;

  time0.now();

  // QR decomposition
  qr(A, Axes(2, 0), Axes(1, 3), Q, R);

  time1.now();

  /* Check */
  TensorD B = transpose(tensordot(Q, R, Axes(2), Axes(0)), Axes(1, 2, 0, 3));
  // now B[i,j,k,l] = A[i,j,k,l]

  double error = 0.0;
  Shape shape_B = B.shape();
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val;
    B.get_value(index, val);

    double exact = func4_1(index, shape_B);

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "QR <double> ( A[N0,N1,N2,N3], Axes(2,0), Axes(1,3), Q, R )\n"
          << "[N0, N1, N2, N3] = " << A.shape() << "\n"
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

//! Test for TensorC::qr
/*! A[i,j,k,l] => Contract( Q[k,i,a] * R[a,j,l])

  This is equivalent to the following numpy code.

  \code
  AT = np.transpose(AT, (2,0,1,3))
  shape = AT.shape
  AT.reshape( (shape[0]*shape[1], shape[2]*shape[3]) )
  Q, R = linalg.qr(AT, mode='reduced')
  Q.reshape( (shape[0], shape[1], Q.shape[2]) )
  R.reshape( (R.shape[0], shape[2], shape[3]) )
  \endcode

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_qr_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
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

  TensorC Q, R;

  time0.now();

  // QR decomposition
  qr(A, Axes(2, 0), Axes(1, 3), Q, R);

  time1.now();

  /* Check */
  TensorC B = transpose(tensordot(Q, R, Axes(2), Axes(0)), Axes(1, 2, 0, 3));
  // now B[i,j,k,l] = A[i,j,k,l]

  double error = 0.0;
  Shape shape_B = B.shape();
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val;
    B.get_value(index, val);

    complex exact = cfunc4_1(index, shape_B);

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);
  if (mpiroot) {
    ostrm << "========================================\n"
          << "QR <complex> ( A[N0,N1,N2,N3], Axes(2,0), Axes(1,3), Q, R )\n"
          << "[N0, N1, N2, N3] = " << A.shape() << "\n"
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
