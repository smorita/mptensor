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
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code for tensordot
*/

#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>

#include <mptensor.hpp>
#include "mpi_tool.hpp"
#include "functions.hpp"
#include "typedef.hpp"

namespace tests {


//! Test for TensorD::tensordot
/*! C = tensordot(A, B, axes=([1,3],[2,0]))

  This test may take a long time for check.

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+4, L+1, L+5), B.shape = (L+5, L+2, L+4, L+3)
  \param ostrm output stream for results
*/
void test_tensordot(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L+1;
  int N2 = L+2;
  int N3 = L+3;
  int M = L+4;
  int K = L+5;

  TensorD A(Shape(N0, M, N1, K));
  TensorD B(Shape(K, N2, M, N3));

  Shape shape_A = A.shape();
  Shape shape_B = B.shape();

  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    double val = func4_1(index, shape_A);
    A.set_value(index, val);
  }

  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    double val = func4_2(index, shape_B);
    B.set_value(index, val);
  }

  time0 = MPI_Wtime();

  // C = tensordot(A, B, axes=([1,3],[2,0]))
  TensorD C = tensordot(A,B,Axes(1,3),Axes(2,0));

  time1 = MPI_Wtime();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(4);
  index_B.resize(4);
  for(size_t i=0;i<C.local_size();++i) {
    Index index = C.global_index(i);
    double val;
    C.get_value(index, val);

    double exact=0;
    index_A[0] = index[0];
    index_A[2] = index[1];
    index_B[1] = index[2];
    index_B[3] = index[3];
    for(int m=0;m<M;++m) {
      for(int k=0;k<K;++k) {
        index_A[1] = m;
        index_A[3] = k;
        index_B[2] = m;
        index_B[0] = k;
        exact += func4_1(index_A, shape_A) * func4_2(index_B, shape_B);
      }
    }

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "tensordot <double> ( C[N0, N1, N2, N3] = A[N0, M, N1, K ] * B[K, N2, M, N3] )\n"
              << "[N0, N1, N2, N3] = [" << N0 <<", "<< N1 <<", "<< N2 <<", "<< N3 <<"] "
              << "M= " << M << " K= " << K << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
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
  MPI_Barrier(comm);
}


//! Test for TensorC::tensordot
/*! C = tensordot(A, B, axes=([1,3],[2,0]))

  This test may take a long time for check.

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+4, L+1, L+5), B.shape = (L+5, L+2, L+4, L+3)
  \param ostrm output stream for results
*/
void test_tensordot_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L+1;
  int N2 = L+2;
  int N3 = L+3;
  int M = L+4;
  int K = L+5;

  TensorC A(Shape(N0, M, N1, K));
  TensorC B(Shape(K, N2, M, N3));

  Shape shape_A = A.shape();
  Shape shape_B = B.shape();

  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    complex val = cfunc4_1(index, shape_A);
    A.set_value(index, val);
  }

  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    complex val = cfunc4_2(index, shape_B);
    B.set_value(index, val);
  }

  time0 = MPI_Wtime();

  // C = tensordot(A, B, axes=([1,3],[2,0]))
  TensorC C = tensordot(A,B,Axes(1,3),Axes(2,0));

  time1 = MPI_Wtime();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(4);
  index_B.resize(4);

  for(size_t i=0;i<C.local_size();++i) {
    Index index = C.global_index(i);
    complex val;
    C.get_value(index, val);

    complex exact=0;
    index_A[0] = index[0];
    index_A[2] = index[1];
    index_B[1] = index[2];
    index_B[3] = index[3];
    for(int m=0;m<M;++m) {
      for(int k=0;k<K;++k) {
        index_A[1] = m;
        index_A[3] = k;
        index_B[2] = m;
        index_B[0] = k;
        exact += cfunc4_1(index_A, shape_A) * cfunc4_2(index_B, shape_B);
      }
    }

    if(error < std::abs(val-exact) ) error = std::abs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "tensordot <complex> ( C[N0, N1, N2, N3] = A[N0, M, N1, K ] * B[K, N2, M, N3] )\n"
              << "[N0, N1, N2, N3] = [" << N0 <<", "<< N1 <<", "<< N2 <<", "<< N3 <<"] "
              << "M= " << M << " K= " << K << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
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
  MPI_Barrier(comm);
}


} // namespace tests
