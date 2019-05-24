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
  \file   svd.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code for singular value decomposition
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


//! Test for TensorD::svd
/*! A[i,j,k,l] => Contract( U[k,i,a] * S[a] * V[a,j,l])

  This is equivalent to the following numpy code.

  \code
  A = np.transpose(A, (2,0,1,3))
  shape = A.shape
  A.reshape( (shape[0]*shape[1], shape[2]*shape[3]) )
  U, S, V = linalg.svd(A, full_matrix=False)
  U.reshape( (shape[0],shape[1],len(S)) )
  V.reshape( (len(S),shape[2],shape[3]) )
  \endcode

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_svd(const MPI_Comm &comm, int L, std::ostream &ostrm) {
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

  TensorD A(Shape(N0, N1, N2, N3));

  Shape shape_A = A.shape();
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    double val = func4_1(index, shape_A);
    A.set_value(index, val);
  }

  TensorD U, V;
  std::vector<double> S;

  time0 = MPI_Wtime();

  // A[i,j,k,l] => Contract( U[k,i,a] * S[a] * V[a,j,l])
  svd(A, Axes(2,0), Axes(1,3), U, S, V);

  time1 = MPI_Wtime();

  /* Check */
  U.multiply_vector(S, 2); // U[k,i,a] <= U[k,i,a] * S[a]
  TensorD B = tensordot(U,V,Axes(2),Axes(0));
  // now B[k,i,j,l] = A[i,j,k,l]

  double error = 0.0;
  Index index;
  index.resize(4);
  for(size_t i=0;i<B.local_size();++i) {
    Index index_B = B.global_index(i);
    double val;
    B.get_value(index_B,val);

    index[0] = index_B[1];
    index[1] = index_B[2];
    index[2] = index_B[0];
    index[3] = index_B[3];
    double exact = func4_1(index, shape_A);

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "SVD <double> ( A[N0,N1,N2,N3], Axes(2,0), Axes(1,3), U, S, V )\n"
              << "[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "U: ";
    U.print_info(ostrm);
    ostrm << "V: ";
    V.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    int n = (S.size() < 5) ? S.size() : 5;
    for(int i=0;i<n;++i) {
      ostrm << "S[" << i << "]= " << S[i] << "\n";
    }
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


//! Test for TensorC::svd
/*! A[i,j,k,l] => Contract( U[k,i,a] * S[a] * V[a,j,l])

  This is equivalent to the following numpy code.

  \code
  A = np.transpose(A, (2,0,1,3))
  shape = A.shape
  A.reshape( (shape[0]*shape[1], shape[2]*shape[3]) )
  U, S, V = linalg.svd(A, full_matrix=False)
  U.reshape( (shape[0],shape[1],len(S)) )
  V.reshape( (len(S),shape[2],shape[3]) )
  \endcode

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_svd_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
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

  TensorC A(Shape(N0, N1, N2, N3));

  Shape shape_A = A.shape();
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    complex val = cfunc4_1(index, shape_A);
    A.set_value(index, val);
  }

  TensorC U, V;
  std::vector<double> S;

  time0 = MPI_Wtime();

  // A[i,j,k,l] => Contract( U[k,i,a] * S[a] * V[a,j,l])
  svd(A, Axes(2,0), Axes(1,3), U, S, V);

  time1 = MPI_Wtime();

  /* Check */
  U.multiply_vector(S, 2); // U[k,i,a] <= U[k,i,a] * S[a]
  TensorC B = tensordot(U,V,Axes(2),Axes(0));
  // now B[k,i,j,l] = A[i,j,k,l]

  double error = 0.0;
  Index index;
  index.resize(4);
  for(size_t i=0;i<B.local_size();++i) {
    Index index_B = B.global_index(i);
    complex val;
    B.get_value(index_B,val);

    index[0] = index_B[1];
    index[1] = index_B[2];
    index[2] = index_B[0];
    index[3] = index_B[3];
    complex exact = cfunc4_1(index, shape_A);

    if(error < std::abs(val-exact) ) error = std::abs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "SVD <complex> ( A[N0,N1,N2,N3], Axes(2,0), Axes(1,3), U, S, V )\n"
              << "[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "U: ";
    U.print_info(ostrm);
    ostrm << "V: ";
    V.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    int n = (S.size() < 5) ? S.size() : 5;
    for(int i=0;i<n;++i) {
      ostrm << "S[" << i << "]= " << S[i] << "\n";
    }
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


} // namespace tests
