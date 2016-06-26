/*
  Apr. 24, 2015
  Copyright (C) 2015 Satoshi Morita
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


//! Test for TensorD::eigh
/*! A[i,j] => Contract( Z[i,a] * W[a] * (Z[a,j])^t )

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L*L, L*L)
  \param ostrm output stream for results
*/
void test_eigh_rank2(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t N = L*L;

  TensorD A(Shape(N, N));

  Shape shape = A.shape();
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    Index index_r(index[1],index[0]);
    double val = func2_1(index, shape);
    val += func2_1(index_r, shape);
    A.set_value(index, val);
  }

  TensorD Z;
  std::vector<double> W;

  time0 = MPI_Wtime();

  // A[i,j] => Contract( Z[i,a] * W[a] * (Z[a,j])^t )
  // Z^t A Z = diag(W)
  eigh(A, W, Z);

  time1 = MPI_Wtime();

  /* Check */
  TensorD B = Z;
  Z.multiply_vector(W, 1); // Z[i,a] <= Z[i,a] * W[a]
  B = tensordot(Z,B,Axes(1),Axes(1));
  // now B[i,j] = A[i,j]

  double error = 0.0;
  Index index;
  index.resize(4);
  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    double val;
    B.get_value(index,val);

    double exact;
    A.get_value(index, exact);

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "eigh <double> ( A[N,N], W, Z )\n"
              << "[N, N] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "Z: ";
    Z.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    int n = (W.size() < 5) ? W.size() : 5;
    for(int i=0;i<n;++i) {
      ostrm << "S[" << i << "]= " << W[i] << "\n";
    }
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


//! Test for TensorC::eigh
/*! A[i,j] => Contract( Z[i,a] * W[a] * conj(Z[a,j]^t) )

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L*L, L*L)
  \param ostrm output stream for results
*/
void test_eigh_rank2_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t N = L*L;

  TensorC A(Shape(N, N));

  Shape shape = A.shape();
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    Index index_r(index[1],index[0]);
    complex val = cfunc2_1(index, shape);
    val += conj(cfunc2_1(index_r, shape));
    A.set_value(index, val);
  }

  TensorC Z;
  std::vector<double> W;

  time0 = MPI_Wtime();

  // A[i,j] => Contract( Z[i,a] * W[a] * (Z[a,j])^t )
  // Z^t A Z = diag(W)
  eigh(A, W, Z);

  time1 = MPI_Wtime();

  /* Check */
  TensorC B = conj(Z);
  Z.multiply_vector(W, 1); // Z[i,a] <= Z[i,a] * W[a]
  B = tensordot(Z,B,Axes(1),Axes(1));
  // now B[i,j] = A[i,j]

  double error = 0.0;
  Index index;
  index.resize(4);
  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    complex val;
    B.get_value(index,val);

    complex exact;
    A.get_value(index, exact);

    if(error < std::abs(val-exact) ) error = std::abs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "eigh <complex> ( A[N,N], W, Z )\n"
              << "[N, N] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "Z: ";
    Z.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    int n = (W.size() < 5) ? W.size() : 5;
    for(int i=0;i<n;++i) {
      ostrm << "S[" << i << "]= " << W[i] << "\n";
    }
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


} // namespace tests
