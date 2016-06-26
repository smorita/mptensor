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


//! Test for TensorD::transpose
/*! B = transpose(A, (2,0,3,1))

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_transpose(const MPI_Comm &comm, int L, std::ostream &ostrm) {
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

  time0 = MPI_Wtime();

  // B = transpose(A, (2,0,3,1))
  TensorD B = transpose(A, Axes(2,0,3,1));

  time1 = MPI_Wtime();

  double error = 0.0;
  Index index;
  index.resize(4);
  for(size_t i=0;i<B.local_size();++i) {
    Index index_B = B.global_index(i);
    double val;
    B.get_value(index_B, val);

    index[0] = index_B[1];
    index[1] = index_B[3];
    index[2] = index_B[0];
    index[3] = index_B[2];

    double exact = func4_1(index, shape_A);

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "transpose <double> (A[N0,N1,N2,N3], Axes(2,0,3,1)) = B[N2,N0,N3,N1]\n"
              << "[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


//! Test for TensorC::transpose
/*! B = transpose(A, (2,0,3,1))

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_transpose_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
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

  time0 = MPI_Wtime();

  // B = transpose(A, (2,0,3,1))
  TensorC B = transpose(A, Axes(2,0,3,1));

  time1 = MPI_Wtime();

  double error = 0.0;
  Index index;
  index.resize(4);
  for(size_t i=0;i<B.local_size();++i) {
    Index index_B = B.global_index(i);
    complex val;
    B.get_value(index_B, val);

    index[0] = index_B[1];
    index[1] = index_B[3];
    index[2] = index_B[0];
    index[3] = index_B[2];

    complex exact = cfunc4_1(index, shape_A);

    if(error < std::abs(val-exact) ) error = std::abs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "transpose <complex> (A[N0,N1,N2,N3], Axes(2,0,3,1)) = B[N2,N0,N3,N1]\n"
              << "[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


} // namespace tests
