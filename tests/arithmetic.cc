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


//! Test for arithmetic operators
/*!
  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_arithmetic(const MPI_Comm &comm, int L, std::ostream &ostrm) {
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
  TensorD B(Shape(N0, N1, N2, N3));
  TensorD C;

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

  time1 = MPI_Wtime();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(4);
  index_B.resize(4);
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    double val;
    A.get_value(index, val);

    double exact = -1.8*func4_1(index, shape_A) - 1.7*func4_2(index, shape_B);

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "Arithmetic Operators <double> ( A[N0, N1, N2, N3] )\n"
              << "[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


//! Test for arithmetic operators (complex version)
/*!
  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_arithmetic_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
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
  TensorC B(Shape(N0, N1, N2, N3));
  TensorC C;

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

  time1 = MPI_Wtime();

  double error = 0.0;
  Index index_A;
  Index index_B;
  index_A.resize(4);
  index_B.resize(4);
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    complex val;
    A.get_value(index, val);

    complex exact = complex(1.8, -3.6)*cfunc4_1(index, shape_A) + complex(2.15, -3.3)*cfunc4_2(index, shape_B);

    if(error < abs(val-exact) ) error = abs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "Arithmetic Operators <complex> ( A[N0, N1, N2, N3] )\n"
              << "[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time= " << time1-time0 << " [sec]\n"
              << "Time(check)= " << time2-time1 << " [sec]\n"
              << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


} // namespace tests
