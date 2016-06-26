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


//! Test for TensorD::contract (partial trace)
/*! B = contract(A, Axes(0), Axes(2)),
  \f$ B_{ab} = \sum_{i} A_{iaib} \f$

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L, L+2)
  \param ostrm output stream for results
*/
void test_contract(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2, time3, time4;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L+1;
  int N2 = L+2;

  TensorD A(Shape(N0, N1, N0, N2));
  Shape shape_A = A.shape();

  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    double val = func4_1(index, shape_A);
    A.set_value(index, val);
  }

  time0 = MPI_Wtime();

  TensorD B = contract(A, 0, 2);

  time1 = MPI_Wtime();

  double error = 0.0;
  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    double val;
    B.get_value(index, val);

    double exact=0.0;
    for(size_t m=0;m<N0;++m) {
      exact += func4_1(Index(m,index[0],m,index[1]), shape_A);
    }

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if(mpiroot) {
    ostrm << "========================================\n"
          << "Contract <double> ( A[N0, N1, N0, N2], Axes(0), Axes(2) )\n"
          << "[N0, N1, N0, N2] = " <<  A.shape() << "\n"
          << "Error=          " << max_error << "\n"
          << "Time=           " << time1-time0 << " [sec]\n"
          << "Time(check)=    " << time2-time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


//! Test for TensorC::contract (partial trace)
/*! B = contract(A, Axes(0), Axes(2)),
  \f$ B_{ab} = \sum_{i} A_{iaib} \f$

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L, L+2)
  \param ostrm output stream for results
*/
void test_contract_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2, time3, time4;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L+1;
  int N2 = L+2;

  TensorC A(Shape(N0, N1, N0, N2));
  Shape shape_A = A.shape();

  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    complex val = cfunc4_1(index, shape_A);
    A.set_value(index, val);
  }

  time0 = MPI_Wtime();

  TensorC B = contract(A, 0, 2);

  time1 = MPI_Wtime();

  double error = 0.0;
  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    complex val;
    B.get_value(index, val);

    complex exact=0.0;
    for(size_t m=0;m<N0;++m) {
      exact += cfunc4_1(Index(m,index[0],m,index[1]), shape_A);
    }

    if(error < std::abs(val-exact) ) error = std::abs(val-exact);
  }

  time2 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if(mpiroot) {
    ostrm << "========================================\n"
          << "Contract <complex> ( A[N0, N1, N0, N2], Axes(0), Axes(2) )\n"
          << "[N0, N1, N0, N2] = " <<  A.shape() << "\n"
          << "Error=          " << max_error << "\n"
          << "Time=           " << time1-time0 << " [sec]\n"
          << "Time(check)=    " << time2-time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  MPI_Barrier(comm);
}


} // namespace tests
