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


//! Test for TensorD::set_slice
/*! A[:, 3:6, :, :] = B and A[1:4, 0:2, :, 2:5] = C

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_set_slice(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2, time3;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L+1;
  int N2 = L+2;
  int N3 = L+3;

  if (N0<4) N0 = 4;
  if (N1<6) N1 = 6;
  if (N3<5) N3 = 5;

  TensorD A(Shape(N0, N1, N2, N3));
  TensorD B(Shape(N0, 3, N2, N3));
  TensorD C(Shape(3, 2, N2, 3));

  A = 1.0;

  Shape shape_B = B.shape();
  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    double val = func4_1(index, shape_B);
    B[i] = val;
  }

  Shape shape_C = C.shape();
  for(size_t i=0;i<C.local_size();++i) {
    Index index = C.global_index(i);
    double val = func4_2(index, shape_C);
    C[i] = val;
  }

  time0 = MPI_Wtime();

  // A[:, 3:6, :, :] = B
  A.set_slice(B, 1, 3, 6);

  time1 = MPI_Wtime();

  // A[1:4, 0:2, :, 2:5] = C
  A.set_slice(C, Index(1,0,0,2), Index(4,2,0,5));

  time2 = MPI_Wtime();

  double error = 0.0;
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    double val, exact;
    A.get_value(index, val);

    if(index[1]>=3 && index[1]<6) {
      index[1] -= 3;
      exact = func4_1(index, shape_B);
    } else if(index[0]>=1 && index[0]<4 &&
              index[1]>=0 && index[1]<2 &&
              index[3]>=2 && index[3]<5) {
      index[0] -= 1;
      index[3] -= 2;
      exact = func4_2(index, shape_C);
    } else {
      exact = 1.0;
    }

    if(error < fabs(val-exact) ) error = fabs(val-exact);
  }

  time3 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if(mpiroot) {
    ostrm << "========================================\n"
              << "set_slice <double> A.set_slice(B, 1, 3, 6)\n"
              << "set_slice <double> A.set_slice(C, Index(1,0,0,2), Index(4,2,0,5))\n"
              << "A[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time(B)= " << time1-time0 << " [sec]\n"
              << "Time(C)= " << time2-time1 << " [sec]\n"
              << "Time(check)= " << time3-time2 << " [sec]\n"
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


//! Test for TensorC::slice
/*! A[:, 3:6, :, :] = B and A[1:4, 0:2, :, 2:5] = C

  \param comm MPI communicator
  \param L size of tensor A.shape = (L, L+1, L+2, L+3)
  \param ostrm output stream for results
*/
void test_set_slice_complex(const MPI_Comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-10;
  double time0, time1, time2, time3;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  int N0 = L;
  int N1 = L+1;
  int N2 = L+2;
  int N3 = L+3;

  if (N0<4) N0 = 4;
  if (N1<6) N1 = 6;
  if (N3<5) N3 = 5;

  TensorC A(Shape(N0, N1, N2, N3));
  TensorC B(Shape(N0, 3, N2, N3));
  TensorC C(Shape(3, 2, N2, 3));

  A = 1.0;

  Shape shape_B = B.shape();
  for(size_t i=0;i<B.local_size();++i) {
    Index index = B.global_index(i);
    complex val = cfunc4_1(index, shape_B);
    B[i] = val;
  }

  Shape shape_C = C.shape();
  for(size_t i=0;i<C.local_size();++i) {
    Index index = C.global_index(i);
    complex val = cfunc4_2(index, shape_C);
    C[i] = val;
  }

  time0 = MPI_Wtime();

  // A[:, 3:6, :, :] = B
  A.set_slice(B, 1, 3, 6);

  time1 = MPI_Wtime();

  // A[1:4, 0:2, :, 2:5] = C
  A.set_slice(C, Index(1,0,0,2), Index(4,2,0,5));

  time2 = MPI_Wtime();

  double error = 0.0;
  for(size_t i=0;i<A.local_size();++i) {
    Index index = A.global_index(i);
    complex val, exact;
    A.get_value(index, val);

    if(index[1]>=3 && index[1]<6) {
      index[1] -= 3;
      exact = cfunc4_1(index, shape_B);
    } else if(index[0]>=1 && index[0]<4 &&
              index[1]>=0 && index[1]<2 &&
              index[3]>=2 && index[3]<5) {
      index[0] -= 1;
      index[3] -= 2;
      exact = cfunc4_2(index, shape_C);
    } else {
      exact = 1.0;
    }

    if(error < std::abs(val-exact) ) error = std::abs(val-exact);
  }

  time3 = MPI_Wtime();

  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  if(mpiroot) {
    ostrm << "========================================\n"
              << "set_slice <complex> A.set_slice(B, 1, 3, 6)\n"
              << "set_slice <complex> A.set_slice(C, Index(1,0,0,2), Index(4,2,0,5))\n"
              << "A[N0, N1, N2, N3] = " <<  A.shape() << "\n"
              << "Error= " << max_error << "\n"
              << "Time(B)= " << time1-time0 << " [sec]\n"
              << "Time(C)= " << time2-time1 << " [sec]\n"
              << "Time(check)= " << time3-time2 << " [sec]\n"
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
