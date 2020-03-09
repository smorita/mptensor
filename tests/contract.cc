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
  \file   contract.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   April 24 2015
  \brief  Test code of tensor contraction
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

//! Test for TensorD::contract (partial trace)
/*! B = contract(A, Axes(0), Axes(2)),
  \f$ B_{ab} = \sum_{i} A_{iaib} \f$

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L, L+2)
  \param ostrm output stream for results
*/
void test_contract(const mpi_comm &comm, int L, std::ostream &ostrm) {
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

  TensorD A(Shape(N0, N1, N0, N2));
  Shape shape_A = A.shape();

  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    double val = func4_1(index, shape_A);
    A.set_value(index, val);
  }

  time0.now();

  TensorD B = contract(A, 0, 2);

  time1.now();

  double error = 0.0;
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    double val;
    B.get_value(index, val);

    double exact = 0.0;
    for (size_t m = 0; m < N0; ++m) {
      exact += func4_1(Index(m, index[0], m, index[1]), shape_A);
    }

    if (error < fabs(val - exact)) error = fabs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);

  if (mpiroot) {
    ostrm << "========================================\n"
          << "Contract <double> ( A[N0, N1, N0, N2], Axes(0), Axes(2) )\n"
          << "[N0, N1, N0, N2] = " << A.shape() << "\n"
          << "Error=          " << max_error << "\n"
          << "Time=           " << time1 - time0 << " [sec]\n"
          << "Time(check)=    " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

//! Test for TensorC::contract (partial trace)
/*! B = contract(A, Axes(0), Axes(2)),
  \f$ B_{ab} = \sum_{i} A_{iaib} \f$

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L, L+1, L, L+2)
  \param ostrm output stream for results
*/
void test_contract_complex(const mpi_comm &comm, int L, std::ostream &ostrm) {
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

  TensorC A(Shape(N0, N1, N0, N2));
  Shape shape_A = A.shape();

  for (size_t i = 0; i < A.local_size(); ++i) {
    Index index = A.global_index(i);
    complex val = cfunc4_1(index, shape_A);
    A.set_value(index, val);
  }

  time0.now();

  TensorC B = contract(A, 0, 2);

  time1.now();

  double error = 0.0;
  for (size_t i = 0; i < B.local_size(); ++i) {
    Index index = B.global_index(i);
    complex val;
    B.get_value(index, val);

    complex exact = 0.0;
    for (size_t m = 0; m < N0; ++m) {
      exact += cfunc4_1(Index(m, index[0], m, index[1]), shape_A);
    }

    if (error < std::abs(val - exact)) error = std::abs(val - exact);
  }

  time2.now();

  double max_error = mpi_reduce_max(error, comm);

  if (mpiroot) {
    ostrm << "========================================\n"
          << "Contract <complex> ( A[N0, N1, N0, N2], Axes(0), Axes(2) )\n"
          << "[N0, N1, N0, N2] = " << A.shape() << "\n"
          << "Error=          " << max_error << "\n"
          << "Time=           " << time1 - time0 << " [sec]\n"
          << "Time(check)=    " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "========================================" << std::endl;
  }
  assert(error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
