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
  \file   eigh_general.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   July 4 2019
  \brief  Test code for eigenvalue decomposition
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

//! Test for TensorD::eigh (generalized eigenvalue problem)
/*! A[i,j,k,l] Z[l,k,a] = B[j,l,i,k] Z[l,k,a] W[a]

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L,L,L,L)
  \param ostrm output stream for results
*/
void test_eigh_general(const mpi_comm &comm, int L, std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-8;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t N = L * L;

  TensorD A(Shape(L, L, L, L));
  TensorD B(Shape(L, L, L, L));

  {
    Shape shape = A.shape();
    for (size_t i = 0; i < A.local_size(); ++i) {
      Index index = A.global_index(i);
      Index index2(index[2], index[3], index[0], index[1]);
      double val = func4_1(index, shape) + func4_1(index2, shape);
      A.set_value(index, val);
    }
  }

  {
    Shape shape = B.shape();
    for (size_t i = 0; i < B.local_size(); ++i) {
      Index index = B.global_index(i);
      double val = func4_2(index, shape);
      B.set_value(index, val);
    }
    TensorD u, vt;
    std::vector<double> s;
    svd(B, Axes(0, 1), Axes(2, 3), u, s, vt);
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] < 1.0e-3) {
        s[i] = 1.0e-3 * (i + 1);
      }
    }
    TensorD us = u;
    us.multiply_vector(s, 2);
    B = transpose(tensordot(us, u, Axes(2), Axes(2)), Axes(1, 3, 0, 2));
  }

  TensorD Z;
  std::vector<double> W;

  time0.now();

  // A[i,j,k,l] Z[l,k,a] = B[j,l,i,k] Z[l,k,a] W[a]
  eigh(A, Axes(1, 0), Axes(3, 2), B, Axes(0, 2), Axes(1, 3), W, Z);

  time1.now();

  /* Check */
  A = tensordot(A.transpose(Axes(1, 0, 3, 2)), Z, Axes(2, 3), Axes(0, 1));
  B = tensordot(B.transpose(Axes(0, 2, 1, 3)), Z, Axes(2, 3), Axes(0, 1));
  B.multiply_vector(W, 2);
  double max_error = max_abs(A - B);

  time2.now();

  if (mpiroot) {
    ostrm << "========================================\n"
          << "eigh <double> (A, Axes(1, 0), Axes(3, 2), B, Axes(0, 2), Axes(1, "
             "3), W, Z)\n"
          << "[L,L,L,L] = " << Shape(L, L, L, L) << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "Z: ";
    Z.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    if (W.size() < 7) {
      for (int i = 0; i < W.size(); ++i) {
        ostrm << "S[" << i << "]= " << W[i] << "\n";
      }
    } else {
      for (int i = 0; i < 3; ++i) {
        ostrm << "S[" << i << "]= " << W[i] << "\n";
      }
      for (int i = W.size() - 3; i < W.size(); ++i) {
        ostrm << "S[" << i << "]= " << W[i] << "\n";
      }
    }
    ostrm << "========================================" << std::endl;
  }
  assert(max_error < EPS);
  mpi_barrier(comm);
}

//! Test for TensorC::eigh
/*! A[i,j,k,l] => Contract( Z[i,k,a] * W[a] * conj(Z[a,l,j]^t) )

  \param comm MPI communicator
  \param L size of tensor, A.shape = (L,L,L,L)
  \param ostrm output stream for results
*/
void test_eigh_general_complex(const mpi_comm &comm, int L,
                               std::ostream &ostrm) {
  using namespace mptensor;
  const double EPS = 1.0e-8;
  Timer time0, time1, time2;
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t N = L * L;

  TensorC A(Shape(L, L, L, L));
  TensorC B(Shape(L, L, L, L));

  {
    Shape shape = A.shape();
    for (size_t i = 0; i < A.local_size(); ++i) {
      Index index = A.global_index(i);
      Index index2(index[2], index[3], index[0], index[1]);
      complex val = cfunc4_2(index, shape) + conj(cfunc4_2(index2, shape));
      A.set_value(index, val);
    }
  }

  {
    Shape shape = B.shape();
    for (size_t i = 0; i < B.local_size(); ++i) {
      Index index = B.global_index(i);
      complex val = cfunc4_1(index, shape);
      B.set_value(index, val);
    }
    TensorC u, vt;
    std::vector<double> s;
    svd(B, Axes(0, 1), Axes(2, 3), u, s, vt);
    for (size_t i = 0; i < s.size(); ++i) {
      if (s[i] < 1.0e-3) {
        s[i] = 1.0e-3 * (i + 1);
      }
    }
    TensorC us = u;
    us.multiply_vector(s, 2);
    B = transpose(tensordot(us, conj(u), Axes(2), Axes(2)), Axes(1, 3, 0, 2));
  }

  TensorC Z;
  std::vector<double> W;

  time0.now();

  // A[i,j,k,l] Z[l,k,a] = B[j,l,i,k] Z[l,k,a] W[a]
  eigh(A, Axes(1, 0), Axes(3, 2), B, Axes(0, 2), Axes(1, 3), W, Z);

  time1.now();

  /* Check */
  A = tensordot(A.transpose(Axes(1, 0, 3, 2)), Z, Axes(2, 3), Axes(0, 1));
  B = tensordot(B.transpose(Axes(0, 2, 1, 3)), Z, Axes(2, 3), Axes(0, 1));
  B.multiply_vector(W, 2);
  double max_error = max_abs(A - B);

  time2.now();

  if (mpiroot) {
    ostrm << "========================================\n"
          << "eigh <complex> (A, Axes(1, 0), Axes(3, 2), B, Axes(0, 2), "
             "Axes(1, 3), W, Z)\n"
          << "[L,L,L,L] = " << Shape(L, L, L, L) << "\n"
          << "Error= " << max_error << "\n"
          << "Time= " << time1 - time0 << " [sec]\n"
          << "Time(check)= " << time2 - time1 << " [sec]\n"
          << "----------------------------------------\n";
    ostrm << "A: ";
    A.print_info(ostrm);
    ostrm << "B: ";
    B.print_info(ostrm);
    ostrm << "Z: ";
    Z.print_info(ostrm);
    ostrm << "----------------------------------------\n";
    if (W.size() < 7) {
      for (int i = 0; i < W.size(); ++i) {
        ostrm << "S[" << i << "]= " << W[i] << "\n";
      }
    } else {
      for (int i = 0; i < 3; ++i) {
        ostrm << "S[" << i << "]= " << W[i] << "\n";
      }
      for (int i = W.size() - 3; i < W.size(); ++i) {
        ostrm << "S[" << i << "]= " << W[i] << "\n";
      }
    }
    ostrm << "========================================" << std::endl;
  }
  assert(max_error < EPS);
  mpi_barrier(comm);
}

}  // namespace tests
