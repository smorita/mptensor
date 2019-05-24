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
  \file   tensor_test.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   January 14 2015
  \brief  Test code for mptensor
*/

#ifndef _TENSOR_TEST_HPP_
#define _TENSOR_TEST_HPP_

#include <iostream>
#include <mpi.h>

//! Test codes for Tensor
namespace tests {

void test_transpose(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_reshape(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_slice(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_set_slice(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_tensordot(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_svd(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_qr(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_qr_rank2(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_eigh(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_eigh_rank2(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_arithmetic(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_trace(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_trace2(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_contract(const MPI_Comm &comm, int L, std::ostream &ostrm);

void test_transpose_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_reshape_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_slice_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_set_slice_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_tensordot_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_svd_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_qr_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_qr_rank2_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_eigh_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_eigh_rank2_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_arithmetic_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_trace_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_trace2_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);
void test_contract_complex(const MPI_Comm &comm, int L, std::ostream &ostrm);

} // namespace tests

#endif // _TENSOR_TEST_HPP_
