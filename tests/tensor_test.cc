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
  \file   tensor_test.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   January 14 2015
  \brief  Test code for mptensor
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "mpi_tool.hpp"
#include "tensor_test.hpp"

/* Main function */
int main(int argc, char **argv) {
  using namespace tests;
  /* Start */
  mpi_init(argc, argv);
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_comm comm = MPI_COMM_WORLD;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpiroot)
      std::cerr << "Usage: a.out N\n"
                << "Warning: assuming N=10" << std::endl;
    n = 10;
  } else {
    n = atoi(argv[1]);
  }

  // std::ofstream fout;
  // char filename[1024];
  // sprintf(filename,"result_n%04d.dat",n);
  // if(mpiroot) fout.open(filename);
  // std::ostream out(fout.rdbuf());

  std::ostream out(std::cout.rdbuf());

  test_transpose(comm, n, out);
  test_reshape(comm, n, out);
  test_slice(comm, n, out);
  test_set_slice(comm, n, out);
  test_tensordot(comm, n, out);
  test_svd(comm, n, out);
  test_qr(comm, n, out);
  test_qr_rank2(comm, n, out);
  test_eigh(comm, n, out);
  test_eigh_rank2(comm, n, out);
  // test_eigh_general(comm, n, out);
  test_arithmetic(comm, n, out);
  test_trace(comm, n, out);
  test_trace2(comm, n, out);
  test_contract(comm, n, out);
  test_kron(comm, n, out);

  if (mpiroot) out << "\n";

  test_transpose_complex(comm, n, out);
  test_reshape_complex(comm, n, out);
  test_slice_complex(comm, n, out);
  test_set_slice_complex(comm, n, out);
  test_tensordot_complex(comm, n, out);
  test_svd_complex(comm, n, out);
  test_qr_complex(comm, n, out);
  test_qr_rank2_complex(comm, n, out);
  test_eigh_complex(comm, n, out);
  test_eigh_rank2_complex(comm, n, out);
  // test_eigh_general_complex(comm, n, out);
  test_arithmetic_complex(comm, n, out);
  test_trace_complex(comm, n, out);
  test_trace2_complex(comm, n, out);
  test_contract_complex(comm, n, out);
  test_kron_complex(comm, n, out);

  /* End */
  /* automatically called by std::atexit() in mpi_tool. */
  // MPI_Finalize();
}
