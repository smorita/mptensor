/*
  Jan. 14, 2015
  Copyright (C) 2015 Satoshi Morita
 */

#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

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
  mpi_info(MPI_COMM_WORLD, mpirank, mpisize, mpiroot);

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpiroot) std::cerr << "Usage: a.out N\n"
                           << "waring: assuming N=10" << std::endl;
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

  MPI_Comm comm = MPI_COMM_WORLD;

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
  test_arithmetic(comm, n, out);
  test_trace(comm, n, out);
  test_trace2(comm, n, out);
  test_contract(comm, n, out);

  if(mpiroot) out << "\n";

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
  test_arithmetic_complex(comm, n, out);
  test_trace_complex(comm, n, out);
  test_trace2_complex(comm, n, out);
  test_contract_complex(comm, n, out);

  /* End */
  /* automatically called by std::atexit() in mpi_tool. */
  // MPI_Finalize();
}
