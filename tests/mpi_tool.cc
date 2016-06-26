/*
  Jan. 14, 2015
  Copyright (C) 2015 Satoshi Morita
 */

#include <cstdlib>
#include <mpi.h>
#include "mpi_tool.hpp"

void mpi_finalize() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if(!mpi_finalized) {
    MPI_Finalize();
  }
};

void mpi_init(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  atexit( mpi_finalize );
}

void mpi_info(const MPI_Comm &comm, int &rank, int &size, bool &is_root) {
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  is_root = (rank==0);
}

