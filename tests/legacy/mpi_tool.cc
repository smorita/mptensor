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
  \file   mpi_tool.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   January 14 2015
  \brief  Some utilities for MPI
*/

#include <cstdlib>

#include "mpi_tool.hpp"

#ifdef _NO_MPI

const mpi_comm MPI_COMM_WORLD = 0;

void mpi_finalize() {}
void mpi_init(int argc, char **argv) {}
void mpi_info(const mpi_comm &comm, int &rank, int &size, bool &is_root) {
  rank = 0;
  size = 1;
  is_root = true;
}
void mpi_barrier(const mpi_comm &comm) {}
double mpi_reduce_max(double send, const mpi_comm &comm) { return send; }

#else  // _NO_MPI

void mpi_finalize() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) {
    MPI_Finalize();
  }
};

void mpi_init(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  atexit(mpi_finalize);
}

void mpi_info(const mpi_comm &comm, int &rank, int &size, bool &is_root) {
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  is_root = (rank == 0);
}

void mpi_barrier(const mpi_comm &comm) { MPI_Barrier(comm); }

double mpi_reduce_max(double send, const mpi_comm &comm) {
  double recv;
  MPI_Reduce(&send, &recv, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
  return recv;
}

#endif  // _NO_MPI
