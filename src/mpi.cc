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
  \file   mpi.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 24 2020

  \brief  Helper functions of MPI library
*/

#include <cstdlib>

#include "mptensor/mpi/mpi.hpp"

namespace mptensor {
namespace mpi {

#ifdef _NO_MPI

const comm_type MPI_COMM_WORLD = 0;

int rank = 0;
int size = 1;
bool is_root = true;

void initialize(int argc, char **argv) {
  get_info(MPI_COMM_WORLD, rank, size, is_root);
}

void finalize() {}

void barrier(const comm_type &comm) {}

void get_info(const comm_type &comm, int &rank, int &size, bool &is_root) {
  rank = 0;
  size = 1;
  is_root = true;
}

#else  // _NO_MPI

int rank = -1; /// Rank of the calling process in the communicator
int size = -1; /// Number of processes in the communicator
bool is_root = false; /// True if the calling process is the root (rank 0), false otherwise

/// @brief Initialize MPI environment and get MPI information
/// @param argc Number of command-line arguments
/// @param argv Array of command-line arguments
///
/// This function should be called before any MPI communication.
/// It also registers the finalize function to be called at exit.
void initialize(int argc, char **argv) {
  int initialized;
  MPI_Initialized(&initialized);
  if(!initialized) {
    MPI_Init(&argc, &argv);
  }
  get_info(MPI_COMM_WORLD, rank, size, is_root);
  atexit(finalize);
}

/// @brief Finalize MPI environment
///
/// This function is registered to be called at exit by initialize().
void finalize() {
  int finalized;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}

/// @brief Barrier synchronization among processes in the communicator
/// @param comm MPI communicator
void barrier(const comm_type &comm) {
  MPI_Barrier(comm);
}

/// @brief Get MPI information: rank, size, and whether the process is the root
/// @param comm MPI communicator
/// @param rank Rank of the calling process in the communicator
/// @param size Number of processes in the communicator
/// @param is_root True if the calling process is the root (rank 0), false otherwise
void get_info(const comm_type &comm, int &rank, int &size, bool &is_root) {
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  is_root = (rank == 0);
}

#endif
}
}
