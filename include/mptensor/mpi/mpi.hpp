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
  \file   mpi.hpp
  \author Satoshi Morita
  \date   Mar 24 2020

  \brief  Helper functions for MPI

  This header file supports build without the MPI environment.
*/

#ifndef _MPTENSOR_MPI_HPP_
#define _MPTENSOR_MPI_HPP_

#ifndef _NO_MPI
#include <mpi.h>
#endif

namespace mptensor {
namespace mpi {

#ifdef _NO_MPI
using comm_type = int;
extern const comm_type MPI_COMM_WORLD;
#else
using comm_type = MPI_Comm;  /// Type for MPI communicator
#endif

//! Communicator of all processes, usable with or without MPI.
/*!
  Returns \c MPI_COMM_WORLD with MPI, and the dummy communicator \c 0
  without MPI (\c _NO_MPI). Write <tt>mpi::comm_world()</tt> instead of
  \c MPI_COMM_WORLD so that the same code builds in both cases.
*/
inline comm_type comm_world() {
  // Inside this namespace, MPI_COMM_WORLD is the macro from mpi.h with MPI,
  // or mptensor::mpi::MPI_COMM_WORLD without MPI.
  return MPI_COMM_WORLD;
}

extern int rank;
extern int size;
extern bool is_root;

void initialize(int argc, char **argv);
void finalize();
void barrier(const comm_type &comm);
void get_info(const comm_type &comm, int &rank, int &size, bool &is_root);

}  // namespace mpi
}  // namespace mptensor

#endif  // _MPTENSOR_MPI_HPP_