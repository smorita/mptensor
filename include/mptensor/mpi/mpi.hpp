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
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
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
#else
using comm_type = MPI_Comm;
#endif

extern int rank;
extern int size;
extern bool is_root;

void initialize(int argc, char **argv);
void finalize();
void barrier(const comm_type &comm);
void get_info(const comm_type &comm, int &rank, int &size, bool &is_root);

}  // namespace mpi
}  // namespace mptensor

#ifdef _NO_MPI
extern const mptensor::mpi::comm_type MPI_COMM_WORLD;
#endif

#endif  // _MPTENSOR_MPI_HPP_