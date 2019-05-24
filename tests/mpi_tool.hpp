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
  \file   mpi_tool.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   January 14 2015
  \brief  Some utilities for MPI
*/

#ifndef _MPI_TOOL_HPP_
#define _MPI_TOOL_HPP_

#include <mpi.h>

void mpi_finalize();
void mpi_init(int argc, char **argv);
void mpi_info(const MPI_Comm &comm, int &rank, int &size, bool &is_root);

#endif // _MPI_TOOL_HPP_
