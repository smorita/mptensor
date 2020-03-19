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
  \file   save.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 03 2018
  \brief  Test code of save function
*/

#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

// #include <mpi.h>
#include <mptensor/mptensor.hpp>

#include "../mpi_tool.hpp"
#include "common.hpp"

template <typename tensor>
void save(const size_t n, const std::string &tag) {
  tensor t = initialize<tensor>(n);
  std::string fname = filename(tag, t.get_comm_size());
  t.save(fname);
  if (t.get_comm_rank() == 0) {
    std::cout << fname << " is saved." << std::endl;
  }
}

/* Main function */
int main(int argc, char **argv) {
  /* Start */
  mpi_init(argc, argv);
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_comm comm = MPI_COMM_WORLD;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t n = 6;

#ifndef _NO_MPI
  save<pdtensor>(n, "pd");
  save<pztensor>(n, "pz");

  if (mpisize == 1) {
    save<sdtensor>(n, "sd");
    save<sztensor>(n, "sz");
  }
#else
  save<sdtensor>(n, "sd");
  save<sztensor>(n, "sz");
#endif

  /* End */
  /* automatically called by std::atexit() in mpi_tool. */
  // MPI_Finalize();
}
