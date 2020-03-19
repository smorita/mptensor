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
  \file   load.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 18 2020
  \brief  Test code for load function
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
void load(const std::string& tag, int proc_size, const tensor& t0) {
  tensor t;
  std::string fname = filename(tag, proc_size);
  t.load(fname);

  double val = max_abs(t - t0);
  if (t.get_comm_rank() == 0) {
    std::cout << fname << " is loaded. error= " << val << std::endl;
  }
}

/* Main function */
int main(int argc, char** argv) {
  /* Start */
  mpi_init(argc, argv);
  int mpirank;
  int mpisize;
  bool mpiroot;
  mpi_comm comm = MPI_COMM_WORLD;
  mpi_info(comm, mpirank, mpisize, mpiroot);

  size_t n = 6;

#ifndef _NO_MPI
  pdtensor pd_0 = initialize<pdtensor>(n);
  pztensor pz_0 = initialize<pztensor>(n);
  for (int proc_size = 1; proc_size <= 4; proc_size++) {
    load<pdtensor>("pd", proc_size, pd_0);
    load<pztensor>("pz", proc_size, pz_0);
  }
  load<pdtensor>("sd", 1, pd_0);
  load<pztensor>("sz", 1, pz_0);

  if (mpisize == 1) {
    std::cout << "# load as non-distributed tensor"
              << "\n";
    sdtensor sd_0 = initialize<sdtensor>(n);
    sztensor sz_0 = initialize<sztensor>(n);
    for (int proc_size = 1; proc_size <= 4; proc_size++) {
      load<sdtensor>("pd", proc_size, sd_0);
      load<sztensor>("pz", proc_size, sz_0);
    }
    load<sdtensor>("sd", 1, sd_0);
    load<sztensor>("sz", 1, sz_0);
  }
#else
  sdtensor sd_0 = initialize<sdtensor>(n);
  sztensor sz_0 = initialize<sztensor>(n);
  load<sdtensor>("sd", 1, sd_0);
  load<sztensor>("sz", 1, sz_0);
#endif

  /* End */
  /* automatically called by std::atexit() in mpi_tool. */
  // MPI_Finalize();
}
