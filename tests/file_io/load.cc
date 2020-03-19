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

#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>

#include <mpi.h>
#include <mptensor/mptensor.hpp>

#include "initialize.hpp"

template <typename tensor>
void load(const char* tag, int proc_size, const tensor& t0) {
  using namespace mptensor;
  char filename[256];
  tensor t;
  std::sprintf(filename, "%s_mpi%04d", tag, proc_size);
  t.load(filename);

  // if (t.get_comm_rank() == 0) {
  //   std::cout << filename << std::endl;
  //   t.print_info(std::cout);
  // }
  // return;
  double val = max_abs(t - t0);
  if (t.get_comm_rank() == 0) {
    std::cout << filename << " error= " << val << std::endl;
  }
}

/* Main function */
int main(int argc, char** argv) {
  using namespace mptensor;
  typedef Tensor<scalapack::Matrix, double> pdtensor;
  typedef Tensor<scalapack::Matrix, complex> pztensor;
  typedef Tensor<lapack::Matrix, double> sdtensor;
  typedef Tensor<lapack::Matrix, complex> sztensor;

  /* Start */
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpirank;
  int mpisize;
  bool mpiroot;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);
  mpiroot = (mpirank == 0);

  char filename[256];

  size_t n = 6;
  pdtensor pd_0 = initialize2<pdtensor>(n);
  pztensor pz_0 = initialize2<pztensor>(n);
  sdtensor sd_0 = initialize2<sdtensor>(n);
  sztensor sz_0 = initialize2<sztensor>(n);

  pdtensor pdt_0(comm, Shape(n, n + 1, n + 2, n + 3));
  pztensor pzt_0(comm, Shape(n, n + 1, n + 2, n + 3));
  sdtensor sdt_0(0, Shape(n, n + 1, n + 2, n + 3));
  sztensor szt_0(0, Shape(n, n + 1, n + 2, n + 3));
  initialize(pdt_0);
  initialize(pzt_0);
  initialize(sdt_0);
  initialize(szt_0);

  for (int proc_size = 1; proc_size <= 4; proc_size++) {
    load<pdtensor>("pd", proc_size, pd_0);
    load<pztensor>("pz", proc_size, pz_0);
  }
  load<pdtensor>("sd", 1, pd_0);
  load<pztensor>("sz", 1, pz_0);

  if (mpisize == 1) {
    std::cout << "# load as non-distributed tensor" << "\n";
    for (int proc_size = 1; proc_size <= 4; proc_size++) {
      load<sdtensor>("pd", proc_size, sd_0);
      load<sztensor>("pz", proc_size, sz_0);
    }
    load<sdtensor>("sd", 1, sd_0);
    load<sztensor>("sz", 1, sz_0);
  }

  /* End */
  MPI_Finalize();
}
