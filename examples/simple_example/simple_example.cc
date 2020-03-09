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
  \file   example.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 03 2015
  \brief  Simple example of mptensor
*/

#include <iostream>

#include <mpi.h>
#include "mptensor/mptensor.hpp"

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;
  typedef Tensor<scalapack::Matrix, double> ptensor;

  /* Start */
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpirank;
  int mpisize;
  bool mpiroot;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);
  mpiroot = (mpirank == 0);

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpiroot)
      std::cerr << "Usage: a.out N\n"
                << "waring: assuming N=10" << std::endl;
    n = 10;
  } else {
    n = atoi(argv[1]);
  }

  /* Construct a tensor */
  ptensor A(Shape(n, n + 1, n + 2, n + 3));

  /* Do something here */

  /* Output */
  for (int i = 0; i < mpisize; ++i) {
    if (i == mpirank) {
      std::cout << "rank=" << i << ": ";
      A.print_info(std::cout);
    }
    MPI_Barrier(comm);
  }

  /* End */
  MPI_Finalize();
}
