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
  \file   save_load.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 03 2015
  \brief  Example of save and load
*/

#include <ctime>
#include <iomanip>
#include <iostream>

#include <mptensor/mptensor.hpp>

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;

  /* Start */
  mpi::initialize(argc, argv);
  mpi::comm_type comm = MPI_COMM_WORLD;

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpi::is_root) {
      std::cerr << "Usage: a.out N\n"
                << "waring: assuming N=10" << std::endl;
    }
    n = 10;
  } else {
    n = atoi(argv[1]);
  }

  /* Construct a tensor */
  DTensor A(Shape(n, n + 1, n + 2, n + 3));

  /* Do something here */
  set_seed(std::time(NULL) + mpi::rank);
  random_tensor::fill(A);
  A.transpose(Axes(3, 1, 0, 2));
  A.save("A.dat");

  DTensor B;
  B.load("A.dat");

  /* Output */
  if (mpi::is_root) std::cout << "########## Saved Tensor ##########\n";
  for (int i = 0; i < mpi::size; ++i) {
    if (i == mpi::rank) {
      std::cout << "rank=" << i << ": ";
      A.print_info(std::cout);
    }
    mpi::barrier(comm);
  }
  if (mpi::is_root) std::cout << "########## Loaded Tensor ##########\n";
  for (int i = 0; i < mpi::size; ++i) {
    if (i == mpi::rank) {
      std::cout << "rank=" << i << ": ";
      B.print_info(std::cout);
    }
    mpi::barrier(comm);
  }

  double val = max_abs(B - A);
  if (mpi::is_root) {
    std::cout << "########## Error ##########\n"
              << std::scientific << std::setprecision(10) << val << std::endl;
  }

  /* End */
}
