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

#include <cstdlib>
#include <iostream>

#include <mptensor/mptensor.hpp>

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;

  /* Start */
  mpi::initialize(argc, argv);
  // This helper function initializes the MPI environment and
  // registers MPI_Finalize at the end of this program.
  // In addition, mpi::rank, mpi::size, and mpi::is_root are set.

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpi::is_root) {
      std::cerr << "Usage: a.out N\n"
                << "Warning: assuming N=10" << std::endl;
    }
    n = 10;
  } else {
    n = std::atoi(argv[1]);
  }

  /* Construct a tensor */
  DTensor A(Shape(n, n + 1, n, n + 1));

  // Initializer lists instead of Index, Shape and Axes class are also possible.
  // DTensor A({n, n + 1, n, n + 1});

  /* Initialize a tensor */
  for (size_t i=0; i < A.local_size(); ++i) {
    // Get a global index form a local index.
    Index idx = A.global_index(i);

    // Set elements of A based on the global index.
    A[i] = double(idx[0] + idx[1] + idx[2] + idx[3]);
  }

  /* Do something here */
  double trace_A = trace(A, {0, 1}, {2, 3});

  /* Output */
  A.print_info_mpi(std::cout);

  if (mpi::is_root) {
    std::cout << "trace(A)= " << trace_A << std::endl;

    // Note that the below line does not work in parallel.
    // Since trace(A) requires communication between processes,
    // it should be called from all the process at the same time.
    // std::cout << "trace(A)= " << trace(A) << std::endl;
  }

  /* End */
  // MPI_Finalize() is automatically called if you use mpi::initialize().
}
