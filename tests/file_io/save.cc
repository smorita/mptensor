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
#include <cstdio>

#include <mpi.h>
#include <mptensor/mptensor.hpp>

#include "initialize.hpp"

template <typename tensor>
void save(const size_t n, const char* tag) {
  using namespace mptensor;
  // tensor t(Shape(n, n + 1, n + 2, n + 3));
  // initialize(t);
  tensor t = initialize2<tensor>(n);
  char filename[256];
  std::sprintf(filename, "%s_mpi%04d", tag, t.get_comm_size());
  t.save(filename);
}


/* Main function */
int main(int argc, char **argv) {
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

  save<pdtensor>(n, "pd");
  save<pztensor>(n, "pz");
  if (mpisize == 1) {
    save<sdtensor>(n, "sd");
    save<sztensor>(n, "sz");
  }

  /* End */
  MPI_Finalize();
}
