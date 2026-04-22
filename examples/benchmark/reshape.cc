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
  \file   benchmark/reshape.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Dec 24 2015
  \brief  Benchmark of reshape
*/

#include <iostream>

#include <mptensor/mptensor.hpp>

#include "timer.hpp"

#ifdef _OPENMP
extern "C" {
int omp_get_max_threads();
}
#else
int omp_get_max_threads() { return 1; }
#endif

inline double elem(mptensor::Index index) {
  return double(index[0] - 2.0 * index[1] + 3.0 * index[2] - 4.0 * index[3]);
}

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;
  using examples::benchmark::Timer;

  /* Start */
  mpi::initialize(argc, argv);

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpi::is_root)
      std::cerr << "Usage: a.out N\n"
                << "Warning: assuming N=10" << std::endl;
    n = 10;
  } else {
    n = atoi(argv[1]);
  }

  Timer timer_all;
  std::vector<Timer> timer(2);

  Shape shape4(n, n + 1, n + 2, n + 3);
  Shape shape2(n * (n + 1), (n + 2) * (n + 3));
  DTensor A(shape4);
  Index index;
  for (int i = 0; i < A.local_size(); ++i) {
    index = A.global_index(i);
    A[i] = elem(index);
  }
  DTensor T = A;

  timer_all.start();
  {
    timer[0].start();
    T = reshape(T, shape2);
    timer[0].stop();

    timer[1].start();
    T = reshape(T, shape4);
    timer[1].stop();
  }
  timer_all.stop();

  double max_error = max_abs(A - T);

  if (mpi::is_root) {
    std::cout << "# ";
    T.print_info(std::cout);
    std::cout << "# mpisize= " << mpi::size << "\n";
    std::cout << "# num_threads= " << omp_get_max_threads() << "\n";
    std::cout << "# error= " << max_error << "\n";
    std::cout << "all: " << timer_all.result() << "\n";
    std::cout << "time[0]: " << timer[0].result() << "\t" << shape4 << " -> "
              << shape2 << "\n";
    std::cout << "time[1]: " << timer[1].result() << "\t" << shape2 << " -> "
              << shape4 << "\n";
  }
  assert(max_error < 1.0e-10);

  /* End */
}
