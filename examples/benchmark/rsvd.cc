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
  \file   benchmark/rsvd.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Feb  32016

  \brief  Benchmark for RSVD.
*/

#include <ctime>
#include <iomanip>
#include <iostream>

#include <mpi.h>
#include <mptensor/mptensor.hpp>

#include "timer.hpp"

#ifdef _OPENMP
extern "C" {
int omp_get_max_threads();
}
#else
int omp_get_max_threads() { return 1; }
#endif

using namespace mptensor;
typedef Tensor<scalapack::Matrix, double> ptensor;

namespace {

double singular_value(size_t i) { return std::pow(1.0 + i, -2.0); }

ptensor test_tensor(size_t n) {
  MPI_Comm comm = MPI_COMM_WORLD;
  ptensor D(Shape(n * n, n * n));
  {
    const int m = D.local_size();
    Index idx;
    for (size_t i = 0; i < m; ++i) {
      idx = D.global_index(i);
      if (idx[0] == idx[1]) {
        D[i] = singular_value(idx[0]);
      }
    }
  }
  ptensor U1, U2;
  {
    ptensor o1(comm, Shape(n + 2, n, n * n), 2);
    ptensor o2(comm, Shape(n + 3, n + 1, n * n), 2);
    random_tensor::fill(o1);
    random_tensor::fill(o2);
    ptensor r;
    qr(o1, Axes(0, 1), Axes(2), U1, r);
    qr(o2, Axes(0, 1), Axes(2), U2, r);
  }

  return transpose(
      tensordot(tensordot(U1, D, Axes(2), Axes(0)), U2, Axes(2), Axes(2)),
      Axes(1, 3, 0, 2));
}

class Multiply_row_02 {
 public:
  Multiply_row_02(const ptensor& t) : t_(t){};
  ptensor operator()(const ptensor& t) {
    return tensordot(t, t_, Axes(0, 1), Axes(0, 2));
  };

 private:
  const ptensor& t_;
};

class Multiply_col_13 {
 public:
  Multiply_col_13(const ptensor& t) : t_(t){};
  ptensor operator()(const ptensor& t) {
    return tensordot(t_, t, Axes(1, 3), Axes(0, 1));
  };

 private:
  const ptensor& t_;
};

}  // namespace

/* Main function */
int main(int argc, char** argv) {
  using examples::benchmark::Timer;

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
  int n, target, oversamp;
  unsigned int seed;
  if (argc < 2) {
    if (mpiroot)
      std::cerr << "Usage: a.out [N [target_rank [oversamp [seed]]]]\n"
                << "waring: assuming N=10" << std::endl;
  }
  n = (argc > 1) ? atoi(argv[1]) : 10;
  target = (argc > 2) ? atoi(argv[2]) : n;
  oversamp = (argc > 3) ? atoi(argv[3]) : target;
  seed = (argc > 4) ? atoi(argv[4]) : std::time(NULL);
  set_seed(seed + mpirank);

  Timer timer_full, timer_rsvd, timer_rsvd_func;

  ptensor A = test_tensor(n);

  timer_full.start();
  std::vector<double> s0;
  ptensor u0, vt0;
  svd(A, Axes(0, 2), Axes(1, 3), u0, s0, vt0);
  timer_full.stop();

  timer_rsvd.start();
  std::vector<double> s1;
  ptensor u1, vt1;
  rsvd(A, Axes(0, 2), Axes(1, 3), u1, s1, vt1, target, oversamp);
  timer_rsvd.stop();

  timer_rsvd_func.start();
  std::vector<double> s2;
  ptensor u2, vt2;
  {
    Multiply_row_02 mr02(A);
    Multiply_col_13 mc13(A);
    Shape shape = A.shape();
    Shape shape_row(shape[0], shape[2]);
    Shape shape_col(shape[1], shape[3]);
    rsvd(mr02, mc13, shape_row, shape_col, u2, s2, vt2, target, oversamp);
  }
  timer_rsvd_func.stop();

  if (mpiroot) {
    std::cout << "# ";
    A.print_info(std::cout);
    std::cout << "# mpisize= " << mpisize << "\n"
              << "# num_threads= " << omp_get_max_threads() << "\n"
              << "# n= " << n << "\n"
              << "# target_rank= " << target << "\n"
              << "# oversamp= " << oversamp << "\n"
              << "# seed= " << seed << "\n"
              << "# time_full= " << timer_full.result() << "\n"
              << "# time_rsvd= " << timer_rsvd.result() << "\n"
              << "# time_rsvd_func= " << timer_rsvd_func.result() << "\n";
    std::cout << "# index s_exact s_full s_rsvd s_rsvd_f s_full-s_rsvd "
                 "s_full-s_rsvd_f\n";
    for (size_t i = 0; i < target; ++i) {
      std::cout << i << " " << std::scientific << std::setprecision(10)
                << singular_value(i) << " " << s0[i] << " " << s1[i] << " "
                << s2[i] << " " << s0[i] - s1[i] << " " << s0[i] - s2[i]
                << "\n";
    }
  }

  /* End */
  MPI_Finalize();
}
