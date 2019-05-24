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
  \file   transpose.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Dec 02 2015
  \brief  Benchmark of transpose
*/

#include <iostream>
#include <mpi.h>
#include <mptensor.hpp>
#include "timer.hpp"

#ifdef _OPENMP
extern "C" {
  int omp_get_max_threads();
}
#else
int omp_get_max_threads(){ return 1; }
#endif

inline double elem(mptensor::Index index) {
  return double(index[0] - 2.0*index[1] + 3.0*index[2] - 4.0*index[3]);
}

inline mptensor::Index permutation(size_t n) {
  mptensor::Index idx(0,1,2,3);
  size_t x, val;
  {
    x = n/6;
    val = idx[x];
    for(size_t i=x;i>0;--i) idx[i] = idx[i-1];
    idx[0] = val;
  }
  {
    x = (n%6)/2+1;
    val = idx[x];
    for(size_t i=x;i>1;--i) idx[i] = idx[i-1];
    idx[1] = val;
  }
  {
    if((n%2)==1) {
      val = idx[2];
      idx[2] = idx[3];
      idx[3] = val;
    }
  }
  return idx;
}

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;
  typedef Tensor<scalapack::Matrix,double> ptensor;
  using examples::benchmark::Timer;

  /* Start */
  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  int mpirank;
  int mpisize;
  bool mpiroot;
  MPI_Comm_rank(comm, &mpirank);
  MPI_Comm_size(comm, &mpisize);
  mpiroot = (mpirank==0);

  /* Get arguments */
  int n;
  if (argc < 2) {
    if (mpiroot) std::cerr << "Usage: a.out N\n"
                           << "waring: assuming N=10" << std::endl;
    n = 10;
  } else {
    n = atoi(argv[1]);
  }

  Timer timer_all;
  std::vector<Timer> timer(24);
  std::vector<Index> axes(24);
  for(int i=0;i<24;++i) {
    axes[i] = permutation(i);
  }

  ptensor A(Shape(n,n+1,n+2,n+3));
  Index index;
  for(int i=0;i<A.local_size();++i) {
    index = A.global_index(i);
    A[i] = elem(index);
  }
  ptensor T = A;

  timer_all.start();
  for(int i=0;i<24;++i) {
    timer[i].start();
    T = transpose(T, axes[i]);
    timer[i].stop();
  }
  timer_all.stop();

  double error = 0.0;
  T = transpose(T, Index(2,3,0,1));
  for(int i=0;i<T.local_size();++i) {
    double diff = A[i]-T[i];
    if(error<std::abs(diff)) error = diff;
  }
  double max_error;
  MPI_Reduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

  if(mpiroot) {
    std::cout << "# ";
    T.print_info(std::cout);
    std::cout << "# mpisize= " << mpisize << "\n";
    std::cout << "# num_threads= " << omp_get_max_threads() << "\n";
    std::cout << "# error= " << max_error << "\n";
    std::cout << "all: " << timer_all.result() << "\n";
    for(int i=0;i<24;++i) {
      std::cout << "time[i]: " << timer[i].result() << "\t" << axes[i] << "\n";
    }
  }
  assert(error < 1.0e-10);

  /* End */
  MPI_Finalize();
}
