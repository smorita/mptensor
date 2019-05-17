/*
  Dec. 24, 2015
  Copyright (C) 2015 Satoshi Morita
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


/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;
  using examples::benchmark::Timer;
  typedef Tensor<scalapack::Matrix,double> ptensor;

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
  std::vector<Timer> timer(2);

  Shape shape4(n,n+1,n+2,n+3);
  Shape shape2(n*(n+1),(n+2)*(n+3));
  ptensor A(shape4);
  Index index;
  for(int i=0;i<A.local_size();++i) {
    index = A.global_index(i);
    A[i] = elem(index);
  }
  ptensor T = A;

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

  double error = 0.0;
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
    std::cout << "time[0]: " << timer[0].result() << "\t" << shape4 << " -> " << shape2 << "\n";
    std::cout << "time[1]: " << timer[1].result() << "\t" << shape2 << " -> " << shape4 << "\n";
  }
  assert(error < 1.0e-10);

  /* End */
  MPI_Finalize();
}
