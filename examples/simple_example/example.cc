/*!
  \file   example.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 03 2015
  \brief  Simple example of mptensor
*/

#include <iostream>
#include <mpi.h>
#include <mptensor.hpp>

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;
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

  /* Construct a tensor */
  ptensor A(Shape(n,n+1,n+2,n+3));

  /* Do something here */

  /* Output */
  for(int i=0;i<mpisize;++i) {
    if(i==mpirank) {
      std::cout << "rank=" << i << ": ";
      A.print_info(std::cout);
    }
    MPI_Barrier(comm);
  }

  /* End */
  MPI_Finalize();
}
