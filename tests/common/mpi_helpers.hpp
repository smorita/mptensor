#ifndef MPTENSOR_TEST_MPI_HELPERS_HPP_
#define MPTENSOR_TEST_MPI_HELPERS_HPP_

#include <mptensor/mptensor.hpp>

#ifndef _NO_MPI
#include <mpi.h>
#endif

namespace mptensor_test {

#ifdef _NO_MPI
//! Maximum of \c local over all ranks (serial build: \c local itself).
inline double max_over_ranks(double local, int /* comm */) { return local; }
//! Number of processes in MPI_COMM_WORLD (serial build: 1).
inline int world_size() { return 1; }
#else
//! Maximum of \c local over all ranks of \c comm. Collective.
inline double max_over_ranks(double local, MPI_Comm comm) {
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, comm);
  return global;
}
//! Number of processes in MPI_COMM_WORLD.
inline int world_size() {
  int size = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return size;
}
#endif

}  // namespace mptensor_test

#endif  // MPTENSOR_TEST_MPI_HELPERS_HPP_
