// A minimal program that uses mptensor as a downstream project would.
// It builds without any mptensor-specific compile options: include paths,
// _NO_MPI, MPI, OpenMP, and LAPACK all come from the mptensor::mptensor target.

#include <iostream>

#include <mptensor/mptensor.hpp>

int main(int argc, char** argv) {
  using namespace mptensor;
  mpi::initialize(argc, argv);

  const DTensor A(mpi::comm_world(), Shape(2, 3, 4));
  const bool ok =
      (A.shape() == Shape(2, 3, 4)) && (A.get_comm_size() == mpi::size);

  if (mpi::is_root) {
    std::cout << (ok ? "package consumer: OK" : "package consumer: FAILED")
              << std::endl;
  }
  mpi::finalize();
  return ok ? 0 : 1;
}
