#pragma once

#include <iomanip>
#include <sstream>
#include <string>

#include <mptensor/mptensor.hpp>

using namespace mptensor;
using sdtensor = Tensor<lapack::Matrix, double>;
using sztensor = Tensor<lapack::Matrix, complex>;

#ifdef _NO_MPI
using pdtensor = Tensor<lapack::Matrix, double>;
using pztensor = Tensor<lapack::Matrix, complex>;
#else
using pdtensor = Tensor<scalapack::Matrix, double>;
using pztensor = Tensor<scalapack::Matrix, complex>;
#endif

template <typename tensor>
tensor initialize(size_t n) {
  tensor t(Shape(n, n + 1, n + 2, n + 3));
  for (size_t i = 0; i < t.local_size(); ++i) {
    Index idx = t.global_index(i);
    t[i] = idx[0] + idx[1] * 100 + idx[2] * 10000 + idx[3] * 1000000;
  }
  t.transpose({3, 1, 0, 2});
  return t;
}

std::string filename(const std::string &prefix, int proc_size) {
  std::ostringstream ss;
  ss << prefix << "_mpi";
  ss << std::setw(4) << std::setfill('0') << proc_size;
  return ss.str();
}