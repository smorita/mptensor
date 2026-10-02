#ifndef MPTENSOR_TEST_FILE_IO_COMMON_HPP_
#define MPTENSOR_TEST_FILE_IO_COMMON_HPP_

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>

#include <mptensor/mptensor.hpp>

#include "tensor_types.hpp"

namespace mptensor_test {

//! Non-distributed tensor types (always LAPACK).
using LocalTensorD = Tensor<lapack::Matrix<double>>;
using LocalTensorC = Tensor<lapack::Matrix<complex>>;

constexpr std::size_t kFileIoN = 6;

//! Shape (n, n+1, n+2, n+3), values i0 + 100 i1 + 10^4 i2 + 10^6 i3,
//! then transposed by (3,1,0,2) so that axes_map is non-trivial.
template <typename TensorType>
TensorType make_initial_tensor() {
  const std::size_t n = kFileIoN;
  TensorType t(Shape(n, n + 1, n + 2, n + 3));
  for (std::size_t i = 0; i < t.local_size(); ++i) {
    const Index idx = t.global_index(i);
    t[i] = idx[0] + idx[1] * 100 + idx[2] * 10000 + idx[3] * 1000000;
  }
  t.transpose(Axes(3, 1, 0, 2));
  return t;
}

//! "<prefix>_mpi000N" (same names as legacy).
inline std::string data_filename(const std::string& prefix, int proc_size) {
  std::ostringstream ss;
  ss << prefix << "_mpi" << std::setw(4) << std::setfill('0') << proc_size;
  return ss.str();
}

//! Per-rank binary file written by Tensor::save: "<base>.NNNN.bin".
inline std::string binary_filename(const std::string& base, int rank) {
  std::ostringstream ss;
  ss << base << "." << std::setw(4) << std::setfill('0') << rank << ".bin";
  return ss.str();
}

}  // namespace mptensor_test

#endif  // MPTENSOR_TEST_FILE_IO_COMMON_HPP_
