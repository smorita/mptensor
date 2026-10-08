#include <string>

#include <gtest/gtest.h>

#include "file_io_common.hpp"
#include "mpi_helpers.hpp"

namespace mptensor_test {
namespace {

// Tensor data is stored in binary form, so the comparison is exact.
template <typename TensorType>
void expect_loaded(const std::string& prefix, int proc_size) {
  const std::string base = data_filename(prefix, proc_size);
  SCOPED_TRACE(base);
  const TensorType expected = make_initial_tensor<TensorType>();

  TensorType t;
  t.load(base);

  EXPECT_EQ(max_abs(t - expected), 0.0);
}

#ifndef _NO_MPI
TEST(Load, AsDistributed) {
  for (int p = 1; p <= 4; ++p) {
    expect_loaded<TensorD>("pd", p);
    expect_loaded<TensorC>("pz", p);
  }
  expect_loaded<TensorD>("sd", 1);
  expect_loaded<TensorC>("sz", 1);
}

TEST(Load, AsNonDistributed) {
  if (world_size() != 1) GTEST_SKIP() << "only with 1 process";
  for (int p = 1; p <= 4; ++p) {
    expect_loaded<LocalTensorD>("pd", p);
    expect_loaded<LocalTensorC>("pz", p);
  }
  expect_loaded<LocalTensorD>("sd", 1);
  expect_loaded<LocalTensorC>("sz", 1);
}
#else
TEST(Load, NonDistributed) {
  expect_loaded<LocalTensorD>("sd", 1);
  expect_loaded<LocalTensorC>("sz", 1);
}
#endif

}  // namespace
}  // namespace mptensor_test
