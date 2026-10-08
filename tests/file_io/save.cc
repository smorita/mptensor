#include <filesystem>
#include <string>

#include <gtest/gtest.h>

#include "file_io_common.hpp"
#include "mpi_helpers.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
void save_and_check(const std::string& prefix) {
  const TensorType t = make_initial_tensor<TensorType>();
  const std::string base = data_filename(prefix, t.get_comm_size());
  SCOPED_TRACE(base);

  t.save(base);

  if (t.get_comm_rank() == 0) {
    EXPECT_TRUE(std::filesystem::exists(base));
  }
  EXPECT_TRUE(
      std::filesystem::exists(binary_filename(base, t.get_comm_rank())));
}

#ifndef _NO_MPI
TEST(Save, Distributed) {
  save_and_check<TensorD>("pd");
  save_and_check<TensorC>("pz");
}
#endif

TEST(Save, NonDistributed) {
  if (world_size() != 1) GTEST_SKIP() << "only with 1 process";
  save_and_check<LocalTensorD>("sd");
  save_and_check<LocalTensorC>("sz");
}

}  // namespace
}  // namespace mptensor_test
