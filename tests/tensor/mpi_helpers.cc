#include <gtest/gtest.h>

#include "mpi_helpers.hpp"
#include "tensor_types.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class CommWorld : public ::testing::Test {};
TYPED_TEST_SUITE(CommWorld, TensorTypes, TensorTypeNames);

// mpi::comm_world() is MPI_COMM_WORLD with MPI and the dummy 0 without it,
// so it matches the communicator of a default-constructed tensor.
TYPED_TEST(CommWorld, MatchesDefaultTensorCommunicator) {
  const TypeParam A;
  EXPECT_EQ(mpi::comm_world(), A.get_comm());
}

TYPED_TEST(CommWorld, ConstructsATensorOnAllProcesses) {
  const TypeParam A(mpi::comm_world(), Shape(2, 3));
  EXPECT_EQ(A.get_comm_size(), world_size());
  EXPECT_EQ(A.shape(), Shape(2, 3));
}

}  // namespace
}  // namespace mptensor_test
