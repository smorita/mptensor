// MPI-aware main for GoogleTest.
//
// With MPI, every rank runs all tests. Rank 0 prints the usual gtest output;
// other ranks print only failed assertions, prefixed with "[rank N]".
// The exit code is non-zero on every rank if any rank failed.

#include <cstdio>

#include <gtest/gtest.h>

#ifndef _NO_MPI
#include <mpi.h>
#endif

namespace {

#ifndef _NO_MPI
class RankFailurePrinter : public ::testing::EmptyTestEventListener {
 public:
  explicit RankFailurePrinter(int rank) : rank_(rank) {}

  void OnTestPartResult(const ::testing::TestPartResult& result) override {
    if (!result.failed()) return;
    std::printf("[rank %d] %s:%d: Failure\n%s\n", rank_,
                result.file_name() ? result.file_name() : "unknown",
                result.line_number(), result.summary());
    std::fflush(stdout);
  }

 private:
  int rank_;
};
#endif

}  // namespace

int main(int argc, char** argv) {
#ifdef _NO_MPI
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
#else
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new RankFailurePrinter(rank));
  }

  const int local_failed = (RUN_ALL_TESTS() != 0) ? 1 : 0;
  int any_failed = 0;
  MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Finalize();
  return any_failed;
#endif
}
