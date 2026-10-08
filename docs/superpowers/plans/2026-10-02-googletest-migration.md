# GoogleTest Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reimplement every test in `tests/legacy` on GoogleTest, running under MPI with 1–4 processes and without MPI, and run them in a CI matrix.

**Architecture:** A shared static library `mptensor_gtest_main` provides an MPI-aware `main` plus header-only helpers (`tests/common`). Each test directory builds one gtest executable that a CMake helper `mptensor_add_test()` registers with CTest once per process count. Tests fill tensors from analytic functions, apply one operation, reduce the maximum error over all ranks, and evaluate `EXPECT_LT` identically on every rank.

**Tech Stack:** C++17, CMake ≥ 3.16, GoogleTest (system ≥ 1.14 or fetched v1.18.0), MPI + ScaLAPACK (optional), LAPACK, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-02-googletest-migration-design.md`

## Global Constraints

- `cmake_minimum_required(VERSION 3.16...3.31)`.
- GoogleTest: `find_package(GTest 1.14 CONFIG QUIET)`, else `FetchContent` of `v1.18.0` with `INSTALL_GTEST=OFF`, `BUILD_GMOCK=OFF`.
- Use only GoogleTest features available in 1.14.
- `option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)`; `tests/legacy` is built only when `BUILD_TESTS` and `BUILD_LEGACY_TESTS` are both `ON`.
- **No file under `tests/legacy/` may be modified.**
- New tests do not include any file from `tests/legacy`.
- CTest names: `<dir>/<name>_mpi000N` for N = 1, 2, 3, 4 with MPI; `<dir>/<name>_serial` without.
- Size `L = 10`; tolerance `1e-10` (`1e-8` for `EighGeneral`); shapes identical to the legacy tests.
- Value types: `double` and `mptensor::complex`; matrix type `scalapack::Matrix` with MPI, `lapack::Matrix` without.
- No timing, no `print_info`, no command-line size argument.
- Commit messages follow the repository style (plain imperative sentence, no prefix) and end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Work on branch `feature/googletest`. Never push to `develop`.

## Review Focus

1. **A failure on only some ranks.** If one rank returns early (e.g. an `ASSERT_*` on a rank-local value), the other ranks block in the next collective call. Expected: the test fails, it does not hang. → Only assert on values that are identical on all ranks (after `max_over_ranks` / `max_abs`); every CTest test gets `TIMEOUT 600` (Task 1), and Task 1 verifies a rank-1-only failure terminates with a non-zero exit code and a `[rank 1]` message.
2. **Stale `file_io` data from a previous run.** If `save` silently fails, `load` could still pass using old files. Expected: every CTest run starts from an empty data directory. → A `file_io/prepare` fixture deletes and recreates the directory (Task 7), verified by planting a bogus file.
3. **Offline builds.** Expected: `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<dir>` builds without network. → Verified in Task 1.
4. **System GoogleTest 1.14 instead of fetched 1.18.** Expected: the same tests compile and pass with both. → CI job 2 installs `libgtest-dev` and checks the configure log says `GoogleTest: found` (Task 8).
5. **Legacy tests after the CMake minimum version bump.** Expected: `-DBUILD_LEGACY_TESTS=ON` still builds and passes `tests/legacy` unchanged. → Verified in Task 1 and in CI job 1.

## Local environment notes

- Local MPI is Intel MPI (`/opt/intel/oneapi/mpi/2021.17/bin/mpiexec`), the machine has 1 core; Intel MPI oversubscribes without extra flags. CI uses Open MPI with `--oversubscribe`.
- Use Ninja and build directories under `build/` (ignored by git):
  - MPI: `cmake -S . -B build/gtest-mpi -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug`
  - No MPI: `cmake -S . -B build/gtest-nompi -G Ninja -DBUILD_TESTS=ON -DENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Debug`
- These are characterization tests of existing, working code, so they are expected to pass when first run. Task 1 proves that the assertions and the failure propagation actually work (mutation checks); later tasks only run the new tests. **If a newly ported test fails, stop and report to the user** (spec Section 7) — do not change the library or weaken the test.

## File Structure

| Path | Responsibility |
|---|---|
| `CMakeLists.txt` (modify) | CMake 3.16 minimum, `BUILD_LEGACY_TESTS` option |
| `README.md` (modify) | CMake version, how to build and run tests |
| `tests/CMakeLists.txt` (rewrite) | GoogleTest acquisition, `mptensor_gtest_main`, `mptensor_add_test()`, subdirectories |
| `tests/common/gtest_mpi_main.cc` | MPI-aware gtest `main` |
| `tests/common/mpi_helpers.hpp` | `max_over_ranks()`, `world_size()` |
| `tests/common/tensor_types.hpp` | `TensorD`, `TensorC`, `TensorTypes`, `TensorTypeNames` |
| `tests/common/test_functions.hpp` | `kL`, `kEps`, `func2_1/func4_1/func4_2`, `conj_value`, `fill_tensor`, `max_error` |
| `tests/tensor/CMakeLists.txt` | `test_tensor` executable and its CTest registration |
| `tests/tensor/*.cc` | One file per operation (see spec Section 4.3) |
| `tests/file_io/CMakeLists.txt` | `test_save`, `test_load`, fixtures |
| `tests/file_io/prepare_data_dir.cmake` | Empties the data directory |
| `tests/file_io/file_io_common.hpp` | Initial tensor, file names, local tensor types |
| `tests/file_io/save.cc`, `load.cc` | Save / load tests |
| `.github/workflows/build.yml` (modify) | CI matrix |

---

### Task 1: Test infrastructure and the first test (`Transpose`)

**Files:**
- Modify: `CMakeLists.txt:1` and `CMakeLists.txt:19-20`
- Modify: `README.md` (Prerequisites and a new "Tests" section)
- Rewrite: `tests/CMakeLists.txt`
- Create: `tests/common/gtest_mpi_main.cc`, `tests/common/mpi_helpers.hpp`, `tests/common/tensor_types.hpp`, `tests/common/test_functions.hpp`
- Create: `tests/tensor/CMakeLists.txt`, `tests/tensor/transpose.cc`

**Interfaces:**
- Produces (used by every later task):
  - CMake target `mptensor_gtest_main` (links `mptensor` and `GTest::gtest`, adds `tests/common` to the include path).
  - CMake function `mptensor_add_test(<name> <target> [WORKING_DIRECTORY <dir>] [FIXTURES_SETUP <f>...] [FIXTURES_REQUIRED <f>...])`.
  - Namespace `mptensor_test` (contains `using namespace mptensor;`):
    - `double max_over_ranks(double local, <comm>)`, `int world_size()`
    - `TensorD`, `TensorC`, `TensorTypes`, `TensorTypeNames`
    - `constexpr int kL = 10;`, `constexpr double kEps = 1.0e-10;`
    - `template <typename T> T func2_1(const Index&, const Shape&)`, same for `func4_1`, `func4_2`
    - `inline double conj_value(double)`, `inline complex conj_value(const complex&)`
    - `template <typename TensorType, typename F> void fill_tensor(TensorType&, F f)` where `f(const Index&, const Shape&)`
    - `template <typename TensorType, typename F> double max_error(const TensorType&, F exact)` where `exact(const Index&)`; returns the maximum over all ranks.

- [ ] **Step 1: Raise the CMake minimum and add `BUILD_LEGACY_TESTS`**

In `CMakeLists.txt`, replace line 1:

```cmake
cmake_minimum_required(VERSION 3.16...3.31)
```

and after `option(BUILD_TESTS "Build tests" OFF)` add:

```cmake
option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)
```

- [ ] **Step 2: Rewrite `tests/CMakeLists.txt`**

```cmake
# GoogleTest: prefer an installed package, otherwise fetch a pinned release.
find_package(GTest 1.14 CONFIG QUIET)
if(GTest_FOUND)
  message(STATUS "GoogleTest: found ${GTest_VERSION} (${GTest_DIR})")
else()
  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.18.0
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
  message(STATUS "GoogleTest: using FetchContent (v1.18.0)")
endif()

# Shared main and helpers for all gtest-based tests.
add_library(mptensor_gtest_main STATIC common/gtest_mpi_main.cc)
target_link_libraries(mptensor_gtest_main PUBLIC mptensor GTest::gtest)
target_include_directories(mptensor_gtest_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/common)

# mptensor_add_test(<name> <target>
#                   [WORKING_DIRECTORY <dir>]
#                   [FIXTURES_SETUP <fixture>...]
#                   [FIXTURES_REQUIRED <fixture>...])
# Registers <target> as <name>_mpi000N (N = 1..4) with MPI, or <name>_serial.
function(mptensor_add_test name target)
  cmake_parse_arguments(ARG "" "WORKING_DIRECTORY" "FIXTURES_SETUP;FIXTURES_REQUIRED" ${ARGN})
  if(NOT ARG_WORKING_DIRECTORY)
    set(ARG_WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()
  set(test_names)
  if(ENABLE_MPI)
    foreach(np RANGE 1 4)
      set(test_name ${name}_mpi000${np})
      add_test(NAME ${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:${target}> ${MPIEXEC_POSTFLAGS}
        WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
      list(APPEND test_names ${test_name})
    endforeach()
  else()
    set(test_name ${name}_serial)
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${target}>
      WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
    list(APPEND test_names ${test_name})
  endif()
  # A hang (e.g. ranks diverging before a collective call) must fail, not block.
  set_tests_properties(${test_names} PROPERTIES TIMEOUT 600)
  if(ARG_FIXTURES_SETUP)
    set_tests_properties(${test_names} PROPERTIES FIXTURES_SETUP "${ARG_FIXTURES_SETUP}")
  endif()
  if(ARG_FIXTURES_REQUIRED)
    set_tests_properties(${test_names} PROPERTIES FIXTURES_REQUIRED "${ARG_FIXTURES_REQUIRED}")
  endif()
endfunction()

add_subdirectory(tensor)

if(BUILD_LEGACY_TESTS)
  add_subdirectory(legacy)
endif()
```

- [ ] **Step 3: Create `tests/common/gtest_mpi_main.cc`**

```cpp
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
```

- [ ] **Step 4: Create `tests/common/mpi_helpers.hpp`**

```cpp
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
```

- [ ] **Step 5: Create `tests/common/tensor_types.hpp`**

```cpp
#ifndef MPTENSOR_TEST_TENSOR_TYPES_HPP_
#define MPTENSOR_TEST_TENSOR_TYPES_HPP_

#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include <mptensor/mptensor.hpp>

namespace mptensor_test {

using namespace mptensor;

#ifdef _NO_MPI
template <typename T>
using DistMatrix = lapack::Matrix<T>;
#else
template <typename T>
using DistMatrix = scalapack::Matrix<T>;
#endif

using TensorD = Tensor<DistMatrix<double>>;
using TensorC = Tensor<DistMatrix<complex>>;

//! Tensor types every TYPED_TEST runs with.
using TensorTypes = ::testing::Types<TensorD, TensorC>;

//! Readable names ("double" / "complex") in gtest output.
struct TensorTypeNames {
  template <typename TensorType>
  static std::string GetName(int) {
    return std::is_same_v<typename TensorType::value_type, double> ? "double"
                                                                   : "complex";
  }
};

}  // namespace mptensor_test

#endif  // MPTENSOR_TEST_TENSOR_TYPES_HPP_
```

- [ ] **Step 6: Create `tests/common/test_functions.hpp`**

```cpp
#ifndef MPTENSOR_TEST_TEST_FUNCTIONS_HPP_
#define MPTENSOR_TEST_TEST_FUNCTIONS_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mptensor/mptensor.hpp>

#include "mpi_helpers.hpp"

namespace mptensor_test {

using namespace mptensor;

//! Base size of test tensors (legacy default N).
constexpr int kL = 10;
//! Default tolerance.
constexpr double kEps = 1.0e-10;

namespace detail {
inline double coord(const Index& idx, const Shape& shape, std::size_t k) {
  return double(idx[k]) / double(shape[k]);
}
}  // namespace detail

//! Analytic matrix element (legacy func2_1 / cfunc2_1).
template <typename T>
T func2_1(const Index& idx, const Shape& shape) {
  const double x0 = detail::coord(idx, shape, 0);
  const double x1 = detail::coord(idx, shape, 1);
  if constexpr (std::is_same_v<T, double>) {
    return 1.0 / (1.0 + std::abs(x0 - std::sqrt(x1)));
  } else {
    return 1.0 / (1.0 + complex(x0, std::sqrt(x1)));
  }
}

//! Analytic rank-4 element (legacy func4_1 / cfunc4_1).
template <typename T>
T func4_1(const Index& idx, const Shape& shape) {
  const double x0 = detail::coord(idx, shape, 0);
  const double x1 = detail::coord(idx, shape, 1);
  const double x2 = detail::coord(idx, shape, 2);
  const double x3 = detail::coord(idx, shape, 3);
  if constexpr (std::is_same_v<T, double>) {
    return (x0 + x1) / (1.0 + std::abs(x2 - x3));
  } else {
    return complex(x0, x1) / complex(1.0 + x2, -x3);
  }
}

//! Analytic rank-4 element (legacy func4_2 / cfunc4_2).
template <typename T>
T func4_2(const Index& idx, const Shape& shape) {
  const double x0 = detail::coord(idx, shape, 0);
  const double x1 = detail::coord(idx, shape, 1);
  const double x2 = detail::coord(idx, shape, 2);
  const double x3 = detail::coord(idx, shape, 3);
  if constexpr (std::is_same_v<T, double>) {
    return std::cos(x0 * x3) + std::sin(x1 / (x2 + 1.0));
  } else {
    return complex(std::cos(x0 * x3), std::sin(x1 / (x2 + 1.0)));
  }
}

//! Complex conjugate that keeps \c double as \c double.
inline double conj_value(double x) { return x; }
inline complex conj_value(const complex& x) { return std::conj(x); }

//! Set every local element to f(global_index, shape).
template <typename TensorType, typename F>
void fill_tensor(TensorType& t, F f) {
  const Shape shape = t.shape();
  for (std::size_t i = 0; i < t.local_size(); ++i) {
    t[i] = f(t.global_index(i), shape);
  }
}

//! Maximum over all ranks of |t[idx] - exact(idx)|. Collective.
template <typename TensorType, typename F>
double max_error(const TensorType& t, F exact) {
  double error = 0.0;
  for (std::size_t i = 0; i < t.local_size(); ++i) {
    error = std::max(error, std::abs(t[i] - exact(t.global_index(i))));
  }
  return max_over_ranks(error, t.get_comm());
}

}  // namespace mptensor_test

#endif  // MPTENSOR_TEST_TEST_FUNCTIONS_HPP_
```

- [ ] **Step 7: Create `tests/tensor/CMakeLists.txt`**

```cmake
add_executable(test_tensor transpose.cc)
target_link_libraries(test_tensor PRIVATE mptensor_gtest_main)

mptensor_add_test(tensor/test_tensor test_tensor)
```

- [ ] **Step 8: Create `tests/tensor/transpose.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Transpose : public ::testing::Test {};
TYPED_TEST_SUITE(Transpose, TensorTypes, TensorTypeNames);

// B = transpose(A, (2,0,3,1)), so B[a,b,c,d] = A[b,d,a,c].
TYPED_TEST(Transpose, PermutesAxes) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam B = transpose(A, Axes(2, 0, 3, 1));

  const double error = max_error(B, [&](const Index& b) {
    return func4_1<T>(Index(b[1], b[3], b[0], b[2]), shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 9: Configure and build with MPI**

Run:
```bash
cmake -S . -B build/gtest-mpi -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug 2>&1 | grep -E "GoogleTest|Error"
cmake --build build/gtest-mpi -j2
```
Expected: `-- GoogleTest: using FetchContent (v1.18.0)`; build succeeds.

- [ ] **Step 10: Run the tests with MPI**

Run: `ctest --test-dir build/gtest-mpi --output-on-failure`
Expected: `tensor/test_tensor_mpi0001` … `_mpi0004` all pass (4 tests); no `legacy/` tests are listed.

- [ ] **Step 11: Mutation check — the assertion really fails, on every process count**

Temporarily change `EXPECT_LT(error, kEps);` in `transpose.cc` to `EXPECT_LT(error, -1.0);`, rebuild, and run:
```bash
cmake --build build/gtest-mpi && ctest --test-dir build/gtest-mpi --output-on-failure
```
Expected: all 4 tests FAIL; output for `_mpi0002`+ contains lines starting with `[rank 1]`. Revert the change.

- [ ] **Step 12: Mutation check — a failure on rank 1 only fails the whole run and does not hang**

Temporarily append to `transpose.cc` (inside `namespace {`):
```cpp
TEST(MainSelfCheck, FailsOnRankOneOnly) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  EXPECT_NE(rank, 1);
}
```
Rebuild and run: `mpiexec -n 2 build/gtest-mpi/tests/tensor/test_tensor; echo "exit=$?"`
Expected: rank 0 output shows `[  PASSED  ]` for this test, a line `[rank 1] ...: Failure` is printed, and `exit=1`. Remove the temporary test and rebuild.

- [ ] **Step 13: Build and run without MPI**

Run:
```bash
cmake -S . -B build/gtest-nompi -G Ninja -DBUILD_TESTS=ON -DENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Debug > /dev/null
cmake --build build/gtest-nompi -j2 && ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: one test `tensor/test_tensor_serial` passes; gtest output lists `Transpose/double.PermutesAxes` and `Transpose/complex.PermutesAxes`.

- [ ] **Step 14: Legacy tests still work when enabled**

Run:
```bash
cmake -S . -B build/gtest-legacy -G Ninja -DBUILD_TESTS=ON -DBUILD_LEGACY_TESTS=ON -DCMAKE_BUILD_TYPE=Debug > /dev/null
cmake --build build/gtest-legacy -j2 && ctest --test-dir build/gtest-legacy --output-on-failure -R legacy/
```
Expected: all `legacy/` tests pass (same set as before this change). If the `${MPIEXEC}` variable used by `tests/legacy/CMakeLists.txt` turns out empty, stop and report — do not edit `tests/legacy`.

- [ ] **Step 15: Offline build with `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`**

Run:
```bash
cmake -S . -B build/gtest-offline -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=$PWD/build/gtest-mpi/_deps/googletest-src \
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON > /dev/null
cmake --build build/gtest-offline -j2 --target test_tensor
```
Expected: configure and build succeed without network access. Then `rm -rf build/gtest-offline build/gtest-legacy`.

- [ ] **Step 16: Update `README.md`**

Change `- CMake (>= 3.6)` to `- CMake (>= 3.16)`, and add before `## Documents`:

````markdown
## Tests

Tests use [GoogleTest](https://github.com/google/googletest).
An installed GoogleTest (>= 1.14) is used if found; otherwise it is downloaded at configure time.

    cmake -B build -DBUILD_TESTS=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

With MPI, each test runs with 1, 2, 3, and 4 processes.
For offline builds, pass a local GoogleTest source tree with `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.
The old test programs in `tests/legacy` are built only with `-DBUILD_LEGACY_TESTS=ON`.
````

- [ ] **Step 17: Commit**

```bash
git add CMakeLists.txt README.md tests/CMakeLists.txt tests/common tests/tensor
git status --short tests/legacy   # must print nothing
git commit -m "Introduce GoogleTest with an MPI-aware main and port the transpose test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Shape operations (`Reshape`, `Slice`, `SetSlice`)

**Files:**
- Create: `tests/tensor/reshape.cc`, `tests/tensor/slice.cc`, `tests/tensor/set_slice.cc`
- Modify: `tests/tensor/CMakeLists.txt`

**Interfaces:**
- Consumes: `TensorTypes`, `TensorTypeNames`, `kL`, `kEps`, `func4_1`, `func4_2`, `fill_tensor`, `max_error` (Task 1).

- [ ] **Step 1: Create `tests/tensor/reshape.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Reshape : public ::testing::Test {};
TYPED_TEST_SUITE(Reshape, TensorTypes, TensorTypeNames);

// B = reshape(A, (N0*N1, N2*N3)); the first index runs fastest.
TYPED_TEST(Reshape, MergesAxes) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2, N3 = kL + 3;
  TypeParam A(Shape(N0, N1, N2, N3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam B = reshape(A, Shape(N0 * N1, N2 * N3));

  const double error = max_error(B, [&](const Index& b) {
    return func4_1<T>(Index(b[0] % N0, b[0] / N0, b[1] % N2, b[1] / N2),
                      shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 2: Create `tests/tensor/slice.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Slice : public ::testing::Test {
 protected:
  // A.shape = (L, max(L+1,6), L+2, max(L+3,5)) filled with func4_1.
  static TensorType make_source() {
    using T = typename TensorType::value_type;
    TensorType A(Shape(kL, std::max(kL + 1, 6), kL + 2, std::max(kL + 3, 5)));
    fill_tensor(A, func4_1<T>);
    return A;
  }
};
TYPED_TEST_SUITE(Slice, TensorTypes, TensorTypeNames);

// B = A[:, 3:6, :, :]
TYPED_TEST(Slice, ScalarRange) {
  using T = typename TypeParam::value_type;
  const TypeParam A = TestFixture::make_source();
  const Shape shape_A = A.shape();

  TypeParam B = slice(A, 1, 3, 6);

  const double error = max_error(B, [&](Index b) {
    b[1] += 3;
    return func4_1<T>(b, shape_A);
  });
  EXPECT_LT(error, kEps);
}

// C = A[:, 2:4, :, 1:5]; begin == end (0, 0) keeps the whole axis.
TYPED_TEST(Slice, IndexRange) {
  using T = typename TypeParam::value_type;
  const TypeParam A = TestFixture::make_source();
  const Shape shape_A = A.shape();

  TypeParam C = slice(A, Index(0, 2, 0, 1), Index(0, 4, 0, 5));

  const double error = max_error(C, [&](Index c) {
    c[1] += 2;
    c[3] += 1;
    return func4_1<T>(c, shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 3: Create `tests/tensor/set_slice.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class SetSlice : public ::testing::Test {};
TYPED_TEST_SUITE(SetSlice, TensorTypes, TensorTypeNames);

// A[:, 3:6, :, :] = B and A[1:4, 0:2, :, 2:5] = C; other elements stay 1.
TYPED_TEST(SetSlice, OverwritesRanges) {
  using T = typename TypeParam::value_type;
  const int N0 = std::max(kL, 4), N1 = std::max(kL + 1, 6), N2 = kL + 2,
            N3 = std::max(kL + 3, 5);
  TypeParam A(Shape(N0, N1, N2, N3));
  TypeParam B(Shape(N0, 3, N2, N3));
  TypeParam C(Shape(3, 2, N2, 3));
  A = 1.0;
  fill_tensor(B, func4_1<T>);
  fill_tensor(C, func4_2<T>);
  const Shape shape_B = B.shape();
  const Shape shape_C = C.shape();

  A.set_slice(B, 1, 3, 6);
  A.set_slice(C, Index(1, 0, 0, 2), Index(4, 2, 0, 5));

  const double error = max_error(A, [&](Index a) -> T {
    if (a[1] >= 3 && a[1] < 6) {
      a[1] -= 3;
      return func4_1<T>(a, shape_B);
    }
    if (a[0] >= 1 && a[0] < 4 && a[1] < 2 && a[3] >= 2 && a[3] < 5) {
      a[0] -= 1;
      a[3] -= 2;
      return func4_2<T>(a, shape_C);
    }
    return T(1.0);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 4: Register the files**

Append to `tests/tensor/CMakeLists.txt`, right after the `add_executable` line:

```cmake
target_sources(test_tensor PRIVATE reshape.cc slice.cc set_slice.cc)
```

- [ ] **Step 5: Build and run (MPI)**

Run: `cmake --build build/gtest-mpi && ctest --test-dir build/gtest-mpi --output-on-failure`
Expected: 4 tests pass. Spot-check names: `mpiexec -n 1 build/gtest-mpi/tests/tensor/test_tensor --gtest_list_tests` lists `Reshape/double`, `Slice/complex`, `SetSlice/double`, etc.

- [ ] **Step 6: Build and run (no MPI)**

Run: `cmake --build build/gtest-nompi && ctest --test-dir build/gtest-nompi --output-on-failure`
Expected: `tensor/test_tensor_serial` passes.

- [ ] **Step 7: Commit**

```bash
git add tests/tensor
git commit -m "Port reshape, slice, and set_slice tests to GoogleTest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Contractions (`Tensordot`, `Contract`, `Trace`, `Trace2`, `Kron`)

**Files:**
- Create: `tests/tensor/tensordot.cc`, `tests/tensor/contract.cc`, `tests/tensor/trace.cc`, `tests/tensor/kron.cc`
- Modify: `tests/tensor/CMakeLists.txt`

**Interfaces:**
- Consumes: `TensorTypes`, `TensorTypeNames`, `kL`, `kEps`, `func2_1`, `func4_1`, `func4_2`, `fill_tensor`, `max_error` (Task 1).

- [ ] **Step 1: Create `tests/tensor/tensordot.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Tensordot : public ::testing::Test {};
TYPED_TEST_SUITE(Tensordot, TensorTypes, TensorTypeNames);

// C = tensordot(A, B, axes=([1,3],[2,0])):
// C[a,b,c,d] = sum_{m,k} A[a,m,b,k] B[k,c,m,d]
TYPED_TEST(Tensordot, ContractsTwoAxes) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2, N3 = kL + 3;
  const int M = kL + 4, K = kL + 5;
  TypeParam A(Shape(N0, M, N1, K));
  TypeParam B(Shape(K, N2, M, N3));
  const Shape shape_A = A.shape();
  const Shape shape_B = B.shape();
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);

  TypeParam C = tensordot(A, B, Axes(1, 3), Axes(2, 0));

  const double error = max_error(C, [&](const Index& c) {
    T exact = 0.0;
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        exact += func4_1<T>(Index(c[0], m, c[1], k), shape_A) *
                 func4_2<T>(Index(k, c[2], m, c[3]), shape_B);
      }
    }
    return exact;
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 2: Create `tests/tensor/contract.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Contract : public ::testing::Test {};
TYPED_TEST_SUITE(Contract, TensorTypes, TensorTypeNames);

// B = contract(A, Axes(0), Axes(2)): B[a,b] = sum_i A[i,a,i,b]
TYPED_TEST(Contract, PartialTrace) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2;
  TypeParam A(Shape(N0, N1, N0, N2));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam B = contract(A, 0, 2);

  const double error = max_error(B, [&](const Index& b) {
    T exact = 0.0;
    for (int m = 0; m < N0; ++m) {
      exact += func4_1<T>(Index(m, b[0], m, b[1]), shape_A);
    }
    return exact;
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 3: Create `tests/tensor/trace.cc`**

The legacy test also computed the matrix trace but only printed it; it is now asserted too.

```cpp
#include <cmath>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Trace : public ::testing::Test {};
TYPED_TEST_SUITE(Trace, TensorTypes, TensorTypeNames);

// trace(A, Axes(0,1), Axes(3,2)) = sum_{i,j} A[i,j,j,i]
TYPED_TEST(Trace, FullTrace) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1;
  TypeParam A(Shape(N0, N1, N1, N0));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  T exact = 0.0;
  for (int i = 0; i < N0; ++i) {
    for (int j = 0; j < N1; ++j) {
      exact += func4_1<T>(Index(i, j, j, i), shape_A);
    }
  }

  {
    SCOPED_TRACE("trace of rank-4 tensor");
    const T result = trace(A, Axes(0, 1), Axes(3, 2));
    EXPECT_LT(std::abs(result - exact) / std::abs(exact), kEps);
  }
  {
    SCOPED_TRACE("trace of reshaped matrix");
    TypeParam M = transpose(A, Axes(0, 1, 3, 2));
    const Shape s = M.shape();
    M = reshape(M, Shape(s[0] * s[1], s[2] * s[3]));
    const T result = trace(M);
    EXPECT_LT(std::abs(result - exact) / std::abs(exact), kEps);
  }
}

template <typename TensorType>
class Trace2 : public ::testing::Test {};
TYPED_TEST_SUITE(Trace2, TensorTypes, TensorTypeNames);

// trace(A, B, Axes(0,3,2,1), Axes(3,2,0,1)) = sum_{i,j,k,l} A[i,j,k,l] B[k,j,l,i]
TYPED_TEST(Trace2, FullContraction) {
  using T = typename TypeParam::value_type;
  const int N0 = kL, N1 = kL + 1, N2 = kL + 2, N3 = kL + 3;
  TypeParam A(Shape(N0, N1, N2, N3));
  TypeParam B(Shape(N2, N1, N3, N0));
  const Shape shape_A = A.shape();
  const Shape shape_B = B.shape();
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);

  const T result = trace(A, B, Axes(0, 3, 2, 1), Axes(3, 2, 0, 1));

  T exact = 0.0;
  for (int i = 0; i < N0; ++i) {
    for (int j = 0; j < N1; ++j) {
      for (int k = 0; k < N2; ++k) {
        for (int l = 0; l < N3; ++l) {
          exact += func4_1<T>(Index(i, j, k, l), shape_A) *
                   func4_2<T>(Index(k, j, l, i), shape_B);
        }
      }
    }
  }
  EXPECT_LT(std::abs(result - exact) / std::abs(exact), kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 4: Create `tests/tensor/kron.cc`**

The legacy `double` version had its assertion commented out; it is enforced here.

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Kron : public ::testing::Test {};
TYPED_TEST_SUITE(Kron, TensorTypes, TensorTypeNames);

// C = kron(A, B): C[i,j] = A[i % nA0, j % nA1] * B[i / nA0, j / nA1]
TYPED_TEST(Kron, KroneckerProduct) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1));
  TypeParam B(Shape(kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  const Shape shape_B = B.shape();
  fill_tensor(A, func2_1<T>);
  fill_tensor(B, func2_1<T>);

  TypeParam C = kron(A, B);

  const double error = max_error(C, [&](const Index& c) {
    return func2_1<T>(Index(c[0] % shape_A[0], c[1] % shape_A[1]), shape_A) *
           func2_1<T>(Index(c[0] / shape_A[0], c[1] / shape_A[1]), shape_B);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 5: Register the files**

Add to `tests/tensor/CMakeLists.txt` after the existing `target_sources` line:

```cmake
target_sources(test_tensor PRIVATE tensordot.cc contract.cc trace.cc kron.cc)
```

- [ ] **Step 6: Build and run (MPI, then no MPI)**

Run:
```bash
cmake --build build/gtest-mpi && ctest --test-dir build/gtest-mpi --output-on-failure
cmake --build build/gtest-nompi && ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: all pass. If `Kron/double` fails, stop and report (spec Section 7).

- [ ] **Step 7: Commit**

```bash
git add tests/tensor
git commit -m "Port tensordot, contract, trace, and kron tests to GoogleTest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Arithmetic (`Arithmetic`)

**Files:**
- Create: `tests/tensor/arithmetic.cc`
- Modify: `tests/tensor/CMakeLists.txt`

**Interfaces:**
- Consumes: `TensorTypes`, `TensorTypeNames`, `kL`, `kEps`, `func4_1`, `func4_2`, `fill_tensor`, `max_error` (Task 1).

The legacy `double` and `complex` versions use different scalars and expected coefficients; both are kept with `if constexpr`.

- [ ] **Step 1: Create `tests/tensor/arithmetic.cc`**

```cpp
#include <type_traits>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Arithmetic : public ::testing::Test {};
TYPED_TEST_SUITE(Arithmetic, TensorTypes, TensorTypeNames);

// A chain of +=, *=, /=, -=, binary +, -, *, / and unary minus.
TYPED_TEST(Arithmetic, OperatorChain) {
  using T = typename TypeParam::value_type;
  const Shape shape(kL, kL + 1, kL + 2, kL + 3);
  TypeParam A(shape);
  TypeParam B(shape);
  TypeParam C;
  fill_tensor(A, func4_1<T>);
  fill_tensor(B, func4_2<T>);

  T coef_1, coef_2;  // expected A = coef_1 * func4_1 + coef_2 * func4_2
  if constexpr (std::is_same_v<T, double>) {
    A += B;
    A *= 2.0;
    B /= 3.0;
    A -= B;
    A = A * 3.0;
    C = A + B;
    B = A / 2.0;
    C = C - B;
    B = -C;
    A = 0.6 * B;
    coef_1 = -1.8;
    coef_2 = -1.7;
  } else {
    A += B;
    A *= complex(0.0, 2.0);
    B /= 3.0;
    A -= B;
    A = A * 3.0;
    C = A + B;
    B = A / complex(1.0, 2.0);
    C = C - B;
    B = -C;
    A = 0.75 * B;
    coef_1 = complex(1.8, -3.6);
    coef_2 = complex(2.15, -3.3);
  }

  const double error = max_error(A, [&](const Index& a) {
    return coef_1 * func4_1<T>(a, shape) + coef_2 * func4_2<T>(a, shape);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 2: Register the file**

Add to `tests/tensor/CMakeLists.txt` after the existing `target_sources` lines:

```cmake
target_sources(test_tensor PRIVATE arithmetic.cc)
```

- [ ] **Step 3: Build and run (MPI, then no MPI)**

Run:
```bash
cmake --build build/gtest-mpi && ctest --test-dir build/gtest-mpi --output-on-failure
cmake --build build/gtest-nompi && ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add tests/tensor
git commit -m "Port arithmetic test to GoogleTest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Decompositions (`Svd`, `Qr`, `QrRank2`)

**Files:**
- Create: `tests/tensor/svd.cc`, `tests/tensor/qr.cc`
- Modify: `tests/tensor/CMakeLists.txt`

**Interfaces:**
- Consumes: `TensorTypes`, `TensorTypeNames`, `kL`, `kEps`, `func2_1`, `func4_1`, `fill_tensor`, `max_error` (Task 1).

- [ ] **Step 1: Create `tests/tensor/svd.cc`**

```cpp
#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Svd : public ::testing::Test {};
TYPED_TEST_SUITE(Svd, TensorTypes, TensorTypeNames);

// A[i,j,k,l] = sum_a U[k,i,a] S[a] V[a,j,l]
TYPED_TEST(Svd, Reconstructs) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam U, V;
  std::vector<double> S;
  svd(A, Axes(2, 0), Axes(1, 3), U, S, V);

  U.multiply_vector(S, 2);  // U[k,i,a] <= U[k,i,a] * S[a]
  TypeParam B = tensordot(U, V, Axes(2), Axes(0));  // B[k,i,j,l] = A[i,j,k,l]

  const double error = max_error(B, [&](const Index& b) {
    return func4_1<T>(Index(b[1], b[2], b[0], b[3]), shape_A);
  });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 2: Create `tests/tensor/qr.cc`**

```cpp
#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

template <typename TensorType>
class Qr : public ::testing::Test {};
TYPED_TEST_SUITE(Qr, TensorTypes, TensorTypeNames);

// A[i,j,k,l] = sum_a Q[k,i,a] R[a,j,l]
TYPED_TEST(Qr, Reconstructs) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL, kL + 1, kL + 2, kL + 3));
  const Shape shape_A = A.shape();
  fill_tensor(A, func4_1<T>);

  TypeParam Q, R;
  qr(A, Axes(2, 0), Axes(1, 3), Q, R);

  TypeParam B =
      transpose(tensordot(Q, R, Axes(2), Axes(0)), Axes(1, 2, 0, 3));

  const double error = max_error(
      B, [&](const Index& b) { return func4_1<T>(b, shape_A); });
  EXPECT_LT(error, kEps);
}

template <typename TensorType>
class QrRank2 : public ::testing::Test {};
TYPED_TEST_SUITE(QrRank2, TensorTypes, TensorTypeNames);

// Q, R = np.linalg.qr(A, mode='reduced')
TYPED_TEST(QrRank2, Reconstructs) {
  using T = typename TypeParam::value_type;
  TypeParam A(Shape(kL * (kL + 1), kL * kL));
  const Shape shape = A.shape();
  fill_tensor(A, func2_1<T>);

  TypeParam Q, R;
  qr(A, Q, R);

  TypeParam B = tensordot(Q, R, Axes(1), Axes(0));

  const double error =
      max_error(B, [&](const Index& b) { return func2_1<T>(b, shape); });
  EXPECT_LT(error, kEps);
}

}  // namespace
}  // namespace mptensor_test
```

- [ ] **Step 3: Register the files**

Add to `tests/tensor/CMakeLists.txt` after the existing `target_sources` lines:

```cmake
target_sources(test_tensor PRIVATE svd.cc qr.cc)
```

- [ ] **Step 4: Build and run (MPI, then no MPI)**

Run:
```bash
cmake --build build/gtest-mpi && ctest --test-dir build/gtest-mpi --output-on-failure
cmake --build build/gtest-nompi && ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/tensor
git commit -m "Port svd and qr tests to GoogleTest

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Eigenvalue problems (`Eigh`, `EighRank2`, `EighGeneral`)

**Files:**
- Create: `tests/tensor/eigh.cc`
- Modify: `tests/tensor/CMakeLists.txt`

**Interfaces:**
- Consumes: `TensorTypes`, `TensorTypeNames`, `kL`, `kEps`, `func2_1`, `func4_1`, `func4_2`, `conj_value`, `fill_tensor` (Task 1); library `conj(Tensor)`, `max_abs(Tensor)` (global maximum).

Legacy compared `B` with `A.get_value(index_B)`, which is only valid when both tensors share the same distribution. These tests compare with `max_abs(B - A)` instead (as legacy `eigh_general` already did); `max_abs` reduces over all ranks.

`EighGeneral` was disabled in legacy for an unknown reason. **If it fails, stop and report to the user** (spec Section 7); only with the user's agreement rename it to `DISABLED_Solves`.

- [ ] **Step 1: Create `tests/tensor/eigh.cc`**

```cpp
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "tensor_types.hpp"
#include "test_functions.hpp"

namespace mptensor_test {
namespace {

// Hermitian matrix A[i,j] = f(i,j) + conj(f(j,i)) with f = func2_1.
template <typename TensorType>
TensorType make_hermitian_matrix(int n) {
  using T = typename TensorType::value_type;
  TensorType A(Shape(n, n));
  fill_tensor(A, [](const Index& idx, const Shape& shape) {
    return func2_1<T>(idx, shape) +
           conj_value(func2_1<T>(Index(idx[1], idx[0]), shape));
  });
  return A;
}

template <typename TensorType>
class Eigh : public ::testing::Test {};
TYPED_TEST_SUITE(Eigh, TensorTypes, TensorTypeNames);

// A[i,j,k,l] = sum_a Z[i,k,a] W[a] conj(Z[l,j,a])
TYPED_TEST(Eigh, Reconstructs) {
  TypeParam A = make_hermitian_matrix<TypeParam>(kL * kL);
  A = transpose(reshape(A, Shape(kL, kL, kL, kL)), Axes(0, 3, 1, 2));

  TypeParam Z;
  std::vector<double> W;
  eigh(A, Axes(0, 2), Axes(3, 1), W, Z);

  TypeParam B = conj(Z);
  Z.multiply_vector(W, 2);  // Z[i,k,a] <= Z[i,k,a] * W[a]
  B = transpose(tensordot(Z, B, Axes(2), Axes(2)), Axes(0, 3, 1, 2));

  EXPECT_LT(max_abs(B - A), kEps);
}

template <typename TensorType>
class EighRank2 : public ::testing::Test {};
TYPED_TEST_SUITE(EighRank2, TensorTypes, TensorTypeNames);

// A[i,j] = sum_a Z[i,a] W[a] conj(Z[j,a])
TYPED_TEST(EighRank2, Reconstructs) {
  const TypeParam A = make_hermitian_matrix<TypeParam>(kL * kL);

  TypeParam Z;
  std::vector<double> W;
  eigh(A, W, Z);

  TypeParam B = conj(Z);
  Z.multiply_vector(W, 1);  // Z[i,a] <= Z[i,a] * W[a]
  B = tensordot(Z, B, Axes(1), Axes(1));

  EXPECT_LT(max_abs(B - A), kEps);
}

template <typename TensorType>
class EighGeneral : public ::testing::Test {};
TYPED_TEST_SUITE(EighGeneral, TensorTypes, TensorTypeNames);

// A[i,j,k,l] Z[l,k,a] = B[j,l,i,k] Z[l,k,a] W[a]
TYPED_TEST(EighGeneral, Solves) {
  using T = typename TypeParam::value_type;
  constexpr double kEpsGeneral = 1.0e-8;
  constexpr bool is_double = std::is_same_v<T, double>;

  // Hermitian A: double uses func4_1, complex uses func4_2 (as in legacy).
  TypeParam A(Shape(kL, kL, kL, kL));
  fill_tensor(A, [](const Index& idx, const Shape& shape) {
    const Index idx2(idx[2], idx[3], idx[0], idx[1]);
    if constexpr (is_double) {
      return func4_1<T>(idx, shape) + func4_1<T>(idx2, shape);
    } else {
      return func4_2<T>(idx, shape) + conj_value(func4_2<T>(idx2, shape));
    }
  });

  // Positive-definite B = U S U^dagger, with small singular values lifted.
  TypeParam B(Shape(kL, kL, kL, kL));
  if constexpr (is_double) {
    fill_tensor(B, func4_2<T>);
  } else {
    fill_tensor(B, func4_1<T>);
  }
  {
    TypeParam u, vt;
    std::vector<double> s;
    svd(B, Axes(0, 1), Axes(2, 3), u, s, vt);
    for (std::size_t i = 0; i < s.size(); ++i) {
      if (s[i] < 1.0e-3) s[i] = 1.0e-3 * (i + 1);
    }
    TypeParam us = u;
    us.multiply_vector(s, 2);
    B = transpose(tensordot(us, conj(u), Axes(2), Axes(2)), Axes(1, 3, 0, 2));
  }

  TypeParam Z;
  std::vector<double> W;
  eigh(A, Axes(1, 0), Axes(3, 2), B, Axes(0, 2), Axes(1, 3), W, Z);

  TypeParam lhs =
      tensordot(A.transpose(Axes(1, 0, 3, 2)), Z, Axes(2, 3), Axes(0, 1));
  TypeParam rhs =
      tensordot(B.transpose(Axes(0, 2, 1, 3)), Z, Axes(2, 3), Axes(0, 1));
  rhs.multiply_vector(W, 2);

  EXPECT_LT(max_abs(lhs - rhs), kEpsGeneral);
}

}  // namespace
}  // namespace mptensor_test
```

Note: in the legacy `double` version, `B = transpose(tensordot(us, u, ...))` has no `conj`; `conj` of a real tensor is the identity, so one code path covers both.

- [ ] **Step 2: Register the file**

Add to `tests/tensor/CMakeLists.txt` after the existing `target_sources` lines:

```cmake
target_sources(test_tensor PRIVATE eigh.cc)
```

- [ ] **Step 3: Build and run (MPI, then no MPI)**

Run:
```bash
cmake --build build/gtest-mpi && ctest --test-dir build/gtest-mpi --output-on-failure
cmake --build build/gtest-nompi && ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: all pass. If `EighGeneral` fails, stop and report the output (which type, which process counts, the error value).

- [ ] **Step 4: Commit**

```bash
git add tests/tensor
git commit -m "Port eigh tests to GoogleTest and enable the generalized eigenproblem test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: File I/O (`Save`, `Load`)

**Files:**
- Create: `tests/file_io/CMakeLists.txt`, `tests/file_io/prepare_data_dir.cmake`, `tests/file_io/file_io_common.hpp`, `tests/file_io/save.cc`, `tests/file_io/load.cc`
- Modify: `tests/CMakeLists.txt` (add `add_subdirectory(file_io)`)

**Interfaces:**
- Consumes: `mptensor_gtest_main`, `mptensor_add_test` (with `WORKING_DIRECTORY`, `FIXTURES_SETUP`, `FIXTURES_REQUIRED`), `TensorD`, `TensorC`, `world_size()` (Task 1).
- Produces (inside `file_io_common.hpp`): `kFileIoN`, `LocalTensorD`, `LocalTensorC`, `make_initial_tensor<TensorType>()`, `data_filename(prefix, proc_size)`, `binary_filename(base, rank)`.

- [ ] **Step 1: Create `tests/file_io/file_io_common.hpp`**

```cpp
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
```

- [ ] **Step 2: Create `tests/file_io/save.cc`**

```cpp
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
```

- [ ] **Step 3: Create `tests/file_io/load.cc`**

```cpp
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
```

- [ ] **Step 4: Create `tests/file_io/prepare_data_dir.cmake`**

```cmake
# Usage: cmake -DDATA_DIR=<dir> -P prepare_data_dir.cmake
# Starts every file_io run from an empty directory, so that stale files
# from an earlier run cannot hide a failure of save().
if(NOT DATA_DIR)
  message(FATAL_ERROR "DATA_DIR is not set")
endif()
file(REMOVE_RECURSE "${DATA_DIR}")
file(MAKE_DIRECTORY "${DATA_DIR}")
```

- [ ] **Step 5: Create `tests/file_io/CMakeLists.txt`**

```cmake
set(FILE_IO_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${FILE_IO_DATA_DIR})

add_executable(test_save save.cc)
target_link_libraries(test_save PRIVATE mptensor_gtest_main)
add_executable(test_load load.cc)
target_link_libraries(test_load PRIVATE mptensor_gtest_main)

# prepare -> save (all process counts) -> load (all process counts)
add_test(NAME file_io/prepare
  COMMAND ${CMAKE_COMMAND} -DDATA_DIR=${FILE_IO_DATA_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/prepare_data_dir.cmake)
set_tests_properties(file_io/prepare PROPERTIES FIXTURES_SETUP file_io_dir)

mptensor_add_test(file_io/save test_save
  WORKING_DIRECTORY ${FILE_IO_DATA_DIR}
  FIXTURES_REQUIRED file_io_dir
  FIXTURES_SETUP file_io_data)
mptensor_add_test(file_io/load test_load
  WORKING_DIRECTORY ${FILE_IO_DATA_DIR}
  FIXTURES_REQUIRED file_io_data)
```

And in `tests/CMakeLists.txt`, after `add_subdirectory(tensor)`, add:

```cmake
add_subdirectory(file_io)
```

- [ ] **Step 6: Build and run (MPI)**

Run:
```bash
cmake -S . -B build/gtest-mpi > /dev/null && cmake --build build/gtest-mpi
ctest --test-dir build/gtest-mpi --output-on-failure -R file_io/
```
Expected: 9 tests pass in this order: `file_io/prepare`, `file_io/save_mpi0001`…`0004`, `file_io/load_mpi0001`…`0004`.

- [ ] **Step 7: Verify the fixtures (ordering and stale data)**

Run:
```bash
ctest --test-dir build/gtest-mpi --output-on-failure -R file_io/load_mpi0002
touch build/gtest-mpi/tests/file_io/data/stale_marker
ctest --test-dir build/gtest-mpi -R file_io/ > /dev/null
ls build/gtest-mpi/tests/file_io/data/stale_marker
ctest --test-dir build/gtest-mpi --output-on-failure -j4 -R file_io/
```
Expected: the first command also runs `file_io/prepare` and all `file_io/save_*` before the load test, and passes; `ls` reports `No such file or directory`; the parallel run passes.

- [ ] **Step 8: Build and run (no MPI)**

Run:
```bash
cmake -S . -B build/gtest-nompi > /dev/null && cmake --build build/gtest-nompi
ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: `file_io/prepare`, `file_io/save_serial`, `file_io/load_serial`, `tensor/test_tensor_serial` pass.

- [ ] **Step 9: Commit**

```bash
git add tests/CMakeLists.txt tests/file_io
git commit -m "Port file I/O tests to GoogleTest with CTest fixtures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: CI matrix

**Files:**
- Modify: `.github/workflows/build.yml`

**Interfaces:**
- Consumes: CMake options `ENABLE_MPI`, `BUILD_TESTS`, `BUILD_LEGACY_TESTS`; configure message `GoogleTest: found ...` (Task 1).

- [ ] **Step 1: Replace `.github/workflows/build.yml`**

```yaml
name: build

on:
  push:
    branches: [ "master", "develop" ]
  pull_request:
    branches: [ "master", "develop" ]

jobs:
  build:
    name: ${{ matrix.build_type }}, MPI ${{ matrix.mpi }}, gtest ${{ matrix.gtest }}
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        include:
          - { build_type: Debug,   mpi: "ON",  gtest: fetch,  legacy: "ON" }
          - { build_type: Release, mpi: "ON",  gtest: system, legacy: "OFF" }
          - { build_type: Debug,   mpi: "OFF", gtest: fetch,  legacy: "OFF" }
          - { build_type: Release, mpi: "OFF", gtest: fetch,  legacy: "OFF" }

    steps:
    - uses: actions/checkout@v6

    - name: apt
      run: |
        sudo apt update
        PACKAGES="cmake libblas-dev liblapack-dev"
        if [ "${{ matrix.mpi }}" = "ON" ]; then
          PACKAGES="$PACKAGES libopenmpi-dev libscalapack-openmpi-dev"
        fi
        if [ "${{ matrix.gtest }}" = "system" ]; then
          PACKAGES="$PACKAGES libgtest-dev"
        fi
        sudo apt install -y $PACKAGES

    - name: Configure CMake
      run: >
        cmake -B ${{github.workspace}}/build
        -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}
        -DENABLE_MPI=${{ matrix.mpi }}
        -DBUILD_TESTS=ON
        -DBUILD_LEGACY_TESTS=${{ matrix.legacy }}
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -DMPIEXEC_PREFLAGS="--oversubscribe"
        | tee configure.log

    - name: Check GoogleTest source
      if: matrix.gtest == 'system'
      run: grep "GoogleTest: found" configure.log

    - name: Build
      run: cmake --build ${{github.workspace}}/build --config ${{ matrix.build_type }} -j4

    - name: Test
      working-directory: ${{github.workspace}}/build
      run: ctest -C ${{ matrix.build_type }} --output-on-failure
```

- [ ] **Step 2: Validate the YAML locally**

Run: `python3 -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/build.yml')); print(len(d['jobs']['build']['strategy']['matrix']['include']))"`
Expected: `4`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/build.yml
git commit -m "Run CI over build type, MPI, and GoogleTest source

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Ask the user before pushing**

CI runs only on pushes to `master`/`develop` and on pull requests into them. Ask the user whether to push `feature/googletest` (`git push -u origin feature/googletest`) and open a pull request into `develop` to run the matrix. Do not push without approval.

---

### Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Release builds, with and without MPI**

Run:
```bash
cmake -S . -B build/gtest-mpi-rel -G Ninja -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build/gtest-mpi-rel -j2 && ctest --test-dir build/gtest-mpi-rel --output-on-failure
cmake -S . -B build/gtest-nompi-rel -G Ninja -DBUILD_TESTS=ON -DENABLE_MPI=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build/gtest-nompi-rel -j2 && ctest --test-dir build/gtest-nompi-rel --output-on-failure
```
Expected: all pass (MPI: 4 `tensor/` + 9 `file_io/` tests; no MPI: 4 tests).

- [ ] **Step 2: Debug builds once more, including legacy**

Run:
```bash
cmake -S . -B build/gtest-mpi -DBUILD_LEGACY_TESTS=ON > /dev/null
cmake --build build/gtest-mpi -j2 && ctest --test-dir build/gtest-mpi --output-on-failure
cmake --build build/gtest-nompi -j2 && ctest --test-dir build/gtest-nompi --output-on-failure
```
Expected: all pass, including `legacy/` tests.

- [ ] **Step 3: Examples still build**

Examples are added with `EXCLUDE_FROM_ALL`; Ninja exposes them as the directory target `examples/all`.

Run:
```bash
ninja -C build/gtest-mpi examples/all
ninja -C build/gtest-nompi examples/all
```
Expected: both builds succeed.

- [ ] **Step 4: Legacy untouched and coverage complete**

Run:
```bash
git diff --stat origin/develop -- tests/legacy        # must print nothing
mpiexec -n 1 build/gtest-mpi/tests/tensor/test_tensor --gtest_list_tests | grep -E '^[A-Za-z]' | sed 's|/.*||' | sort -u
```
Expected: no diff; the suite list is exactly `Arithmetic Contract Eigh EighGeneral EighRank2 Kron Qr QrRank2 Reshape SetSlice Slice Svd Tensordot Trace Trace2 Transpose` (16 suites, matching spec Section 4.3).

- [ ] **Step 5: Clean up extra build directories**

Run: `rm -rf build/gtest-mpi-rel build/gtest-nompi-rel`
