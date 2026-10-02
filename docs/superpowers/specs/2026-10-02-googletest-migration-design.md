# Migrating Tests to GoogleTest: Design

- Date: 2026-10-02
- Branch: `feature/googletest` (from `origin/develop`)

## 1. Goals

Introduce GoogleTest and reimplement every test in `tests/legacy` on top of it.

- Tests must verify results with GoogleTest assertions, so that they are effective in Release builds as well (the legacy tests rely on `assert`, which `NDEBUG` disables).
- A failure in one test must not stop the remaining tests (the legacy tests abort on the first failed `assert`).
- Tests run under MPI with 1, 2, 3, and 4 processes, and also without MPI.
- `tests/legacy` is kept as an archive and is not modified.

This work precedes the signed `Index` / negative indexing change (spec `2026-10-02-signed-index-design.md` on the local `develop` branch), which is postponed. That spec's test section will be rewritten on top of this infrastructure later.

## 2. Scope

### In scope

- GoogleTest acquisition in CMake (`find_package`, falling back to `FetchContent`).
- A shared MPI-aware `main` and shared test helpers.
- Reimplementation of all `test_all` tests (including the currently disabled `eigh_general`) and of the `file_io` tests.
- A `BUILD_LEGACY_TESTS` option (default `OFF`).
- A CI matrix over build type and MPI.

### Out of scope

- Any modification of the files in `tests/legacy`.
- Timing measurements, `print_info` output, and the command-line size argument `N` of the legacy tests.
- New tests beyond what `tests/legacy` covers.
- Any change to the library itself, except for fixes needed when a newly enforced check fails (see Section 7).

## 3. Build system

### 3.1 Top-level `CMakeLists.txt`

- Change `cmake_minimum_required` to `VERSION 3.16...3.31`.
- Add `option(BUILD_LEGACY_TESTS "Build legacy tests" OFF)`. `tests/legacy` is built and registered only when both `BUILD_TESTS` and `BUILD_LEGACY_TESTS` are `ON`.

### 3.2 GoogleTest acquisition (`tests/CMakeLists.txt`)

1. `find_package(GTest 1.14 CONFIG QUIET)`.
2. If not found, fetch GoogleTest `v1.18.0` with `FetchContent`, with `INSTALL_GTEST=OFF` and `BUILD_GMOCK=OFF`. In offline environments, a local source tree can be given with `FETCHCONTENT_SOURCE_DIR_GOOGLETEST`.
3. Print which method was used with `message(STATUS ...)`.

Tests use only features available in GoogleTest 1.14 (the version shipped with Ubuntu 24.04), such as `TYPED_TEST`, `EXPECT_THROW`, `SCOPED_TRACE`, and test event listeners.

### 3.3 Layout

```
tests/
├── CMakeLists.txt          # GoogleTest acquisition, common settings, add_subdirectory
├── common/
│   ├── gtest_mpi_main.cc   # shared MPI-aware main
│   ├── mpi_helpers.hpp     # e.g. max over all ranks (replaces legacy mpi_tool)
│   ├── test_functions.hpp  # analytic functions (replaces legacy functions.hpp)
│   └── tensor_types.hpp    # type lists for TYPED_TEST
├── tensor/                 # counterpart of legacy test_all
│   ├── CMakeLists.txt
│   └── transpose.cc, reshape.cc, ...  (one file per operation)
├── file_io/                # counterpart of legacy file_io
│   ├── CMakeLists.txt
│   └── save.cc, load.cc
└── legacy/                 # unchanged; built only with BUILD_LEGACY_TESTS=ON
```

The new tests do not include any file from `tests/legacy`; the helpers are rewritten in `tests/common`.

### 3.4 `gtest_mpi_main.cc`

- With MPI: `MPI_Init` → `testing::InitGoogleTest` → `RUN_ALL_TESTS` → `MPI_Allreduce` (logical OR of failures) → `MPI_Finalize`. If any rank fails, every rank returns a non-zero exit code.
- On ranks other than 0, remove the default result printer and register a listener that prints only failed assertions, prefixed with `[rank N]`.
- When `_NO_MPI` is defined, it behaves as a normal gtest `main`.

### 3.5 CTest registration

- Tests are registered per executable, not per test case, because `gtest_discover_tests` does not combine well with `mpiexec`. The name of a failed test case appears in the gtest output.
- With MPI: one test per process count, `${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} N ${MPIEXEC_PREFLAGS} <exe>` for N = 1, 2, 3, 4, named `<dir>/<name>_mpi000N`.
- Without MPI: one test named `<dir>/<name>_serial`.

## 4. `tests/tensor`

### 4.1 Test structure

- `tensor_types.hpp` defines `TensorTypes`: `Tensor<scalapack::Matrix<double>>` and `Tensor<scalapack::Matrix<complex>>` with MPI, `Tensor<lapack::Matrix<double>>` and `Tensor<lapack::Matrix<complex>>` without.
- Each file uses `TYPED_TEST_SUITE(<Suite>, TensorTypes)`. The legacy pairs `test_X` / `test_X_complex` are merged into one template.
- `test_functions.hpp` provides the analytic functions used to fill tensors, selecting the `double` or `complex` variant by value type (counterparts of legacy `func2_1`, `func4_1`, `func4_2`, `cfunc2_1`, `cfunc4_1`, `cfunc4_2`).
- Verification: each rank computes its local maximum error → `MPI_Allreduce` (MAX) → every rank evaluates `EXPECT_LT(max_error, eps)`. Collective calls therefore match on all ranks, and all ranks reach the same verdict.
- When a legacy test checks several things in one function (e.g. addition, subtraction, and scalar multiplication in `arithmetic`), each check becomes a separate assertion, with `SCOPED_TRACE` identifying it.
- All files in `tests/tensor` are linked into a single executable `test_tensor`.

### 4.2 Sizes and tolerances

Keep the legacy defaults: `L = 10`, the same tensor shapes as each legacy test (e.g. `(L, L+1, L+2, L+3)`), and the same tolerances (`1e-10`; `1e-8` for `eigh_general`).

### 4.3 Mapping

| Legacy file | New file | Test suites |
|---|---|---|
| `transpose.cc` | `tensor/transpose.cc` | `Transpose` |
| `reshape.cc` | `tensor/reshape.cc` | `Reshape` |
| `slice.cc` | `tensor/slice.cc` | `Slice` |
| `set_slice.cc` | `tensor/set_slice.cc` | `SetSlice` |
| `tensordot.cc` | `tensor/tensordot.cc` | `Tensordot` |
| `svd.cc` | `tensor/svd.cc` | `Svd` |
| `qr.cc`, `qr_rank2.cc` | `tensor/qr.cc` | `Qr`, `QrRank2` |
| `eigh.cc`, `eigh_rank2.cc`, `eigh_general.cc` | `tensor/eigh.cc` | `Eigh`, `EighRank2`, `EighGeneral` |
| `arithmetic.cc` | `tensor/arithmetic.cc` | `Arithmetic` |
| `trace.cc` | `tensor/trace.cc` | `Trace`, `Trace2` |
| `contract.cc` | `tensor/contract.cc` | `Contract` |
| `kron.cc` | `tensor/kron.cc` | `Kron` |

## 5. `tests/file_io`

Tensor data is saved in binary form, so loaded values are compared exactly (`EXPECT_EQ(max_abs(t - t0), 0.0)`). The initial tensor is the same as in legacy `common.hpp`: shape `(n, n+1, n+2, n+3)` with `n = 6`, values `i0 + 100 i1 + 10000 i2 + 1000000 i3`, followed by `transpose({3, 1, 0, 2})`.

- `test_save`:
  - Saves the distributed tensors (`pd`, `pz`) as `pd_mpi000N`, `pz_mpi000N`.
  - With 1 process or without MPI, also saves the non-distributed tensors (`sd`, `sz`).
  - Checks that the saved files exist.
- `test_load` loads the same combinations as legacy and compares them with the initial tensor:
  - With MPI: `pd` / `pz` saved with 1–4 processes and `sd` / `sz`, loaded as distributed tensors; with 1 process, additionally loaded as non-distributed tensors.
  - Without MPI: `sd` / `sz`.
- CTest registration:
  - `file_io/save_mpi000{1..4}` with `FIXTURES_SETUP file_io_data`.
  - `file_io/load_mpi000{1..4}` with `FIXTURES_REQUIRED file_io_data`.
  - Without MPI: `file_io/save_serial` and `file_io/load_serial` with the same fixture relation.
  - Working directory: `${CMAKE_CURRENT_BINARY_DIR}/data` (created at configure time).

## 6. CI (`.github/workflows/build.yml`)

Use `strategy.matrix` with `fail-fast: false`:

| Job | Build type | MPI | GoogleTest | Legacy tests |
|---|---|---|---|---|
| 1 | Debug | ON | FetchContent | `BUILD_LEGACY_TESTS=ON` |
| 2 | Release | ON | `apt install libgtest-dev` (1.14) | OFF |
| 3 | Debug | OFF | FetchContent | OFF |
| 4 | Release | OFF | FetchContent | OFF |

- MPI jobs install `libopenmpi-dev libblas-dev liblapack-dev libscalapack-openmpi-dev` and keep `-DMPIEXEC_PREFLAGS="--oversubscribe"`.
- Non-MPI jobs install only `libblas-dev liblapack-dev` and configure with `-DENABLE_MPI=OFF`.
- Legacy tests are meaningful only with `assert` enabled, so they run only in the Debug + MPI job, to keep checking that the archive still builds and passes.
- Replace `ctest -V` with `ctest --output-on-failure`.

## 7. Known risks

Two legacy checks were never actually enforced:

- `kron.cc`: the `double` version has its `assert(error < EPS)` commented out.
- `file_io/load.cc`: prints the error without checking it.

In addition, `eigh_general` is disabled in legacy `tensor_test.cc` for an unknown reason.

Also, builds without MPI (`ENABLE_MPI=OFF`) and Release builds have never been tested in CI, so jobs 2–4 may expose existing problems.

The new tests enforce all of them. If any fails, investigate whether the cause is in the test or in the library, and consult before changing the library or disabling the test (as a fallback, `eigh_general` may be marked `DISABLED_`).

## 8. Completion criteria

- All four CI jobs pass.
- In job 1, the legacy tests also pass.
- `tests/legacy` has no changes.
