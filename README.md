# mptensor

[![GitHub](https://img.shields.io/github/license/smorita/mptensor)][License]
[![build](https://github.com/smorita/mptensor/actions/workflows/build.yml/badge.svg)](https://github.com/smorita/mptensor/actions/workflows/build.yml)
[![docs](https://github.com/smorita/mptensor/actions/workflows/docs.yml/badge.svg)](https://smorita.github.io/mptensor/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3514466.svg)](https://doi.org/10.5281/zenodo.3514466)


"mptensor" is parallel C++ library for tensor calculations.
It provides similar interfaces as Numpy and Scipy in Python.

## Prerequisites

- C++17 compiler
- CMake (>= 3.16)
- [LAPACK](https://www.netlib.org/lapack/)

### For parallel computing

- MPI Library
- [ScaLAPACK](https://www.netlib.org/scalapack/)

### For document generation

- Doxygen (>= 1.9.8)

## How to Use

### Build and install

    cmake -B build
    cmake --build build
    sudo cmake --install build

The default install directory is `/usr/local`. It can be changed with
`cmake --install build --prefix your_install_prefix` (or `-DCMAKE_INSTALL_PREFIX` at configure time).

Main options:

- `-DMPTENSOR_ENABLE_MPI=OFF`: build without MPI and ScaLAPACK (default: `ON`).
  The old name `ENABLE_MPI` is still accepted but deprecated.
- `-DBUILD_SHARED_LIBS=ON`: build a shared library (default: static).

See also the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html).

### Use from a CMake project

Link the `mptensor::mptensor` target. It carries the include path, the C++17
requirement, `_NO_MPI` for a non-MPI build, and the MPI, OpenMP, LAPACK, and
ScaLAPACK libraries, so no other settings are needed.

- **Installed mptensor** (`find_package`)

        find_package(mptensor 0.6 REQUIRED)
        target_link_libraries(myapp PRIVATE mptensor::mptensor)

  Set `CMAKE_PREFIX_PATH` to the install prefix if it is not a standard location.
  The variable `mptensor_ENABLE_MPI` tells whether it was built with MPI.

- **Git submodule** (`add_subdirectory`)

        set(MPTENSOR_ENABLE_MPI ON)
        add_subdirectory(external/mptensor)
        target_link_libraries(myapp PRIVATE mptensor::mptensor)

  As a subproject, mptensor is not installed together with your project
  (`MPTENSOR_INSTALL` defaults to `OFF`).

- **FetchContent**

        include(FetchContent)
        FetchContent_Declare(mptensor
          GIT_REPOSITORY https://github.com/smorita/mptensor.git
          GIT_TAG v0.6.0)
        FetchContent_MakeAvailable(mptensor)
        target_link_libraries(myapp PRIVATE mptensor::mptensor)

Without CMake, add the include directory, link `-lmptensor` with LAPACK (and
ScaLAPACK and MPI), and define `_NO_MPI` for a build without MPI.

## Tests

Tests use [GoogleTest](https://github.com/google/googletest).
An installed GoogleTest (>= 1.14) is used if found; otherwise it is downloaded at configure time.

    cmake -B build -DBUILD_TESTS=ON
    cmake --build build
    ctest --test-dir build --output-on-failure

With MPI, each test runs with 1, 2, 3, and 4 processes.
For offline builds, pass a local GoogleTest source tree with `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.
The old test programs in `tests/legacy` are built only with `-DBUILD_LEGACY_TESTS=ON`.

## Documents

The HTML documents are available in [here][Documents].

## Examples

    #include <mptensor/mptensor.hpp>
    using namespace mptensor;
    DTensor A(Shape(3, 4, 5));  // real (double) tensor
    ZTensor B(Shape(3, 4, 5));  // complex tensor

`DTensor` and `ZTensor` are distributed with ScaLAPACK when MPI is enabled,
and use LAPACK on a single process otherwise.

Example codes of TRG and HOTRG for the 2D Ising model are in `examples/Ising_2D`.

## License

GNU Lesser General Public License v3.0 (see [LICENSE][License])

## Links

- [TeNeS](https://www.pasums.issp.u-tokyo.ac.jp/tenes/en): Parallel tensor network solver for 2D quantum lattice systems
- [Tensordot](https://github.com/smorita/Tensordot): Code generator for tensor contraction
- [cuscalapack](https://github.com/smorita/cuscalapack): pdgemm and pzgemm with cuBLAS

[Documents]: https://smorita.github.io/mptensor/
[License]: https://github.com/smorita/mptensor/blob/master/LICENSE
[TravisCI]: https://travis-ci.org/smorita/mptensor
