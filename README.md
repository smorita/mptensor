# mptensor

[![GitHub](https://img.shields.io/github/license/smorita/mptensor)][License]
[![Build Status](https://travis-ci.org/smorita/mptensor.svg?branch=master)][TravisCI]
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3735474.svg)](https://doi.org/10.5281/zenodo.3735474)

"mptensor" is parallel C++ libarary for tensor calculations.
It provides similar interfaces as Numpy and Scipy in Python.

## Prerequisites

- C++11 compiler
- CMake (>= 3.6)
- [LAPACK](https://www.netlib.org/lapack/)

### For parallel computing

- MPI Library
- [ScaLAPACK](https://www.netlib.org/scalapack/)

### For document generation

- Doxygen (< 1.8.12)

## How to Use

1. Build and install mptensor library

        mkdir build
        cd build
        cmake ../
        make
        sudo cmake install

2. Include the header file `mptensor/mptensor.hpp` in your codes.
3. Complie your applications with link option `-lmptensor` .

The default install directory is `/usr/local`. It can be changed by `-DCMAKE_INSTALL_PREFIX` option.

    $ cmake -DCMAKE_INSTALL_PREFIX=/home/smorita/.local ../

See also the [cmake page](https://cmake.org/cmake/help/v3.6/variable/CMAKE_INSTALL_PREFIX.html).

## Documents

The HTML documents are available in [here][Documents].

## Examples

    #include <mptensor.hpp>
    using namespace mptensor;
    typedef Tensor<scalapack::Matrix,double> ptensor;
    ptensor A(Shape(3,4,5));

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
