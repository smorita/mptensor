# mptensor v0.2

[![GitHub](https://img.shields.io/github/license/smorita/mptensor)](LICENSE)
[![Build Status](https://travis-ci.org/smorita/mptensor.svg?branch=master)](https://travis-ci.org/smorita/mptensor)

"mptensor" is parallel C++ libarary for tensor calculations.
It provides similar interfaces as Numpy and Scipy in Python.

## Requirements

- MPI Library
- [ScaLAPACK](http://www.netlib.org/scalapack/)
- (For RSVD) C++11 std::random, [Boost C++ library](http://www.boost.org/),
  or [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/)

## How to Use

1. Compile mptensor
    - Modify `Makefile.option` as your environment and then `make`.
    - or use `cmake`
2. Include `src/mptensor.hpp` in your codes.
3. Complie your applications with `src/libmptensor.a` .

## Documents

The HTML documents are available in [here](https://smorita.github.io/mptensor/).

By `make doc`, HTML and LaTeX documents are generated in `doxygen_docs`.

## Examples

    #include <mptensor.hpp>
    using namespace mptensor;
    typedef Tensor<scalapack::Matrix,double> ptensor;
    ptensor A(Shape(3,4,5));

Example codes of TRG and HOTRG for the 2D Ising model are in `examples/Ising_2D`.

## License

GNU Lesser General Public License v3.0 (see [LICENSE](./LICENSE))

## Links

- [Tensordot](https://github.com/smorita/Tensordot): Code generator for tensor contraction
- [cuscalapack](https://github.com/smorita/cuscalapack): pdgemm and pzgemm with cuBLAS

[Documents]: https://smorita.github.io/mptensor/
[LICENSE]: ./LICENSE
