# mptensor v0.1

mptensor is parallel C++ libarary for tensor calculations.
It provides similar interfaces as Numpy and Scipy in Python.

## Requirements
- MPI Library
- [ScaLAPACK](http://www.netlib.org/scalapack/)
- (For RSVD) C++11 std::random, [Boost C++ library](http://www.boost.org/),
  or [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/)

## How to Use
- Modify `Makefile.option` as your environment and then `make`.
- Include `src/mptensor.hpp` in your codes.
- Complie your applications with `src/libmptensor.a` .

## Documents
By `make doc`, HTML and LaTeX documents are generated in `doxygen_docs`.


## Examples

```cpp
#include <mptensor.hpp>
using namespace mptensor;
typedef Tensor<scalapack::Matrix,double> ptensor;
ptensor A(Shape(3,4,5));
```


## Links

- [Tensordot](https://github.com/smorita/Tensordot):
Code generator for tensor contraction

- [cuscalapack](https://github.com/smorita/cuscalapack):
pdgemm and pzgemm with cuBLAS
