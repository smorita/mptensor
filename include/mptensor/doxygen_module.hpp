/*
  mptensor - Parallel Library for Tensor Network Methods

  Copyright 2016 Satoshi Morita

  mptensor is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as
  published by the Free Software Foundation, either version 3 of the
  License, or (at your option) any later version.

  mptensor is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with mptensor.  If not, see
  <https://www.gnu.org/licenses/>.
*/

/*!
  \file   doxygen_module.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Sep 4 2019
  \brief  Define modules for doxygen
*/

/*!
  \defgroup Tensor Tensor class
  \{
    \defgroup TensorConstructor Constructors
    \defgroup TensorOps Tensor operations
    \{
      \defgroup ShapeChange Shape change
      Operations in order to change the shape of a tensor
      \defgroup LinearAlgebra Linear algebra
      Operations for linear algebra.
      \{
        \defgroup Decomposition Decompositions
        Functions to decompose a tensor into some tensors.
        \defgroup LinearEq Linear equation
        Functions to solve a linear equation.
      \}
      \defgroup Arithmetic Arithmetic operations
      Functions for arithmetic operations.
      \defgroup Misc Useful operations
      Other useful operations.
      \defgroup Output Output
      Function to output information of a tensor.
      \defgroup Random Randomized algorithm
      Function to decompose a tensor by randomized algorithms.
    \}
  \}
  \defgroup Index Index, Axes and Shape
  Lists of integers that specify element indices, axes, and shapes.

  \c Index, \c Axes, and \c Shape are aliases of the same class,
  \c BasicIndex<std::ptrdiff_t>. Create one like a python list:
  <tt>Index(0, 1, 2)</tt>, <tt>Axes{2, 0, 1}</tt>, or <tt>Axes a = 2;</tt>.

  Negative values count from the end, as in numpy:
  - axes: \c -1 is the last axis, e.g. <tt>transpose(A, Axes(-1, 0, 1))</tt>;
  - element indices: \c -1 is the last element,
    e.g. <tt>A.get_value(Index(-1, 0), v)</tt>;
  - slices: the end may be negative, e.g. <tt>slice(A, 0, 1, -1)</tt> is
    <tt>A[1:-1]</tt>.

  Out-of-range values throw \c std::out_of_range, and negative sizes in a
  \c Shape throw \c std::invalid_argument, before any communication starts.

  <tt>range(start, stop, step)</tt> creates a sequence as in python, e.g.
  <tt>range(-1, -5, -1)</tt> reverses the axes of a rank-4 tensor.
  \defgroup Matrix Matrix class
  \{
    \defgroup ScaLAPACK ScaLAPACK
    Parallelized matrix class using ScaLAPACK
    \defgroup LAPACK LAPACK
    Non-parallelized matrix class using LAPACK
  \}
  \defgroup Complex Complex numbers
  Value type of complex numbers
  \defgroup Internal Internal API
  \warning Not part of the public API. These names may change or be removed
  without notice, and user code should not use them.

  Helpers used inside the library. Public functions convert their arguments
  to the internal types (\c internal::UIndex, \c internal::UAxes,
  \c internal::UShape, all holding \c size_t) once at the entry point;
  everything after that works on \c size_t.
*/
