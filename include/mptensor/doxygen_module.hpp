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
  \defgroup Index Index class
  \defgroup Matrix Matrix class
  \{
    \defgroup ScaLAPACK ScaLAPACK
    Parallelized matrix class using ScaLAPACK
    \defgroup LAPACK LAPACK
    Non-parallelized matrix class using LAPACK
  \}
  \defgroup Complex Complex numbers
  Value type of complex numbers
*/
