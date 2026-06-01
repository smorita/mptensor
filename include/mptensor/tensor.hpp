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
  \file   tensor.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jan 14 2015

  \brief  Tensor class
*/

#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "mptensor/complex.hpp"
#include "mptensor/index.hpp"
#include "mptensor/matrix/matrix.hpp"

namespace mptensor {

/* Alias */
using Axes  = Index;
using Shape = Index;

/* Class definition */
//! Tensor class. The main object of mptensor.
/*!
  \ingroup Tensor
*/
template <typename MatrixType>
class Tensor {
 public:
  using value_type  = typename MatrixType::value_type;  //!< \c double or \c complex
  using matrix_type = MatrixType;                       //!< type of Matrix class
  using comm_type   = typename MatrixType::comm_type;   //!< type of communicator. \c MPI_Comm or \c int.

  template <typename D>
  using rebind = Tensor<typename MatrixType::template rebind<D>>;  //!< Tensor with scalar type replaced by \c D.

  //! \ingroup TensorConstructor
  //! \{
  Tensor();
  explicit Tensor(const Shape &);
  explicit Tensor(const comm_type &);
  Tensor(const comm_type &, const Shape &);
  Tensor(const comm_type &, const Shape &, size_t upper_rank);
  Tensor(const comm_type &, const Tensor<lapack::Matrix<value_type>> &);
  Tensor(const comm_type &, const std::vector<value_type> &);
  //! \}

  const Shape &shape() const;
  size_t rank() const;
  size_t ndim() const;
  size_t local_size() const;
  size_t get_upper_rank() const;
  const Axes &get_axes_map() const;

  const MatrixType &get_matrix() const;
  MatrixType &get_matrix();

  const comm_type &get_comm() const;
  int get_comm_size() const;
  int get_comm_rank() const;

  Index global_index(size_t i) const;
  void global_index_fast(size_t i, Index &idx) const;
  void local_position(const Index &idx, int &comm_rank,
                      size_t &local_idx) const;

  const value_type &operator[](size_t local_idx) const;
  value_type &operator[](size_t local_idx);

  bool get_value(const Index &idx, value_type &val) const;
  void set_value(const Index &idx, value_type val);

  void print_info(std::ostream &out, const std::string &tag = "") const;
  void print_info_mpi(std::ostream &, const std::string &tag = "") const;

  void save(const std::string &filename) const;
  void load(const std::string &filename);

  Tensor<MatrixType> &transpose(const Axes &axes);

  template <typename D>
  Tensor<MatrixType> &multiply_vector(const std::vector<D> &vec, size_t n_axes);
  template <typename D0, typename D1>
  Tensor<MatrixType> &multiply_vector(const std::vector<D0> &vec0,
                                      size_t n_axes0,
                                      const std::vector<D1> &vec1,
                                      size_t n_axes1);
  template <typename D0, typename D1, typename D2>
  Tensor<MatrixType> &multiply_vector(
      const std::vector<D0> &vec0, size_t n_axes0, const std::vector<D1> &vec1,
      size_t n_axes1, const std::vector<D2> &vec2, size_t n_axes2);
  template <typename D0, typename D1, typename D2, typename D3>
  Tensor<MatrixType> &multiply_vector(
      const std::vector<D0> &vec0, size_t n_axes0, const std::vector<D1> &vec1,
      size_t n_axes1, const std::vector<D2> &vec2, size_t n_axes2,
      const std::vector<D3> &vec3, size_t n_axes3);

  Tensor<MatrixType> &set_slice(const Tensor &a, size_t n_axes, size_t i_begin,
                                size_t i_end);
  Tensor<MatrixType> &set_slice(const Tensor &a, const Index &index_begin,
                                const Index &index_end);

  Tensor<lapack::Matrix<value_type>> gather();
  std::vector<value_type> flatten();

  Tensor<MatrixType> &operator+=(const Tensor &rhs);
  Tensor<MatrixType> &operator-=(const Tensor &rhs);
  Tensor<MatrixType> &operator*=(value_type rhs);
  Tensor<MatrixType> &operator/=(value_type rhs);
  Tensor<MatrixType> &operator=(value_type rhs);

  template <typename UnaryOperation>
  Tensor<MatrixType> &map(UnaryOperation op);

  void prep_global_to_local() const;
  void prep_local_to_global() const;

  void make_l2g_map() const;
  void global_index_l2g_map(size_t lindex, size_t gindex[]) const;
  void global_index_l2g_map_transpose(size_t lindex, const size_t axes_trans[],
                                      size_t index_new[]) const;

  void local_position_fast(size_t g_row, size_t g_col, int &comm_rank,
                           size_t &local_idx) const;

 private:
  MatrixType Mat;  //!< local storage.
  Shape Dim;       //!< Shape of tensor.

  size_t upper_rank;  //!< Upper rank for matrix representation.

  //! Map of axes for lazy evaluation of transpose.
  /*!
    This is the inverse permutation of axes given in transpose(), i.e.
    axes_map[axes[i]]=i. The i-th index of the orignal tensor is moved to the
    (axes_map[i])-th index of the transposed tensor.
  */
  Axes axes_map;

  void init(const Shape &, size_t upper_rank);
  void init(const Shape &, size_t upper_rank, const Axes &map);
  void change_configuration(const size_t new_upper_rank,
                            const Axes &new_axes_map);
  bool local_index(const Index &, size_t &i) const;

  mutable std::vector<size_t> l2g_map_row;
  mutable std::vector<size_t> l2g_map_col;

  void save_ver_0_2(const char *filename) const;
  void load_ver_0_2(const char *filename);
};

/* Operations */
//! \ingroup ShapeChange
//! \{
template <typename MatrixType>
Tensor<MatrixType> transpose(Tensor<MatrixType> a, const Axes &axes);
template <typename MatrixType>
Tensor<MatrixType> transpose(const Tensor<MatrixType> &a, const Axes &axes,
                             size_t urank_new);
template <typename MatrixType>
Tensor<MatrixType> reshape(const Tensor<MatrixType> &a, const Shape &shape_new);
template <typename MatrixType>
Tensor<MatrixType> slice(const Tensor<MatrixType> &a, size_t n_axes,
                         size_t i_begin, size_t i_end);
template <typename MatrixType>
Tensor<MatrixType> slice(const Tensor<MatrixType> &a, const Index &index_begin,
                         const Index &index_end);
template <typename MatrixType>
Tensor<MatrixType> extend(const Tensor<MatrixType> &a, const Shape &shape_new);
//! \}

//! \ingroup LinearAlgebra
//! \{
template <typename MatrixType>
typename MatrixType::value_type trace(const Tensor<MatrixType> &a);
template <typename MatrixType>
typename MatrixType::value_type trace(const Tensor<MatrixType> &a, const Axes &axes_1, const Axes &axes_2);
template <typename MatrixType>
typename MatrixType::value_type trace(const Tensor<MatrixType> &a, const Tensor<MatrixType> &b,
                                      const Axes &axes_a, const Axes &axes_b);
template <typename MatrixType>
Tensor<MatrixType> contract(const Tensor<MatrixType> &a, const Axes &axes_1,
                            const Axes &axes_2);
template <typename MatrixType>
Tensor<MatrixType> tensordot(const Tensor<MatrixType> &a,
                             const Tensor<MatrixType> &b, const Axes &axes_a,
                             const Axes &axes_b);
template <typename MatrixType>
Tensor<MatrixType> kron(const Tensor<MatrixType> &a,
                        const Tensor<MatrixType> &b);
//! \}

//! \ingroup Decomposition
//! \{
template <typename MatrixType>
int svd(const Tensor<MatrixType> &a, std::vector<double> &s);
template <typename MatrixType>
int svd(const Tensor<MatrixType> &a, Tensor<MatrixType> &u,
        std::vector<double> &s, Tensor<MatrixType> &vt);
template <typename MatrixType>
int svd(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
        std::vector<double> &s);
template <typename MatrixType>
int svd(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
        Tensor<MatrixType> &u, std::vector<double> &s, Tensor<MatrixType> &vt);

template <typename MatrixType>
int psvd(const Tensor<MatrixType> &a, std::vector<double> &s,
         const size_t target_rank);
template <typename MatrixType>
int psvd(const Tensor<MatrixType> &a, Tensor<MatrixType> &u,
         std::vector<double> &s, Tensor<MatrixType> &vt,
         const size_t target_rank);
template <typename MatrixType>
int psvd(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
         std::vector<double> &s, const size_t target_rank);
template <typename MatrixType>
int psvd(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
         Tensor<MatrixType> &u, std::vector<double> &s, Tensor<MatrixType> &vt,
         const size_t target_rank);

template <typename MatrixType>
int qr(const Tensor<MatrixType> &a, Tensor<MatrixType> &q, Tensor<MatrixType> &r);
template <typename MatrixType>
int qr(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
       Tensor<MatrixType> &q, Tensor<MatrixType> &r);

template <typename MatrixType>
int eigh(const Tensor<MatrixType> &a, std::vector<double> &eigval,
         Tensor<MatrixType> &eigvec);
template <typename MatrixType>
int eigh(const Tensor<MatrixType> &a, std::vector<double> &eigval);
template <typename MatrixType>
int eigh(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
         std::vector<double> &eigval, Tensor<MatrixType> &eigvec);
template <typename MatrixType>
int eigh(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
         std::vector<double> &eigval);
template <typename MatrixType>
int eigh(const Tensor<MatrixType> &a, const Axes &axes_row_a,
         const Axes &axes_col_a, const Tensor<MatrixType> &b,
         const Axes &axes_row_b, const Axes &axes_col_b,
         std::vector<double> &eigval, Tensor<MatrixType> &eigvec);


template <typename MatrixType>
int eig(const Tensor<MatrixType> &a, std::vector<complex> &eigval,
        typename Tensor<MatrixType>::template rebind<complex> &eigvec);
template <typename MatrixType>
int eig(const Tensor<MatrixType> &a, std::vector<complex> &eigval);
template <typename MatrixType>
int eig(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
        std::vector<complex> &eigval,
        typename Tensor<MatrixType>::template rebind<complex> &eigvec);
template <typename MatrixType>
int eig(const Tensor<MatrixType> &a, const Axes &axes_row, const Axes &axes_col,
        std::vector<complex> &eigval);
//! \}

//! \ingroup LinearEq
//! \{
template <typename MatrixType>
int solve(const Tensor<MatrixType> &a,
          const std::vector<typename MatrixType::value_type> &b,
          std::vector<typename MatrixType::value_type> &x);
template <typename MatrixType>
int solve(const Tensor<MatrixType> &a, const Tensor<MatrixType> &b,
          Tensor<MatrixType> &x);
template <typename MatrixType>
int solve(const Tensor<MatrixType> &a, const Tensor<MatrixType> &b,
          Tensor<MatrixType> &x, const Axes &axes_row_a, const Axes &axes_col_a,
          const Axes &axes_row_b, const Axes &axes_col_b);
//! \}

//! \ingroup Arithmetic
//! \{
template <typename MatrixType>
Tensor<MatrixType> operator+(Tensor<MatrixType> rhs);
template <typename MatrixType>
Tensor<MatrixType> operator-(Tensor<MatrixType> rhs);
template <typename MatrixType>
Tensor<MatrixType> operator+(Tensor<MatrixType> lhs,
                             const Tensor<MatrixType> &rhs);
template <typename MatrixType>
Tensor<MatrixType> operator-(Tensor<MatrixType> lhs,
                             const Tensor<MatrixType> &rhs);
template <typename MatrixType, typename D>
Tensor<MatrixType> operator*(Tensor<MatrixType> lhs, D rhs);
template <typename MatrixType, typename D>
Tensor<MatrixType> operator/(Tensor<MatrixType> lhs, D rhs);
template <typename MatrixType, typename D>
Tensor<MatrixType> operator*(D lhs, Tensor<MatrixType> rhs);
//! \}

//! \ingroup Misc
//! \{
template <typename MatrixType>
Tensor<MatrixType> sqrt(
    Tensor<MatrixType> t);  //!< Take square-root of each element.
template <typename MatrixType>
Tensor<MatrixType> conj(
    Tensor<MatrixType> t);  //!< Take conjugate of each element.

template <typename MatrixType>
double max(const Tensor<MatrixType> &t);
template <typename MatrixType>
double min(const Tensor<MatrixType> &t);
template <typename MatrixType>
double max_abs(const Tensor<MatrixType> &t);
template <typename MatrixType>
double min_abs(const Tensor<MatrixType> &t);
//! \}

//! \ingroup Output
//! \{
template <typename MatrixType>
std::ostream &operator<<(std::ostream &out, const Tensor<MatrixType> &t);
//! \}

}  // namespace mptensor

#include "mptensor/tensor_impl.hpp"

#endif  // _TENSOR_HPP_
