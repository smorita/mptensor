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

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <string>

#include "complex.hpp"
#include "index.hpp"
#include "matrix.hpp"

namespace mptensor {

/* Alias */
typedef Index Axes;
typedef Index Shape;

/* Class definition */
template <template<typename> class Matrix, typename C> class Tensor {
public:
  typedef C value_type; //!< \c double or \c complex
  typedef Matrix<C> matrix_type; //!< type of Matrix class
  typedef typename Matrix<C>::comm_type comm_type; //!< type of communicator.
                                                   /*!< \c MPI_Comm or \c int. */

  Tensor();
  explicit Tensor(const Shape&);
  explicit Tensor(const comm_type&);
  Tensor(const comm_type&, const Shape&);
  Tensor(const comm_type&, const Shape&, size_t upper_rank);
  Tensor(const comm_type&, const Tensor<lapack::Matrix,C>&);
  Tensor(const comm_type&, const std::vector<C>&);

  const Shape& shape() const;
  size_t rank() const;
  size_t ndim() const;
  size_t local_size() const;
  size_t get_upper_rank() const;
  const Axes& get_axes_map() const;

  const Matrix<C>& get_matrix() const;
  Matrix<C>& get_matrix();

  const comm_type &get_comm() const;
  int get_comm_size() const;
  int get_comm_rank() const;

  Index global_index(size_t i) const;
  void global_index_fast(size_t i, Index& idx) const;
  void local_position(const Index& idx, int& comm_rank, size_t& local_idx) const;

  const C &operator[](size_t local_idx) const;
  C &operator[](size_t local_idx);

  bool get_value(const Index& idx, C &val) const;
  void set_value(const Index& idx, C val);

  void print_info(std::ostream& out, const std::string& tag="") const;
  void print_info_mpi(std::ostream&, const std::string& tag="") const;

  void save(const char* filename) const;
  void load(const char* filename);

  Tensor<Matrix,C>& transpose(const Axes &axes);

  template <typename D>
  Tensor<Matrix,C>& multiply_vector(const std::vector<D> &vec, size_t n_axes);
  template <typename D0, typename D1>
  Tensor<Matrix,C>& multiply_vector(const std::vector<D0> &vec0, size_t n_axes0,
                                    const std::vector<D1> &vec1, size_t n_axes1);
  template <typename D0, typename D1, typename D2>
  Tensor<Matrix,C>& multiply_vector(const std::vector<D0> &vec0, size_t n_axes0,
                                    const std::vector<D1> &vec1, size_t n_axes1,
                                    const std::vector<D2> &vec2, size_t n_axes2);
  template <typename D0, typename D1, typename D2, typename D3>
  Tensor<Matrix,C>& multiply_vector(const std::vector<D0> &vec0, size_t n_axes0,
                                    const std::vector<D1> &vec1, size_t n_axes1,
                                    const std::vector<D2> &vec2, size_t n_axes2,
                                    const std::vector<D3> &vec3, size_t n_axes3);

  Tensor<Matrix,C>& set_slice(const Tensor &a, size_t n_axes, size_t i_begin, size_t i_end);
  Tensor<Matrix,C>& set_slice(const Tensor &a, const Index &index_begin, const Index &index_end);

  std::vector<C> flatten();

  Tensor<Matrix,C>& operator+=(const Tensor &rhs);
  Tensor<Matrix,C>& operator-=(const Tensor &rhs);
  Tensor<Matrix,C>& operator*=(C rhs);
  Tensor<Matrix,C>& operator/=(C rhs);
  Tensor<Matrix,C>& operator=(C rhs);

  template <typename UnaryOperation>
  Tensor<Matrix,C>& map(UnaryOperation op);

  void prep_global_to_local() const;
  void prep_local_to_global() const;

  void make_l2g_map() const;
  void global_index_l2g_map(size_t lindex, size_t gindex[]) const;
  void global_index_l2g_map_transpose(size_t lindex, const size_t axes_trans[], size_t index_new[]) const;

  void local_position_fast(size_t g_row, size_t g_col, int& comm_rank, size_t& local_idx) const;

private:
  Matrix<C> Mat; //!< local storage.
  Shape Dim; //!< Shape of tensor.

  size_t upper_rank; //!< Upper rank for matrix representation.

  //! Map of axes for lazy evaluation of transpose.
  /*!
    This is the inverse permutation of axes given in transpose(), i.e. axes_map[axes[i]]=i.
    The i-th index of the orignal tensor is moved to the (axes_map[i])-th index of the transposed tensor.
  */
  Axes axes_map;

  void init(const Shape&, size_t upper_rank);
  void init(const Shape&, size_t upper_rank, const Axes& map);
  void change_configuration(const size_t new_upper_rank, const Axes &new_axes_map);
  bool local_index(const Index&, size_t &i) const;

  mutable std::vector<size_t> l2g_map_row;
  mutable std::vector<size_t> l2g_map_col;
};


/* Operations */
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> transpose(Tensor<Matrix,C> a, const Axes& axes);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> transpose(const Tensor<Matrix,C> &a, const Axes& axes, size_t urank_new);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> reshape(const Tensor<Matrix,C> &a, const Shape& shape_new);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> slice(const Tensor<Matrix,C> &a, size_t n_axes, size_t i_begin, size_t i_end);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> slice(const Tensor<Matrix,C> &a, const Index &index_begin, const Index &index_end);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> extend(const Tensor<Matrix,C> &a, const Shape& shape_new);

template <template<typename> class Matrix, typename C> C trace(const Tensor<Matrix,C> &a);
template <template<typename> class Matrix, typename C> C trace(const Tensor<Matrix,C> &a, const Axes &axes_1, const Axes &axes_2);
template <template<typename> class Matrix, typename C> C trace(const Tensor<Matrix,C> &a, const Tensor<Matrix,C> &b, const Axes &axes_a, const Axes &axes_b);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> contract(const Tensor<Matrix,C> &a, const Axes &axes_1, const Axes &axes_2);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> tensordot(const Tensor<Matrix,C> &a, const Tensor<Matrix,C> &b, const Axes& axes_a, const Axes& axes_b);

template <template<typename> class Matrix, typename C> int svd(const Tensor<Matrix,C> &a, std::vector<double> &s);
template <template<typename> class Matrix, typename C> int svd(const Tensor<Matrix,C> &a, Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt);
template <template<typename> class Matrix, typename C> int svd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col, std::vector<double> &s);
template <template<typename> class Matrix, typename C> int svd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col, Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt);

template <template<typename> class Matrix, typename C> int psvd(const Tensor<Matrix,C> &a, std::vector<double> &s, const size_t target_rank);
template <template<typename> class Matrix, typename C> int psvd(const Tensor<Matrix,C> &a, Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt, const size_t target_rank);
template <template<typename> class Matrix, typename C> int psvd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col, std::vector<double> &s, const size_t target_rank);
template <template<typename> class Matrix, typename C> int psvd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col, Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt, const size_t target_rank);

template <template<typename> class Matrix, typename C> int qr(const Tensor<Matrix,C> &a, Tensor<Matrix,C> &q, Tensor<Matrix,C> &r);
template <template<typename> class Matrix, typename C> int qr(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col, Tensor<Matrix,C> &q, Tensor<Matrix,C> &r);

template <template<typename> class Matrix, typename C> int eigh(const Tensor<Matrix,C> &a, std::vector<double> &eigval, Tensor<Matrix,C> &eigvec);
template <template<typename> class Matrix, typename C> int eigh(const Tensor<Matrix,C> &a, std::vector<double> &eigval);
template <template<typename> class Matrix, typename C> int eigh(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col, std::vector<double> &eigval, Tensor<Matrix,C> &eigvec);

template <template<typename> class Matrix, typename C> int solve(const Tensor<Matrix,C> &a, const std::vector<C> &b, std::vector<C> &x);
template <template<typename> class Matrix, typename C> int solve(const Tensor<Matrix,C> &a, const Tensor<Matrix,C> &b, Tensor<Matrix,C> &x);
template <template<typename> class Matrix, typename C> int solve(const Tensor<Matrix,C> &a, const Tensor<Matrix,C> &b, Tensor<Matrix,C> &x, const Axes &axes_row_a, const Axes &axes_col_a, const Axes &axes_row_b, const Axes &axes_col_b);

template <template<typename> class Matrix, typename C> Tensor<Matrix,C> operator+(Tensor<Matrix,C> rhs);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> operator-(Tensor<Matrix,C> rhs);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> operator+(Tensor<Matrix,C> lhs, const Tensor<Matrix,C> &rhs);
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> operator-(Tensor<Matrix,C> lhs, const Tensor<Matrix,C> &rhs);
template <template<typename> class Matrix, typename C, typename D> Tensor<Matrix,C> operator*(Tensor<Matrix,C> lhs, D rhs);
template <template<typename> class Matrix, typename C, typename D> Tensor<Matrix,C> operator/(Tensor<Matrix,C> lhs, D rhs);
template <template<typename> class Matrix, typename C, typename D> Tensor<Matrix,C> operator*(D lhs, Tensor<Matrix,C> rhs);

template <template<typename> class Matrix, typename C> Tensor<Matrix,C> sqrt(Tensor<Matrix,C> t); //!< Take square-root of each element.
template <template<typename> class Matrix, typename C> Tensor<Matrix,C> conj(Tensor<Matrix,C> t); //!< Take conjugate of each element.

template <template<typename> class Matrix, typename C> double max(const Tensor<Matrix,C> &t);
template <template<typename> class Matrix, typename C> double min(const Tensor<Matrix,C> &t);
template <template<typename> class Matrix, typename C> double max_abs(const Tensor<Matrix,C> &t);
template <template<typename> class Matrix, typename C> double min_abs(const Tensor<Matrix,C> &t);

template <template<typename> class Matrix, typename C> std::ostream& operator<<(std::ostream& out, const Tensor<Matrix,C> &t);

} // namespace mptensor

#include "tensor_impl.hpp"

#endif // _TENSOR_HPP_
