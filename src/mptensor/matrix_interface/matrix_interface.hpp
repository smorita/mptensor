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
  \file   matrix_interface.hpp
  \author Satoshi Morita

  \brief  Interface of Matrix class.
*/

#ifndef _MATRIX_INTERFACE_HPP_
#define _MATRIX_INTERFACE_HPP_

#include <iostream>
#include <vector>

namespace mptensor {
namespace matrix_interface {

template <typename C>
class Matrix {
 public:
  typedef C value_type;   //!< \c double or \c complex
  typedef int comm_type;  //!< Set \c MPI_Comm for MPI programm.

  Matrix();
  explicit Matrix(const comm_type& comm);
  Matrix(size_t n_row, size_t n_col);
  Matrix(const comm_type& comm, size_t n_row, size_t n_col);

  void init(size_t n_row, size_t n_col);

  const comm_type& get_comm() const;
  int get_comm_size() const;
  int get_comm_rank() const;

  void print_info(std::ostream&) const;

  const C& operator[](size_t i) const;
  C& operator[](size_t i);
  const C* head() const;
  C* head();

  size_t local_size() const;
  void global_index(size_t i, size_t& g_row, size_t& g_col) const;
  bool local_index(size_t g_row, size_t g_col, size_t& i) const;
  void local_position(size_t g_row, size_t g_col, int& comm_rank,
                      size_t& lindex) const;

  size_t local_row_size() const;
  size_t local_col_size() const;
  size_t local_row_index(size_t lindex) const;
  size_t local_col_index(size_t lindex) const;
  size_t global_row_index(size_t lindex_row) const;
  size_t global_col_index(size_t lindex_col) const;

  Matrix& operator+=(const Matrix& rhs);
  Matrix& operator-=(const Matrix& rhs);
  Matrix& operator*=(C rhs);
  Matrix& operator/=(C rhs);

  template <typename UnaryOperation>
  Matrix& map(UnaryOperation op);

  std::vector<C> flatten();

  void barrier() const;
  C allreduce_sum(C value) const;
  template <typename D>
  void bcast(D* buffer, int count, int root) const;

  void prep_local_to_global() const;
  void prep_global_to_local() const;
};

template <typename C>
void replace_matrix_data(const Matrix<C>& M, const std::vector<int>& dest_rank,
                         const std::vector<size_t>& local_position,
                         Matrix<C>& M_new);
template <typename C>
void sum_matrix_data(const Matrix<C>& M, const std::vector<int>& dest_rank,
                     const std::vector<size_t>& local_position,
                     Matrix<C>& M_new);
template <typename C>
C matrix_trace(const Matrix<C>& a);
template <typename C>
void matrix_product(const Matrix<C>& a, const Matrix<C>& b, Matrix<C>& c);
template <typename C>
int matrix_svd(Matrix<C>& a, Matrix<C>& u, std::vector<double>& s,
               Matrix<C>& v);
template <typename C>
int matrix_svd(Matrix<C>& a, std::vector<double>& s);
template <typename C>
int matrix_qr(Matrix<C>& a, Matrix<C>& r);
template <typename C>
int matrix_eigh(Matrix<C>& a, std::vector<double>& s, Matrix<C>& u);
template <typename C>
int matrix_eigh(Matrix<C>& a, std::vector<double>& s);
template <typename C>
int matrix_solve(Matrix<C>& a, Matrix<C>& b);

template <typename C>
double max(const Matrix<C>& a);
template <typename C>
double min(const Matrix<C>& a);
template <typename C>
double max_abs(const Matrix<C>& a);
template <typename C>
double min_abs(const Matrix<C>& a);

}  // namespace matrix_interface
}  // namespace mptensor

#include "matrix_interface_doc.hpp"

#endif  // _MATRIX_INTERFACE_HPP_
