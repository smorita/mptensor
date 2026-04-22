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
  \file   matrix_scalapack.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Dec 12 2014
  \brief  scalapack::Matrix class
*/

#ifndef _MATRIX_SCALAPACK_HPP_
#define _MATRIX_SCALAPACK_HPP_
#ifndef _NO_MPI

#include <iostream>
#include <vector>
#include <string>

#include <mpi.h>

#include "blacsgrid.hpp"

namespace mptensor {
namespace scalapack {

//! \ingroup ScaLAPACK
//! \{

template <typename C>
class Matrix {
 public:
  typedef C value_type;
  typedef MPI_Comm comm_type;

  constexpr static size_t matrix_type_tag = MATRIX_TYPE_TAG_SCALAPACK;
  constexpr static char* matrix_type_name = (char*)"ScaLAPACK";

  Matrix();
  explicit Matrix(const MPI_Comm& comm);
  Matrix(size_t n_row, size_t n_col);
  Matrix(const MPI_Comm& comm, size_t n_row, size_t n_col);

  void init(size_t n_row, size_t n_col);

  const MPI_Comm& get_comm() const;
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

  int n_row() const;
  int n_col() const;
  const int* descriptor() const;

  const Matrix transpose();
  void save_index(const std::string &filename) const;

 private:
  BlacsGrid grid;
  std::vector<C> V;       // local strage
  std::vector<int> desc;  // descriptor
  static const size_t BLOCK_SIZE;
  size_t local_row_size_;
  size_t local_col_size_;

  int get_lld() const;
  // void set(C (*element)(size_t g_row, size_t g_col));
  // const BlacsGrid &get_grid() const;
  // int blacs_context() const;
  // int nb_row() const;
  // int nb_col() const;

  mutable bool has_local_to_global;  //!< Flag for local-to-global mapping.
  mutable bool has_global_to_local;  //!< Flag for global-to-local mapping.
  mutable std::vector<size_t>
      global_row;  //!< Mapping data from local row to global row.
  mutable std::vector<size_t>
      global_col;  //!< Mapping data from local column to global column.
  mutable std::vector<size_t>
      local_row;  //!< Mapping data from global row to local row.
  mutable std::vector<size_t>
      local_col;  //!< Mapping data from global column to local column.
  mutable std::vector<int>
      proc_row;  //!< Mapping data from global row to processor row.
  mutable std::vector<int>
      proc_col;  //!< Mapping data from global column to processor column.
  mutable std::vector<int> lld_list;  //!< Mapping data from processor row to
                                      //!< local leading dimension.
};

//! \name Matrix operations
//! \{
template <typename C>
void replace_matrix_data(const Matrix<C>& M, const std::vector<int>& dest_rank,
                         const std::vector<size_t>& local_position,
                         Matrix<C>& M_new);
template <typename C>
void replace_matrix_data(const std::vector<C>& V, const std::vector<int>& dest_rank,
                         const std::vector<size_t>& local_position,
                         Matrix<C>& M_new);
template <typename C>
void sum_matrix_data(const Matrix<C>& M, const std::vector<int>& dest_rank,
                     const std::vector<size_t>& local_position,
                     Matrix<C>& M_new);

template <typename C>
double max_abs(const Matrix<C>& a);
template <typename C>
double min_abs(const Matrix<C>& a);
template <typename C>
C matrix_trace(const Matrix<C>& a);

// defined in matrix_scalapack.cc
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
int matrix_eigh(Matrix<C>& a, Matrix<C>& b, std::vector<double>& s,
                Matrix<C>& u);
// template <typename C>
// int matrix_eig(Matrix<C>& a, std::vector<complex>& s, Matrix<complex>& u);
// template <typename C>
// int matrix_eig(Matrix<C>& a, std::vector<complex>& s);
template <typename C>
int matrix_solve(Matrix<C>& a, Matrix<C>& b);
template <typename C>
double max(const Matrix<C>& a);
template <typename C>
double min(const Matrix<C>& a);
//! \}

//! \}

}  // namespace scalapack
}  // namespace mptensor

#include "matrix_scalapack_impl.hpp"

#endif  // _NO_MPI
#endif  // _MATRIX_SCALAPACK_HPP_
