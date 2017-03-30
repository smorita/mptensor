/*!
  \file   matrix_lapack_impl.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>

  \brief  Implemation of mptensor::lapack::Matrix.

  Copyright (C) 2015 Satoshi Morita
*/

#ifndef _MATRIX_LAPACK_IMPL_HPP_
#define _MATRIX_LAPACK_IMPL_HPP_

#include <vector>
#include <algorithm>
#include <iostream>
#include <cassert>
#include "../complex.hpp"

namespace mptensor {

//! Namespace for the non-distributed Matrix class with LAPACK.
namespace lapack {

//! Non-distributed matrix.
/*!
  \note The matrix is stored in column-major order.
*/
template <typename C> class Matrix;

/* ---------- constructors ---------- */

//! Default constructor.
/*!
  The size of matrix is set to 0 times 0.
*/
template <typename C>
Matrix<C>::Matrix() {
  init(0, 0);
};

//! Constructor.
/*!
  \param comm_dummy Dummy argument. It is always ignored.
*/
template <typename C>
Matrix<C>::Matrix(const comm_type& comm_dummy) {
  init(0, 0);
};

//! Constructor.
/*!
  \param n_row The number of rows.
  \param n_col The number of columns.
*/
template <typename C>
Matrix<C>::Matrix(size_t n_row, size_t n_col) {
  init(n_row, n_col);
};

//! Constructor.
/*!
  \param comm_dummy Dummy argument. It is always ignored.
  \param n_row The number of rows.
  \param n_col The number of columns.
*/
template <typename C>
Matrix<C>::Matrix(const comm_type& comm_dummy, size_t n_row, size_t n_col) {
  init(n_row, n_col);
};

/* ---------- member functions ---------- */
template <typename C> inline
const C& Matrix<C>::operator[](size_t i) const { return V[i]; }

template <typename C> inline
C& Matrix<C>::operator[](size_t i) { return V[i]; }

template <typename C> inline
const C* Matrix<C>::head() const { return &(V[0]); }

template <typename C> inline
C* Matrix<C>::head() { return &(V[0]); }

template <typename C> inline
size_t Matrix<C>::local_size() const { return V.size(); }

//! Always returns 0.
template <typename C> inline
const typename Matrix<C>::comm_type& Matrix<C>::get_comm() const { return comm_; }

//! Always returns 1.
template <typename C> inline
int Matrix<C>::get_comm_size() const { return 1; }

//! Always returns 0.
template <typename C> inline
int Matrix<C>::get_comm_rank() const { return 0; }

template <typename C> inline
int Matrix<C>::n_row() const { return n_row_; }

template <typename C> inline
int Matrix<C>::n_col() const { return n_col_; }

template <typename C> inline
void Matrix<C>::global_index(size_t i, size_t &g_row, size_t &g_col) const {
  g_row = i % n_row_;
  g_col = i / n_row_;
};

template <typename C> inline
bool Matrix<C>::local_index(size_t g_row, size_t g_col, size_t &i) const {
  i = g_row + g_col * n_row_; // column major
  return true;
};

//! Convert a global index to an index of local storage.
/*!
  \param[in] g_row Global index of a row.
  \param[in] g_col Global index of a column.
  \param[out] comm_rank Always set to 0.
  \param[out] lindex Index of local storage.
*/
template <typename C> inline
void Matrix<C>::local_position(size_t g_row, size_t g_col, int &comm_rank, size_t &lindex) const {
  comm_rank = 0;
  lindex = g_row + g_col * n_row_;
}


template <typename C> inline
size_t Matrix<C>::local_row_size() const { return n_row_; }

template <typename C> inline
size_t Matrix<C>::local_col_size() const { return n_col_; }

template <typename C> inline
size_t Matrix<C>::local_row_index(size_t lindex) const { return lindex % n_row_; }

template <typename C> inline
size_t Matrix<C>::local_col_index(size_t lindex) const { return lindex / n_row_; }

template <typename C> inline
size_t Matrix<C>::global_row_index(size_t lindex_row) const { return lindex_row; }

template <typename C> inline
size_t Matrix<C>::global_col_index(size_t lindex_col) const { return lindex_col; }


template <typename C> inline
void Matrix<C>::init(size_t n_row, size_t n_col) {
  comm_ = 0;
  n_row_ = n_row;
  n_col_ = n_col;
  V.resize(n_row * n_col);
};

template <typename C> inline
void Matrix<C>::print_info(std::ostream& out) const {
  out << "Matrix: local_size= " << local_size()
      << "\n";
};

template <typename C> inline
Matrix<C>& Matrix<C>::operator+=(const Matrix<C> &rhs) {
  assert( V.size() == rhs.local_size() );
  for(size_t i=0;i<V.size();++i) V[i] += rhs[i];
  return *this;
};

template <typename C> inline
Matrix<C>& Matrix<C>::operator-=(const Matrix<C> &rhs) {
  assert( V.size() == rhs.local_size() );
  for(size_t i=0;i<V.size();++i) V[i] -= rhs[i];
  return *this;
};

template <typename C> inline
Matrix<C>& Matrix<C>::operator*=(C rhs) {
  for(size_t i=0;i<V.size();++i) V[i] *= rhs;
  return *this;
};

template <typename C> inline
Matrix<C>& Matrix<C>::operator/=(C rhs) {
  for(size_t i=0;i<V.size();++i) V[i] /= rhs;
  return *this;
};

template <typename C>
template <typename UnaryOperation>
Matrix<C>& Matrix<C>::map(UnaryOperation op) {
  std::transform(V.begin(), V.end(), V.begin(), op);
  return *this;
};

template <typename C>
const Matrix<C> Matrix<C>::transpose() {
  /* new matrix */
  Matrix<C> M_new(get_comm(), n_col(), n_row());

  size_t g_col, g_row, i_new;
  for(size_t i=0;i<local_size();i++) {
    global_index(i, g_row, g_col);
    M_new.local_index(g_col, g_row, i_new);

    M_new[i_new] = V[i];
  }

  return M_new;
}

template <typename C> inline
std::vector<C> Matrix<C>::flatten() {
  return V;
};

template <typename C> inline
void Matrix<C>::barrier() const { return; }

template <typename C> inline
C Matrix<C>::allreduce_sum(C value) const {
  return value;
}

//! Do nothing.
template <typename C> inline
void Matrix<C>::prep_local_to_global() const { return; }

//! Do nothing.
template <typename C> inline
void Matrix<C>::prep_global_to_local() const { return; }

/* ---------- non-member functions ---------- */

template <typename C>
void replace_matrix_data(const Matrix<C>& M,
                         const std::vector<int>& dest_rank,
                         const std::vector<size_t>& local_position,
                         Matrix<C>& M_new) {
  assert(dest_rank.size() == M.local_size());
  assert(local_position.size() == M.local_size());

  const C* mat = M.head();
  C *mat_new = M_new.head();

  for(size_t i=0;i<M.local_size();i++) {
    if(dest_rank[i] == 0) {
      mat_new[local_position[i]] = mat[i];
    }
  }
}

template <typename C>
void sum_matrix_data(const Matrix<C>& M,
                     const std::vector<int>& dest_rank,
                     const std::vector<size_t>& local_position,
                     Matrix<C>& M_new) {
  assert(dest_rank.size() == M.local_size());
  assert(local_position.size() == M.local_size());

  const C* mat = M.head();
  C *mat_new = M_new.head();

  for(size_t i=0;i<M.local_size();i++) {
    if(dest_rank[i] == 0) {
      mat_new[local_position[i]] += mat[i];
    }
  }
}


template <typename C>
C matrix_trace(const Matrix<C>& A) {
  const size_t n = A.local_size();
  size_t g_row, g_col;
  C val(0.0);

  for(size_t i=0;i<n;++i) {
    A.global_index(i, g_row, g_col);
    if(g_row==g_col) val += A[i];
  }

  return val;
};


template <typename C> double max_abs(const Matrix<C>& a) {
  const size_t n = a.local_size();
  double val = 0.0;
  for(size_t i=0;i<n;++i) {
    val = std::max(val, std::abs(a[i]));
  }
  return val;
};


template <typename C> double min_abs(const Matrix<C>& a) {
  const size_t n = a.local_size();
  if(n==0) return 0.0;
  double val = std::abs(a[0]);
  double recv;
  for(size_t i=0;i<n;++i) {
    val = std::min(val, std::abs(a[i]));
  }
  return val;
};


} // namespace lapack
} // namespace mptensor


#endif // _MATRIX_LAPACK_IMPL_HPP_
