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
  \file   tensor_impl.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jan 14 2015

  \brief  Implementation of tensor class
*/

#ifndef _TENSOR_IMPL_HPP_
#define _TENSOR_IMPL_HPP_

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "complex.hpp"
#include "index.hpp"
#include "matrix.hpp"
#include "tensor.hpp"

namespace mptensor {

/* Utilities */
bool is_no_transpose(const Axes &axes, const Axes &axes_map, size_t rank);
namespace debug {
bool check_total_size(const Shape &s1, const Shape &s2);
bool check_extend(const Shape &s_old, const Shape &s_new);
bool check_transpose_axes(const Axes &axes, size_t rank);
bool check_svd_axes(const Axes &axes_row, const Axes &axes_col, size_t rank);
bool check_trace_axes(const Axes &axes_1, const Axes &axes_2, size_t rank);
bool check_trace_axes(const Axes &axes_a, const Axes &axes_b,
                      const Shape &shape_a, const Shape &shape_b);
bool check_contract_axes(const Axes &axes_1, const Axes &axes_2, size_t rank);
bool check_square(const Shape &shape, size_t urank);
}  // namespace debug

/* ---------- constructors ---------- */

//! Default constructor of tensor.
/*!
  \note Communicator is set to MPI_COMM_WORLD or MPI_COMM_SELF depending on
  Matrix class.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor() : Mat(), upper_rank(0), axes_map(){};

//! Constructor of tensor.
/*!
  \param[in] shape Shape of tensor.
  \note Communicator is set to MPI_COMM_WORLD or MPI_COMM_SELF depending on
  Matrix class. \note Upper rank for matrix representation is set to \c rank/2.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor(const Shape &shape) : Mat() {
  init(shape, shape.size() / 2);
};

//! Constructor of tensor.
/*!
  \param[in] comm Communicator.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor(const comm_type &comm)
    : Mat(comm), upper_rank(0), axes_map(){};

//! Constructor of tensor.
/*!
  \param[in] comm Communicator.
  \param[in] shape Shape of tensor.
  \note Upper rank for matrix representation is set to \c rank/2.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor(const comm_type &comm, const Shape &shape)
    : Mat(comm) {
  init(shape, shape.size() / 2);
};

//! Constructor of tensor.
/*!
  \param[in] comm Communicator.
  \param[in] shape Shape of tensor.
  \param[in] upper_rank Upper rank for matrix representation.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor(const comm_type &comm, const Shape &shape,
                          const size_t upper_rank)
    : Mat(comm) {
  init(shape, upper_rank);
};

//! Constructor of tensor from non-distributed tensor.
/*!
  \param[in] comm Communicator.
  \param[in] t Non-distributed tensor.
  \attention It is assumed that all processes have the same data.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor(const comm_type &comm,
                          const Tensor<lapack::Matrix, C> &t)
    : Mat(comm) {
  init(t.shape(), t.get_upper_rank());
  const size_t n = Mat.local_size();
  size_t idx;
  int dummy;
  for (size_t i = 0; i < n; ++i) {
    t.local_position(global_index(i), dummy, idx);
    Mat[i] = t[idx];
  }
};

//! Constructor of tensor from non-distributed vector.
/*!
  \param[in] comm Communicator.
  \param[in] v Non-distributed vector.
  \attention It is assumed that all processes have the same data.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C>::Tensor(const comm_type &comm, const std::vector<C> &v)
    : Mat(comm) {
  init(Shape(v.size()), 0);
  const size_t n = Mat.local_size();
  Index idx;
  for (size_t i = 0; i < n; ++i) {
    idx = global_index(i);
    Mat[i] = v[idx[0]];
  }
};

/* ---------- inline member functions ---------- */

//! Shape of tensor.
template <template <typename> class Matrix, typename C>
inline const Shape &Tensor<Matrix, C>::shape() const {
  return Dim;
};

//! Rank of tensor.
template <template <typename> class Matrix, typename C>
inline size_t Tensor<Matrix, C>::rank() const {
  return Dim.size();
};

//! Rank of tensor.
/*!
  \note Same as rank(). This function is for numpy compatibility with numpy.
 */
template <template <typename> class Matrix, typename C>
inline size_t Tensor<Matrix, C>::ndim() const {
  return Dim.size();
};

//! Size of local storage.
template <template <typename> class Matrix, typename C>
inline size_t Tensor<Matrix, C>::local_size() const {
  return Mat.local_size();
};

//! Upper rank for matrix representation
template <template <typename> class Matrix, typename C>
inline size_t Tensor<Matrix, C>::get_upper_rank() const {
  return upper_rank;
};

//! Map of axes for lazy evaluation of transpose
template <template <typename> class Matrix, typename C>
inline const Axes &Tensor<Matrix, C>::get_axes_map() const {
  return axes_map;
};

//! distributed Matrix
template <template <typename> class Matrix, typename C>
inline const Matrix<C> &Tensor<Matrix, C>::get_matrix() const {
  return Mat;
};

//! distributed Matrix
template <template <typename> class Matrix, typename C>
inline Matrix<C> &Tensor<Matrix, C>::get_matrix() {
  return Mat;
};

//! Communicator
template <template <typename> class Matrix, typename C>
inline const typename Tensor<Matrix, C>::comm_type &
Tensor<Matrix, C>::get_comm() const {
  return Mat.get_comm();
};

//! Size of communicator
template <template <typename> class Matrix, typename C>
inline int Tensor<Matrix, C>::get_comm_size() const {
  return Mat.get_comm_size();
};

//! Rank of process
template <template <typename> class Matrix, typename C>
inline int Tensor<Matrix, C>::get_comm_rank() const {
  return Mat.get_comm_rank();
};

//! Const array subscript operator
/*!
  \attention This function does not check validity of local index.
 */
template <template <typename> class Matrix, typename C>
inline const C &Tensor<Matrix, C>::operator[](size_t local_idx) const {
  return Mat[local_idx];
}

//! Array subscript operator
/*!
  \attention This function does not check validity of local index.
 */
template <template <typename> class Matrix, typename C>
inline C &Tensor<Matrix, C>::operator[](size_t local_idx) {
  return Mat[local_idx];
}

//! Preprocess for fast conversion from global index to local one.
template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::prep_global_to_local() const {
  Mat.prep_global_to_local();
};

//! Preprocess for fast conversion from local index to global one.
template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::prep_local_to_global() const {
  Mat.prep_local_to_global();
};

//! Preprocess for fast conversion from local index to global one.
template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::make_l2g_map() const {
  const size_t rank = Dim.size();
  const size_t rank0 = upper_rank;
  const size_t rank1 = rank - rank0;  // lower_rank
  const size_t l_row = Mat.local_row_size();
  const size_t l_col = Mat.local_col_size();
  const size_t *axes_map0 = &(axes_map[0]);
  const size_t *axes_map1 = &(axes_map[rank0]);

  l2g_map_row.resize(l_row * rank0);
  l2g_map_col.resize(l_col * rank1);

#pragma omp parallel default(shared)
  {
    // Type int instead of size_t is used because of std::div()
    int dim0[rank0];
    int dim1[rank1];
    for (size_t i = 0; i < rank0; ++i) dim0[i] = int(Dim[axes_map0[i]]);
    for (size_t i = 0; i < rank1; ++i) dim1[i] = int(Dim[axes_map1[i]]);

    int g_row, g_col;  // global matrix row- and column-index
    std::div_t divresult;
    size_t *map;
#pragma omp for
    for (size_t k = 0; k < l_row; ++k) {
      map = &(l2g_map_row[k * rank0]);
      g_row = int(Mat.global_row_index(k));
      for (size_t i = 0; i < rank0; ++i) {
        divresult = std::div(g_row, dim0[i]);
        map[i] = divresult.rem;
        g_row = divresult.quot;
      }
    }
#pragma omp for
    for (size_t k = 0; k < l_col; ++k) {
      map = &(l2g_map_col[k * rank1]);
      g_col = int(Mat.global_col_index(k));
      for (size_t i = 0; i < rank1; ++i) {
        divresult = std::div(g_col, dim1[i]);
        map[i] = divresult.rem;
        g_col = divresult.quot;
      }
    }
  }

  return;
};

//! Convert local index to global index using l2g_map.
/*!
  \param[in] lindex Local index
  \param[out] gindex Global index.

  \warning The size of gindex should be larger than the rank of tensor.
*/
template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::global_index_l2g_map(size_t lindex,
                                                    size_t gindex[]) const {
  const size_t rank = Dim.size();
  const size_t rank0 = upper_rank;
  const size_t rank1 = rank - rank0;  // lower_rank
  const size_t lindex_row = Mat.local_row_index(lindex);
  const size_t lindex_col = Mat.local_col_index(lindex);
  const size_t *map_row = &(l2g_map_row[lindex_row * rank0]);
  const size_t *map_col = &(l2g_map_col[lindex_col * rank1]);
  const size_t *axes_map0 = &(axes_map[0]);
  const size_t *axes_map1 = &(axes_map[rank0]);
  for (size_t i = 0; i < rank0; ++i) {
    gindex[axes_map0[i]] = map_row[i];
  }
  for (size_t i = 0; i < rank1; ++i) {
    gindex[axes_map1[i]] = map_col[i];
  }
  return;
};

//! Convert local index to global index of transposed tensor using l2g_map.
/*!
  \param[in] lindex Local index.
  \param[in] axes_trans axes mapping (see below).
  \param[out] index_new Global index of transposed tensor.

  \code axes_trans[i] = axes_inv[axes_map[i]] \endcode
  Here axes_inv is inverse of axes in transpose().

  \warning The size of axes_trans and index_new should be larger than the rank
  of tensor.
*/
template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::global_index_l2g_map_transpose(
    size_t lindex, const size_t axes_trans[], size_t index_new[]) const {
  const size_t rank = Dim.size();
  const size_t rank0 = upper_rank;
  const size_t rank1 = rank - rank0;  // lower_rank
  const size_t lindex_row = Mat.local_row_index(lindex);
  const size_t lindex_col = Mat.local_col_index(lindex);
  const size_t *map_row = &(l2g_map_row[lindex_row * rank0]);
  const size_t *map_col = &(l2g_map_col[lindex_col * rank1]);
  // const size_t* axes_map0 = &(axes_map[0]);
  // const size_t* axes_map1 = &(axes_map[rank0]);
  for (size_t i = 0; i < rank0; ++i) {
    index_new[axes_trans[i]] = map_row[i];
    // index_new[axes_inv[axes_map0[i]]] = map_row[i];
  }
  for (size_t i = 0; i < rank1; ++i) {
    index_new[axes_trans[i + rank0]] = map_col[i];
    // index_new[axes_inv[axes_map1[i]]] = map_col[i];
  }
  return;
};

template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::local_position_fast(size_t g_row, size_t g_col,
                                                   int &comm_rank,
                                                   size_t &local_idx) const {
  Mat.local_position(g_row, g_col, comm_rank, local_idx);
}

/* ---------- member functions ---------- */

//! Initialization.
/*!
  \param shape Shape of tensor.
  \param urank Upper rank for matrix representation.
*/
template <template <typename> class Matrix, typename C>
inline void Tensor<Matrix, C>::init(const Shape &shape, size_t urank) {
  init(shape, urank, range(shape.size()));
}

//! Initialization.
/*!
  \param shape Shape of tensor.
  \param urank Upper rank for matrix representation.
  \param map Axes mapping.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::init(const Shape &shape, size_t urank,
                             const Axes &map) {
  Dim = shape;
  axes_map = map;
  const size_t rank = Dim.size();
  assert(urank <= rank);
  upper_rank = urank;

  size_t n_row = 1;
  size_t n_col = 1;
  for (size_t i = 0; i < upper_rank; ++i) n_row *= shape[map[i]];
  for (size_t i = upper_rank; i < rank; ++i) n_col *= shape[map[i]];
  Mat.init(n_row, n_col);
}

//! Output information of tensor.
/*!
  Every process outputs information to output stream.

  \param out Output stream.
  \param tag (optional) A tag which is inserted at the head of the line.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::print_info(std::ostream &out,
                                   const std::string &tag) const {
  if (tag.size() > 0) {
    out << tag << ":\t";
  }
  out << "Tensor: shape= " << Dim << " upper_rank= " << upper_rank
      << " axes_map= " << axes_map << "\t";
  Mat.print_info(out);
};

//! Output information of tensor.
/*!
  All processes output information of a tensor one by one.
  \note Since this function uses MPI_Barrier, all the processes should call this
  function at the same time. \param out Output stream. \param tag (optional) A
  tag which is inserted at the head of each line.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::print_info_mpi(std::ostream &out,
                                       const std::string &tag) const {
  const int mpisize = get_comm_size();
  const int mpirank = get_comm_rank();
  const typename Tensor<Matrix, C>::comm_type &comm = get_comm();
  Mat.barrier();
  for (int i = 0; i < mpisize; ++i) {
    if (i == mpirank) {
      if (tag.size() > 0) {
        out << tag << ":\t";
      }
      out << "mpirank: " << mpirank << "\t";
      print_info(out, "");
    }
    Mat.barrier();
  }
}

//! Convert global index to local index.
/*!
  If the calling process does not have the element, lindex is not changed.

  \param[in] gindex Global index.
  \param[out] lindex Local index.

  \return True if my process has the element specified by the global index.
*/
template <template <typename> class Matrix, typename C>
bool Tensor<Matrix, C>::local_index(const Index &gindex, size_t &lindex) const {
  const size_t rank = gindex.size();
  assert(rank == Dim.size());
  assert(rank > 0);
  size_t g_row(0), g_col(0);
  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < upper_rank; ++i) {
    size_t j = axes_map[i];
    g_row += gindex[j] * d_row;
    d_row *= Dim[j];
  }
  for (size_t i = upper_rank; i < rank; ++i) {
    size_t j = axes_map[i];
    g_col += gindex[j] * d_col;
    d_col *= Dim[j];
  }
  return Mat.local_index(g_row, g_col, lindex);
};

//! Convert local index to global index
/*!
  \param[in] lindex Local index
  \return Global index
*/
template <template <typename> class Matrix, typename C>
Index Tensor<Matrix, C>::global_index(size_t lindex) const {
  const size_t rank = Dim.size();
  Index gindex;
  size_t g_row, g_col;
  Mat.global_index(lindex, g_row, g_col);
  gindex.resize(rank);
  std::div_t divresult;
  for (size_t i = 0; i < upper_rank; ++i) {
    size_t j = axes_map[i];
    divresult = std::div(int(g_row), int(Dim[j]));
    gindex[j] = divresult.rem;
    g_row = divresult.quot;
  }
  for (size_t i = upper_rank; i < rank; ++i) {
    size_t j = axes_map[i];
    divresult = std::div(int(g_col), int(Dim[j]));
    gindex[j] = divresult.rem;
    g_col = divresult.quot;
  }
  return gindex;
};

//! Convert local index to global index fast.
/*!
  \param[in] lindex Local index
  \param[out] gindex Global index.

  \warning The size of gindex should be larger than the rank of tensor.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::global_index_fast(size_t lindex, Index &gindex) const {
  const size_t rank = Dim.size();
  size_t g_row, g_col;
  Mat.global_index(lindex, g_row, g_col);
  std::div_t divresult;
  for (size_t i = 0; i < upper_rank; ++i) {
    size_t j = axes_map[i];
    divresult = std::div(int(g_row), int(Dim[j]));
    gindex[j] = divresult.rem;
    g_row = divresult.quot;
  }
  for (size_t i = upper_rank; i < rank; ++i) {
    size_t j = axes_map[i];
    divresult = std::div(int(g_col), int(Dim[j]));
    gindex[j] = divresult.rem;
    g_col = divresult.quot;
  }
  return;
};

//! Get an element.
/*!
  \param[in] idx Global index.
  \param[out] val Value of the element.
  \return True if my process has the element specified by the global index.
*/
template <template <typename> class Matrix, typename C>
bool Tensor<Matrix, C>::get_value(const Index &idx, C &val) const {
  size_t li;
  if (local_index(idx, li)) {
    val = Mat[li];
    return true;
  } else {
    return false;
  }
}

//! Set an element.
/*!
  If my process does not have the element, this function does nothing.

  \param[in] idx Global index.
  \param[in] val Value of the element.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::set_value(const Index &idx, C val) {
  size_t li;
  if (local_index(idx, li)) {
    Mat[li] = val;
  }
}

//! Calculate rank which has the given global index and its local index.
/*!
  \param[in] index Global index.
  \param[out] comm_rank Rank which has the element at the global index.
  \param[out] local_idx Local index in comm_rank.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::local_position(const Index &index, int &comm_rank,
                                       size_t &local_idx) const {
  const size_t rank = Dim.size();
  size_t g_row(0), g_col(0);
  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < upper_rank; ++i) {
    size_t j = axes_map[i];
    g_row += index[j] * d_row;
    d_row *= Dim[j];
  }
  for (size_t i = upper_rank; i < rank; ++i) {
    size_t j = axes_map[i];
    g_col += index[j] * d_col;
    d_col *= Dim[j];
  }
  Mat.local_position(g_row, g_col, comm_rank, local_idx);
  return;
}

//! Change the upper rank and the axes map.
/*!
  \param[in] new_upper_rank New upper rank.
  \param[in] new_axes_map New axes map.

  \note This function may cause communications.
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::change_configuration(const size_t new_upper_rank,
                                             const Axes &new_axes_map) {
  if ((upper_rank == new_upper_rank) && (axes_map == new_axes_map)) return;

  Tensor<Matrix, C> T_old(*this);

  const size_t rank = this->rank();
  Shape dim;
  Axes axes;
  dim.resize(rank);
  axes.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    dim[i] = Dim[new_axes_map[i]];
    axes[new_axes_map[i]] = i;
  }
  init(dim, new_upper_rank);
  transpose(axes);

  /* create lists of local position and destination rank */
  const size_t local_size = T_old.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  T_old.prep_local_to_global();
  this->prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index;
    index.resize(rank);

#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      T_old.global_index_fast(i, index);

      int dest;
      size_t pos;
      this->local_position(index, dest, pos);

      local_position[i] = pos;
      dest_mpirank[i] = dest;
    }
  }

  /* exchange data */
  replace_matrix_data(T_old.get_matrix(), dest_mpirank, local_position, Mat);

  return;
}

//! Transposition of tensor with lazy evaluation
/*!
  When \c axes=[1,2,0], \f$ T_{ijk} \f$ is transformed into \f$ T_{jki} \f$.
  \param[in] axes Order of axes.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::transpose(const Axes &axes) {
  const size_t rank = Dim.size();
  assert(debug::check_transpose_axes(axes, rank));

  Shape dim_now = Dim;
  Axes map_now = axes_map;
  Axes axes_inv;
  axes_inv.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    axes_inv[axes[i]] = i;
  }

  for (size_t i = 0; i < rank; ++i) {
    Dim[i] = dim_now[axes[i]];
    axes_map[i] = axes_inv[map_now[i]];
  }

  return (*this);
}

//! Element-wise vector multiplication.
/*!
  For example, \c T.multiply_vector(v,1) is equivalent to
  \f$
  T_{ijk} := v_j T_{ijk}.
  \f$
  \param vec Vector
  \param n_axes Axes to multiply the vector
  \attention The size of \c vec should be larger than the bond dimension of \c
  n_axes.
*/
template <template <typename> class Matrix, typename C>
template <typename D>
Tensor<Matrix, C> &Tensor<Matrix, C>::multiply_vector(const std::vector<D> &vec,
                                                      size_t n_axes) {
  assert(Dim[n_axes] <= vec.size());
  const size_t local_size = this->local_size();
  prep_local_to_global();
#pragma omp parallel default(shared)
  {
    Index idx;
    idx.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      global_index_fast(i, idx);
      Mat[i] *= vec[idx[n_axes]];
    }
  }
  return (*this);
};

//! Element-wise vector multiplication.
/*!
  For example, \c T.multiply_vector(v,1,w,0) is equivalent to
  \f$
  T_{ijk} := v_j w_i T_{ijk}.
  \f$
  \param vec0 Vector
  \param n_axes0 Axes to multiply the vector \c vec0.
  \param vec1 Vector
  \param n_axes1 Axes to multiply the vector \c vec1.
  \attention The size of \c vecX should be larger than the bond dimension of \c
  n_axesX.
*/
template <template <typename> class Matrix, typename C>
template <typename D0, typename D1>
Tensor<Matrix, C> &Tensor<Matrix, C>::multiply_vector(
    const std::vector<D0> &vec0, size_t n_axes0, const std::vector<D1> &vec1,
    size_t n_axes1) {
  assert(Dim[n_axes0] <= vec0.size());
  assert(Dim[n_axes1] <= vec1.size());
  const size_t local_size = this->local_size();
  prep_local_to_global();
#pragma omp parallel default(shared)
  {
    Index idx;
    idx.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      global_index_fast(i, idx);
      Mat[i] *= vec0[idx[n_axes0]] * vec1[idx[n_axes1]];
    }
  }
  return (*this);
};

//! Element-wise vector multiplication.
/*!
  For example, \c T.multiply_vector(v,1,w,0,x,2) is equivalent to
  \f$
  T_{ijkl} := v_j w_i x_k T_{ijkl}.
  \f$
  \param vec0 Vector
  \param n_axes0 Axes to multiply the vector \c vec0.
  \param vec1 Vector
  \param n_axes1 Axes to multiply the vector \c vec1.
  \param vec2 Vector
  \param n_axes2 Axes to multiply the vector \c vec2.
  \attention The size of \c vecX should be larger than the bond dimension of \c
  n_axesX.
*/
template <template <typename> class Matrix, typename C>
template <typename D0, typename D1, typename D2>
Tensor<Matrix, C> &Tensor<Matrix, C>::multiply_vector(
    const std::vector<D0> &vec0, size_t n_axes0, const std::vector<D1> &vec1,
    size_t n_axes1, const std::vector<D2> &vec2, size_t n_axes2) {
  assert(Dim[n_axes0] <= vec0.size());
  assert(Dim[n_axes1] <= vec1.size());
  assert(Dim[n_axes2] <= vec2.size());
  const size_t local_size = this->local_size();
  prep_local_to_global();
#pragma omp parallel default(shared)
  {
    Index idx;
    idx.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      global_index_fast(i, idx);
      Mat[i] *= vec0[idx[n_axes0]] * vec1[idx[n_axes1]] * vec2[idx[n_axes2]];
    }
  }
  return (*this);
};

//! Element-wise vector multiplication.
/*!
  For example, \c T.multiply_vector(v,1,w,0,x,2,y,4) is equivalent to
  \f$
  T_{ijklm} := v_j w_i x_k y_m T_{ijklm}.
  \f$
  \param vec0 Vector
  \param n_axes0 Axes to multiply the vector \c vec0.
  \param vec1 Vector
  \param n_axes1 Axes to multiply the vector \c vec1.
  \param vec2 Vector
  \param n_axes2 Axes to multiply the vector \c vec2.
  \param vec3 Vector
  \param n_axes3 Axes to multiply the vector \c vec3.
  \attention The size of \c vecX should be larger than the bond dimension of \c
  n_axesX.
*/
template <template <typename> class Matrix, typename C>
template <typename D0, typename D1, typename D2, typename D3>
Tensor<Matrix, C> &Tensor<Matrix, C>::multiply_vector(
    const std::vector<D0> &vec0, size_t n_axes0, const std::vector<D1> &vec1,
    size_t n_axes1, const std::vector<D2> &vec2, size_t n_axes2,
    const std::vector<D3> &vec3, size_t n_axes3) {
  assert(Dim[n_axes0] <= vec0.size());
  assert(Dim[n_axes1] <= vec1.size());
  assert(Dim[n_axes2] <= vec2.size());
  assert(Dim[n_axes3] <= vec3.size());
  const size_t local_size = this->local_size();
  prep_local_to_global();
#pragma omp parallel default(shared)
  {
    Index idx;
    idx.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      global_index_fast(i, idx);
      Mat[i] *= vec0[idx[n_axes0]] * vec1[idx[n_axes1]] * vec2[idx[n_axes2]] *
                vec3[idx[n_axes3]];
    }
  }
  return (*this);
};

//! Inverse of slice().
/*!
  <tt>T.set_slice( slice(T,r,i,j), r,i,j)</tt> does nothing.
  For example, <tt>T.set_slice(A,1,4,10);</tt> is equal to
  <tt>T[:,4:10,:]=A[:,:,:]</tt> in Python.

  \param[in] a A sliced tensor to be set.
  \param[in] n_axes sliced Axes.
  \param[in] i_begin Start index of slice.
  \param[in] i_end End index of slice.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::set_slice(const Tensor<Matrix, C> &a,
                                                const size_t n_axes,
                                                const size_t i_begin,
                                                const size_t i_end) {
  assert(rank() == a.rank());
  assert(n_axes < rank());
  assert(i_begin < i_end);
  assert(i_end <= shape()[n_axes]);
  assert(i_end - i_begin == a.shape()[n_axes]);

  /* create lists of local position and destination rank */
  const size_t local_size = a.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  a.prep_local_to_global();
  prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index;
    index.resize(rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      a.global_index_fast(i, index);
      index[n_axes] += i_begin;

      int dest;
      size_t pos;
      this->local_position(index, dest, pos);

      local_position[i] = pos;
      dest_mpirank[i] = dest;
    }
  }

  /* exchange data */
  replace_matrix_data(a.get_matrix(), dest_mpirank, local_position,
                      get_matrix());

  return (*this);
}

//! Inverse of slice().
/*!
  If <tt>index_begin[r]==index_end[r]</tt>, this rank is not sliced.
  <tt>T.set_slice( slice(T,i,j), i,j)</tt> does nothing.
  For example, <tt>T.set_slice(A,Index(1,0,3),Index(4,0,6));</tt>
  is equal to <tt>T[1:4,:,3:6]=A[:,:,:]</tt> in Python.

  \param[in] a A sliced tensor to be set.
  \param[in] index_begin Start indices.
  \param[in] index_end End indices.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::set_slice(const Tensor<Matrix, C> &a,
                                                const Index &index_begin,
                                                const Index &index_end) {
  const size_t nr = rank();
  assert(nr == a.rank());
  assert(index_begin.size() == nr);
  assert(index_end.size() == nr);
  for (size_t r = 0; r < nr; ++r) {
    assert(index_end[r] <= shape()[r]);
    assert(index_begin[r] <= index_end[r]);
  }

  /* create lists of local position and destination rank */
  const size_t local_size = a.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  a.prep_local_to_global();
  prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index;
    index.resize(nr);
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      a.global_index_fast(i, index);
      for (size_t r = 0; r < nr; ++r) {
        if (index_begin[r] != index_end[r]) index[r] += index_begin[r];
      }

      int dest;
      size_t pos;
      this->local_position(index, dest, pos);

      local_position[i] = pos;
      dest_mpirank[i] = dest;
    }
  }

  /* exchange data */
  replace_matrix_data(a.get_matrix(), dest_mpirank, local_position,
                      get_matrix());

  return (*this);
}


//! Gather all elements into a non-distributed tensor.
/*!
  \return A non-distributed global tensor.
  \attention A non-distributed global tensor may require huge memory.
*/
template <template <typename> class Matrix, typename C>
Tensor<lapack::Matrix, C> Tensor<Matrix, C>::gather() {
  const size_t n = rank();
  if (!(axes_map == range(n))) {
    change_configuration(n / 2, range(n));
  }
  Tensor<lapack::Matrix, C> T(get_comm(), get_matrix().flatten());
  return reshape(T, Dim);
}


//! Return a copy of the flattened vector.
/*!
  \return A flattened vector, which is global (not distriuted).
  \attention A flattened vector may require huge memory.
*/
template <template <typename> class Matrix, typename C>
std::vector<C> Tensor<Matrix, C>::flatten() {
  const size_t n = rank();
  if (!(axes_map == range(n))) {
    change_configuration(n / 2, range(n));
  }
  return get_matrix().flatten();
}

//! Addition assignment.
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::operator+=(const Tensor<Matrix, C> &rhs) {
  assert(Dim == rhs.shape());
  change_configuration(rhs.get_upper_rank(), rhs.get_axes_map());
  Mat += rhs.get_matrix();
  return (*this);
}

//! Subtraction assignment.
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::operator-=(const Tensor<Matrix, C> &rhs) {
  assert(Dim == rhs.shape());
  change_configuration(rhs.get_upper_rank(), rhs.get_axes_map());
  Mat -= rhs.get_matrix();
  return (*this);
}

//! Scalar-multiplication assignment.
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::operator*=(C rhs) {
  Mat *= rhs;
  return (*this);
}

//! Scalar-division assignment.
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::operator/=(C rhs) {
  Mat /= rhs;
  return (*this);
}

//! Initialize all elements by rhs.
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> &Tensor<Matrix, C>::operator=(C rhs) {
  const size_t n = Mat.local_size();
  for (int i = 0; i < n; ++i) Mat[i] = rhs;
  return (*this);
}

//! Apply a unary operation to each element.
/*!
  \param op Unary operation.
  \return Result.
*/
template <template <typename> class Matrix, typename C>
template <typename UnaryOperation>
Tensor<Matrix, C> &Tensor<Matrix, C>::map(UnaryOperation op) {
  Mat.map(op);
  return (*this);
}

/* ---------- non-member functions (Opeations) ---------- */

//! Transposition of tensor with lazy evaluation
/*!
  When \c axes=[1,2,0], \f$ T_{ijk} \f$ is transformed into \f$ T_{jki} \f$.
  \param[in] T Tensor to be transposed.
  \param[in] axes Order of axes
  \return Transposed tensor
*/
template <template <typename> class Matrix, typename C>
inline Tensor<Matrix, C> transpose(Tensor<Matrix, C> T, const Axes &axes) {
  return T.transpose(axes);
}

//! Transposition of tensor \b without lazy evaluation
/*!
  When \c axes=[1,2,0], \f$ T_{ijk} \f$ is transformed into \f$ T_{jki} \f$.
  \param[in] T Tensor to be transposed.
  \param[in] axes Order of axes.
  \param[in] urank_new Upper rank of new tensor.
  \return Transposed tensor

  \note This function may cause communications.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> transpose(const Tensor<Matrix, C> &T, const Axes &axes,
                            size_t urank_new) {
  const size_t rank = T.rank();
  assert(debug::check_transpose_axes(axes, rank));
  assert(urank_new <= rank);

  if (urank_new == T.get_upper_rank()) {
    if (is_no_transpose(axes, T.get_axes_map(), rank)) return T;
  }

  /* new index dimension */
  Shape dim_old = T.shape();
  Shape dim_new;
  dim_new.resize(rank);
  for (size_t r = 0; r < rank; ++r) {
    dim_new[r] = dim_old[axes[r]];
  }

  /* new tensor */
  Tensor<Matrix, C> T_new(T.get_comm(), dim_new, urank_new);

  /* create lists of local position and destination rank */
  const size_t local_size = T.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  T.make_l2g_map();
  T_new.prep_global_to_local();

#pragma omp parallel default(shared)
  {
    // Assuming axes_map of T_new is [0,1,2,...]
    size_t d_offset[rank];
    size_t d_row(1), d_col(1);
    for (size_t r = 0; r < urank_new; ++r) {
      d_offset[r] = d_row;
      d_row *= dim_new[r];
    }
    for (size_t r = urank_new; r < rank; ++r) {
      d_offset[r] = d_col;
      d_col *= dim_new[r];
    }

    Axes axes_map = T.get_axes_map();
    size_t axes_inv[rank];
    size_t axes_trans[rank];
    for (size_t r = 0; r < rank; ++r) {
      axes_inv[axes[r]] = r;
    }
    for (size_t r = 0; r < rank; ++r) {
      axes_trans[r] = axes_inv[axes_map[r]];
    }

    size_t index_new[rank];

#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      T.global_index_l2g_map_transpose(i, axes_trans, index_new);

      size_t g_row(0),
          g_col(0);  // indices of global matrix for transposed tensor
      for (size_t r = 0; r < urank_new; ++r) {
        g_row += index_new[r] * d_offset[r];
      }
      for (size_t r = urank_new; r < rank; ++r) {
        g_col += index_new[r] * d_offset[r];
      }

      T_new.local_position_fast(g_row, g_col, dest_mpirank[i],
                                local_position[i]);
    }
  }

  /* exchange data */
  replace_matrix_data(T.get_matrix(), dest_mpirank, local_position,
                      T_new.get_matrix());

  return T_new;
};

//! Change the shape of tensor.
/*!

  \param[in] T Tensor to be reshaped.
  \param[in] shape_new New shape.

  \return Reshaped tensor.

  \note The new shape should be compatible with the original shape.
  The total size of tensor does not change.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> reshape(const Tensor<Matrix, C> &T, const Shape &shape_new) {
  assert(debug::check_total_size(shape_new, T.shape()));

  const Shape &shape = T.shape();
  const size_t rank = shape.size();
  const size_t rank_new = shape_new.size();

  /* initialize new tensor */
  Tensor<Matrix, C> T_new(T.get_comm(), shape_new);

  /* create lists of local position and destination rank */
  const size_t local_size = T.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  T.prep_local_to_global();
  T_new.prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index, index_new;
    index.resize(rank);
    index_new.resize(rank_new);
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      T.global_index_fast(i, index);
      size_t global_position(0);
      size_t dim(1);
      for (size_t r = 0; r < rank; ++r) {
        global_position += index[r] * dim;
        dim *= shape[r];
      }

      for (size_t r = 0; r < rank_new; ++r) {
        index_new[r] = global_position % shape_new[r];
        global_position /= shape_new[r];
      }

      int dest;
      size_t pos;
      T_new.local_position(index_new, dest, pos);

      local_position[i] = pos;
      dest_mpirank[i] = dest;
    }
  }

  /* exchange data */
  replace_matrix_data(T.get_matrix(), dest_mpirank, local_position,
                      T_new.get_matrix());

  return T_new;
};

//! Slice a tensor.
/*!
  This function mimics slicing in python such as \c [start:end].

  \param[in] T Tensor to be sliced.
  \param[in] n_axes Axes to be sliced.
  \param[in] i_begin Start of index.
  \param[in] i_end End of index.

  \return Sliced tensor.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> slice(const Tensor<Matrix, C> &T, size_t n_axes,
                        size_t i_begin, size_t i_end) {
  const int mpisize = T.get_comm_size();
  const Shape &shape = T.shape();
  assert(n_axes < T.rank());
  assert(i_begin < i_end);
  assert(i_end <= shape[n_axes]);

  Shape shape_new = shape;
  shape_new[n_axes] = i_end - i_begin;

  /* initialize new tensor */
  Tensor<Matrix, C> T_new(T.get_comm(), shape_new);

  /* create lists of local position and destination rank */
  const size_t local_size = T.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  T.prep_local_to_global();
  T_new.prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index;
    index.resize(T.rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      T.global_index_fast(i, index);
      if (index[n_axes] >= i_begin && index[n_axes] < i_end) {
        index[n_axes] -= i_begin;

        int dest;
        size_t pos;
        T_new.local_position(index, dest, pos);

        local_position[i] = pos;
        dest_mpirank[i] = dest;
      } else {
        // not send
        local_position[i] = 0;
        dest_mpirank[i] = mpisize;
      }
    }
  }

  /* exchange data */
  replace_matrix_data(T.get_matrix(), dest_mpirank, local_position,
                      T_new.get_matrix());

  return T_new;
}

//! Slice a tensor.
/*!
  This function mimics slicing in python such as \c [start:end].
  If <tt>index_begin[r]==index_end[r]</tt>, this rank is not sliced.
  For example,
  \code slice(T, Index(1,0,3), Index(4,0,6)); \endcode
  will return T[1:4, :, 3:6].

  \param[in] T Tensor to be sliced.
  \param[in] index_begin Start of indices.
  \param[in] index_end End of indices.

  \return Sliced tensor.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> slice(const Tensor<Matrix, C> &T, const Index &index_begin,
                        const Index &index_end) {
  const int mpisize = T.get_comm_size();
  const Shape &shape = T.shape();
  const size_t rank = T.rank();
  assert(index_begin.size() == rank);
  assert(index_end.size() == rank);

  Shape shape_new = shape;
  for (size_t r = 0; r < rank; ++r) {
    assert(index_end[r] <= shape[r]);
    assert(index_begin[r] <= index_end[r]);
    shape_new[r] = index_end[r] - index_begin[r];
    if (shape_new[r] == 0) shape_new[r] = shape[r];
  }

  /* initialize new tensor */
  Tensor<Matrix, C> T_new(T.get_comm(), shape_new);

  /* create lists of local position and destination rank */
  const size_t local_size = T.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  T.prep_local_to_global();
  T_new.prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index;
    index.resize(T.rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      T.global_index_fast(i, index);

      bool is_send = true;
      for (size_t r = 0; r < rank; ++r) {
        size_t idx = index[r];
        if (index_begin[r] == index_end[r]) {
          continue;
        } else if (idx >= index_begin[r] && idx < index_end[r]) {
          index[r] -= index_begin[r];
        } else {
          is_send = false;
          break;
        }
      }

      if (is_send) {
        int dest;
        size_t pos;
        T_new.local_position(index, dest, pos);

        local_position[i] = pos;
        dest_mpirank[i] = dest;
      } else {
        // not send
        local_position[i] = 0;
        dest_mpirank[i] = mpisize;
      }
    }
  }

  /* exchange data */
  replace_matrix_data(T.get_matrix(), dest_mpirank, local_position,
                      T_new.get_matrix());

  return T_new;
};

//! Extend the size of a tensor.
/*!
  \param[in] T Tensor to be extended.
  \param[in] shape_new New shape.

  \return Extended tensor.

  \note New shape should be larger than the original shape.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> extend(const Tensor<Matrix, C> &T, const Shape &shape_new) {
  assert(T.rank() == shape_new.size());
  assert(debug::check_extend(T.shape(), shape_new));

  /* initialize new tensor */
  Tensor<Matrix, C> T_new(T.get_comm(), shape_new);

  /* create lists of local position and destination rank */
  const size_t local_size = T.local_size();
  std::vector<int> dest_mpirank(local_size);
  std::vector<unsigned long int> local_position(local_size);

  T.prep_local_to_global();
  T_new.prep_global_to_local();

#pragma omp parallel default(shared)
  {
    Index index;
    index.resize(T.rank());
#pragma omp for
    for (size_t i = 0; i < local_size; ++i) {
      T.global_index_fast(i, index);

      int dest;
      size_t pos;
      T_new.local_position(index, dest, pos);

      local_position[i] = pos;
      dest_mpirank[i] = dest;
    }
  }

  /* exchange data */
  replace_matrix_data(T.get_matrix(), dest_mpirank, local_position,
                      T_new.get_matrix());

  return T_new;
}

//! Trace of matrix (rank-2 tensor)
/*!
  \param M Rank-2 tensor

  \return Value of trace. \f$ \sum_i M_{ii} \f$
*/
template <template <typename> class Matrix, typename C>
C trace(const Tensor<Matrix, C> &M) {
  assert(M.rank() == 2);
  return matrix_trace(M.get_matrix());
};

//! Full trace of tensor.
/*!
  \c axes_1[k] and \c axes_2[k] are contracted. For example,
  \code
  trace(T, Axes(0,1,3), Axes(2,4,5))
  \endcode
  returns \f$ \sum_{ijk} T_{ijikjk}\f$.

  \param T Tensor to trace.
  \param axes_1 Axes of tensor.
  \param axes_2 Axes of tensor.

  \return Value of tensor.
*/
template <template <typename> class Matrix, typename C>
C trace(const Tensor<Matrix, C> &T, const Axes &axes_1, const Axes &axes_2) {
  assert(axes_1.size() == axes_2.size());
  assert(axes_1.size() + axes_2.size() == T.rank());
  assert(debug::check_trace_axes(axes_1, axes_2, T.rank()));

  if (T.rank() == 2) return trace(T);

  const size_t n = T.local_size();
  const size_t l = axes_1.size();
  Index index;
  bool check;
  C sum(0.0);

  index.resize(T.rank());
  for (size_t i = 0; i < n; ++i) {
    T.global_index_fast(i, index);
    check = true;
    for (size_t k = 0; k < l; ++k) {
      if (index[axes_1[k]] != index[axes_2[k]]) {
        check = false;
        break;
      }
    }
    if (check) {
      sum += T[i];
    }
  }

  return T.get_matrix().allreduce_sum(sum);
};

//! Full contraction of two tensors.
/*!
  \c axes_a[k] and \c axes_b[k] are contracted. For example,
  \code trace(A, B, Axes(0,1,2), Axes(2,0,1)) \endcode
  returns \f$ \sum_{ijk} A_{ijk} B_{kij} \f$.
  \param A Tensor to contract.
  \param B Tensor to contract.
  \param axes_a Axes order of tensor \c A.
  \param axes_b Axes order of tensor \c B.

  \return Value of full contraction.

  \note This function is more effective than tensordot and MPI_Bcast.
*/
template <template <typename> class Matrix, typename C>
C trace(const Tensor<Matrix, C> &A, const Tensor<Matrix, C> &B,
        const Axes &axes_a, const Axes &axes_b) {
  assert(A.rank() == B.rank());
  assert(A.rank() == axes_a.size());
  assert(B.rank() == axes_b.size());
  assert(debug::check_trace_axes(axes_a, axes_b, A.shape(), B.shape()));

  const size_t rank = A.rank();
  Axes axes;
  Axes axes_a_inv;
  Axes axes_map = A.get_axes_map();
  axes.resize(rank);
  axes_a_inv.resize(rank);

  for (size_t i = 0; i < rank; ++i) {
    axes_a_inv[axes_a[i]] = i;
  }
  for (size_t i = 0; i < rank; ++i) {
    axes[i] = axes_b[axes_a_inv[axes_map[i]]];
  }

  const size_t n = A.local_size();
  Tensor<Matrix, C> B_t = transpose(B, axes, A.get_upper_rank());
  C sum(0.0);

  for (size_t i = 0; i < n; ++i) {
    sum += A[i] * B_t[i];
  }

  return A.get_matrix().allreduce_sum(sum);
}

//! Partial trace of tensor.
/*!
  \c axes_1[k] and \c axes_2[k] are contracted. For example.
  \code B = trace(A, Axes(0,2), Axes(5,4)) \endcode
  returns \f$ B_{ab} = \sum_{ij} A_{iajbji} \f$.
  \param T Tensor to partially trace.
  \param axes_1 Axes to trace.
  \param axes_2 Axes to trace.

  \return Contracted tensor.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> contract(const Tensor<Matrix, C> &T, const Axes &axes_1,
                           const Axes &axes_2) {
  const int mpisize = T.get_comm_size();
  assert(axes_1.size() == axes_2.size());
  assert(axes_1.size() + axes_2.size() < T.rank());
  if (axes_1.size() == 0) return T;
  assert(debug::check_contract_axes(axes_1, axes_2, T.rank()));

  Shape shape = T.shape();
  Shape shape_new;
  Axes axes_new;
  {
    Axes v = axes_1 + axes_2;
    v.sort();
    size_t k = 0;
    size_t n = v.size();
    for (size_t i = 0; i < T.rank(); ++i) {
      if (k < n && i == v[k]) {
        ++k;
      } else {
        shape_new.push(shape[i]);
        axes_new.push(i);
      }
    }
  }

  /* initialize new tensor */
  Tensor<Matrix, C> T_new(T.get_comm(), shape_new);

  /* create lists of local position and destination rank */
  std::vector<int> dest_mpirank(T.local_size());
  std::vector<unsigned long int> local_position(T.local_size());

  Index index, index_new;
  index.resize(T.rank());
  index_new.resize(T_new.rank());
  bool check;
  const size_t n = axes_1.size();

  for (size_t i = 0; i < T.local_size(); ++i) {
    T.global_index_fast(i, index);
    check = true;
    for (size_t k = 0; k < n; ++k) {
      if (index[axes_1[k]] != index[axes_2[k]]) {
        check = false;
        continue;
      }
    }

    if (check) {
      for (size_t k = 0; k < index_new.size(); ++k) {
        index_new[k] = index[axes_new[k]];
      }

      int dest;
      size_t pos;
      T_new.local_position(index_new, dest, pos);

      local_position[i] = pos;
      dest_mpirank[i] = dest;
    } else {
      // not send
      local_position[i] = 0;
      dest_mpirank[i] = mpisize;
    }
  }

  /* sum data */
  sum_matrix_data(T.get_matrix(), dest_mpirank, local_position,
                  T_new.get_matrix());

  return T_new;
}

//! Compute tensor dot product.
/*!
  For example, \f$ T_{abcd} = \sum_{ij} A_{iajb} B_{cjdi} \f$ is
  \code T = tensordot(A, B, Axes(0,2), Axes(3,1)); \endcode

  \param a Tensor.
  \param b Tensor.
  \param axes_a Axes of tensor \c a to contract.
  \param axes_b Axes of tensor \c b to contract.

  \return Result.
*/
template <template <typename> class Matrix, typename C>
Tensor<Matrix, C> tensordot(const Tensor<Matrix, C> &a,
                            const Tensor<Matrix, C> &b, const Axes &axes_a,
                            const Axes &axes_b) {
  assert(axes_a.size() == axes_b.size());
  assert(a.get_comm() == b.get_comm());
  Shape shape_a = a.shape();
  Shape shape_b = b.shape();
  for (size_t i = 0; i < axes_a.size(); ++i) {
    assert(shape_a[axes_a[i]] == shape_b[axes_b[i]]);
  }

  const typename Tensor<Matrix, C>::comm_type &comm = a.get_comm();

  Shape shape_c;
  const size_t rank_row_c = a.rank() - axes_a.size();
  const size_t rank_col_c = b.rank() - axes_b.size();
  shape_c.resize(rank_row_c + rank_col_c);

  Axes trans_axes_a;
  Axes trans_axes_b;
  size_t urank_a;
  size_t urank_b;

  {
    const size_t rank = a.rank();
    const size_t rank_row = rank - axes_a.size();
    const size_t rank_col = axes_a.size();
    size_t v[rank];
    for (size_t i = 0; i < rank; ++i) v[i] = i;
    for (size_t i = 0; i < rank_col; ++i) v[axes_a[i]] = rank;
    std::sort(v, v + rank);
    for (size_t i = 0; i < rank_col; ++i) v[rank_row + i] = axes_a[i];

    trans_axes_a.assign(rank, v);
    urank_a = rank_row;

    for (size_t i = 0; i < rank_row; ++i) shape_c[i] = shape_a[v[i]];
  }

  {
    const size_t rank = b.rank();
    const size_t rank_row = axes_b.size();
    const size_t rank_col = rank - axes_b.size();
    size_t v[rank];
    for (size_t i = 0; i < rank; ++i) v[i] = i;
    for (size_t i = 0; i < rank_row; ++i) v[axes_b[i]] = 0;
    std::sort(v, v + rank);
    for (size_t i = 0; i < rank_row; ++i) v[i] = axes_b[i];

    trans_axes_b.assign(rank, v);
    urank_b = rank_row;

    for (size_t i = 0; i < rank_col; ++i)
      shape_c[i + rank_row_c] = shape_b[v[i + rank_row]];
  }

  Tensor<Matrix, C> c(comm, shape_c, rank_row_c);
  matrix_product(transpose(a, trans_axes_a, urank_a).get_matrix(),
                 transpose(b, trans_axes_b, urank_b).get_matrix(),
                 c.get_matrix());

  return c;
};

//! Singular value decomposition for rank-2 tensor (matrix)
/*!
  This function only caluculates singular values.

  \param[in] a Rank-2 tensor to be decomposed.
  \param[out] s Singular values.

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int svd(const Tensor<Matrix, C> &a, std::vector<double> &s) {
  assert(a.rank() == 2);
  int info;
  info = svd(a, Axes(0), Axes(1), s);
  return info;
}

//! Singular value decomposition for rank-2 tensor (matrix)
/*!
  \f[
  a_{ij} = \sum_k u_{ik} s_k (v^{\dagger})_{kj}
  \f]

  \param[in] a Rank-2 tensor to be decomposed.
  \param[out] u Tensor correspond to left singular vectors.
  \param[out] s Singular values.
  \param[out] vt Tensor correspond to right singular vectors.

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int svd(const Tensor<Matrix, C> &a, Tensor<Matrix, C> &u,
        std::vector<double> &s, Tensor<Matrix, C> &vt) {
  assert(a.rank() == 2);
  int info;
  info = svd(a, Axes(0), Axes(1), u, s, vt);
  return info;
}

//! Singular value decomposition for tensor.
/*!
  This function only calculates singular values.

  \param[in] a Tensor to be decomposed.
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] s Singular values.

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int svd(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
        std::vector<double> &s) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);
  assert(debug::check_svd_axes(axes_row, axes_col, a.rank()));

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();

  Axes axes = axes_row + axes_col;

  Tensor<Matrix, C> a_t = transpose(a, axes, urank);
  const Shape &shape = a_t.shape();

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;

  s.resize(size);

  int info;
  info = matrix_svd(a_t.get_matrix(), s);

  return info;
}

//! Singular value decomposition for tensor.
/*! This may be useful for creation of an isometry.
  For example,
  \code
  svd(A, Axes(0,3), Axes(1,2), U, S, VT)
  \endcode
  calculates the following decomposition.
  \f[
  A_{abcd} = \sum_i U_{adi} S_i (V^\dagger)_{ibc}.
  \f]

  \param[in] a A tensor to be decomposed.
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] u Tensor \f$ U \f$ corresponds to left singular vectors.
  \param[out] s Singular values.
  \param[out] vt Tensor \f$ V^\dagger \f$ corresponds to right singular vectors.

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int svd(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
        Tensor<Matrix, C> &u, std::vector<double> &s, Tensor<Matrix, C> &vt) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);
  assert(debug::check_svd_axes(axes_row, axes_col, a.rank()));

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();

  Axes axes = axes_row + axes_col;

  Tensor<Matrix, C> a_t = transpose(a, axes, urank);
  const Shape &shape = a_t.shape();

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;

  Shape shape_u;
  shape_u.resize(urank + 1);
  for (size_t i = 0; i < urank; ++i) shape_u[i] = shape[i];
  shape_u[urank] = size;

  Shape shape_vt;
  shape_vt.resize(rank - urank + 1);
  shape_vt[0] = size;
  for (size_t i = urank; i < rank; ++i) shape_vt[i - urank + 1] = shape[i];

  size_t urank_u = urank;
  size_t urank_vt = 1;

  u = Tensor<Matrix, C>(a.get_comm(), shape_u, urank_u);
  vt = Tensor<Matrix, C>(a.get_comm(), shape_vt, urank_vt);
  s.resize(size);

  int info;
  info = matrix_svd(a_t.get_matrix(), u.get_matrix(), s, vt.get_matrix());

  return info;
}

//! Partial SVD for rank-2 tensor (matrix)
/*!
  This function only caluculates singular values.

  \param[in] a Rank-2 tensor to be decomposed.
  \param[out] s Singular values.
  \param[in] target_rank The number of singular values to be calculated.

  \return Information from linear-algebra library.

  \attention Computational cost is the same as full SVD.
*/
template <template <typename> class Matrix, typename C>
int psvd(const Tensor<Matrix, C> &a, std::vector<double> &s,
         const size_t target_rank) {
  int info = svd(a, s);
  if (s.size() > target_rank) s.resize(target_rank);
  return info;
}

//! Partial SVD for rank-2 tensor (matrix)
/*!
  \f[
  a_{ij} = \sum_k u_{ik} s_k (v^{\dagger})_{kj}
  \f]

  \param[in] a Rank-2 tensor to be decomposed.
  \param[out] u Tensor correspond to left singular vectors.
  \param[out] s Singular values.
  \param[out] vt Tensor correspond to right singular vectors.
  \param[in] target_rank The number of singular values to be calculated.

  \return Information from linear-algebra library.

  \attention Computational cost is the same as full SVD.
*/
template <template <typename> class Matrix, typename C>
int psvd(const Tensor<Matrix, C> &a, Tensor<Matrix, C> &u,
         std::vector<double> &s, Tensor<Matrix, C> &vt,
         const size_t target_rank) {
  int info = svd(a, u, s, vt);
  if (s.size() > target_rank) {
    s.resize(target_rank);
    u = slice(u, 1, 0, target_rank);
    vt = slice(vt, 0, 0, target_rank);
  }
  return info;
}

//! Partial SVD for tensor.
/*!
  This function only calculates singular values.

  \param[in] a Tensor to be decomposed.
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] s Singular values.
  \param[in] target_rank The number of singular values to be calculated.

  \return Information from linear-algebra library.

  \attention Computational cost is the same as full SVD.
*/
template <template <typename> class Matrix, typename C>
int psvd(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
         std::vector<double> &s, const size_t target_rank) {
  int info = svd(a, axes_row, axes_col, s);
  if (s.size() > target_rank) s.resize(target_rank);
  return info;
}

//! Partial SVD for tensor.
/*! This may be useful for creation of an isometry.
  For example,
  \code
  psvd(A, Axes(0,3), Axes(1,2), U, S, VT, n)
  \endcode
  calculates the following decomposition.
  \f[
  A_{abcd} \simeq \sum_{i=0}^{n-1} U_{adi} S_i (V^\dagger)_{ibc}.
  \f]

  \param[in] a A tensor to be decomposed.
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] u Tensor \f$ U \f$ corresponds to left singular vectors.
  \param[out] s Singular values.
  \param[out] vt Tensor \f$ V^\dagger \f$ corresponds to right singular vectors.
  \param[in] target_rank The number of singular values to be calculated.

  \return Information from linear-algebra library.

  \attention Computational cost is the same as full SVD.
*/
template <template <typename> class Matrix, typename C>
int psvd(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
         Tensor<Matrix, C> &u, std::vector<double> &s, Tensor<Matrix, C> &vt,
         const size_t target_rank) {
  int info = svd(a, axes_row, axes_col, u, s, vt);
  if (s.size() > target_rank) {
    s.resize(target_rank);
    u = slice(u, axes_row.size(), 0, target_rank);
    vt = slice(vt, 0, 0, target_rank);
  }
  return info;
}

//! QR decomposition of matrix (rank-2 tensor)
/*!
  The sizes of \c q and \c r are reduced.
  When \c a is \f$ m\times n \f$ matrix and \f$ k=\min\{m,n\} \f$,
  the dimensions of \c q and \c r is \f$ m\times k \f$ and
  \f$ k\times n \f$.

  \param[in] a A matrix (rank-2 tensor) to be decomposed
  \param[out] q Orthogonal matrix
  \param[out] r Upper triangular matrix

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int qr(const Tensor<Matrix, C> &a, Tensor<Matrix, C> &q, Tensor<Matrix, C> &r) {
  assert(a.rank() == 2);
  int info;
  info = qr(a, Axes(0), Axes(1), q, r);
  return info;
}

//! QR decomposition of tensor
/*! This may be useful for creation of an isometry.
  For example,
  \code
  qr(A, Axes(0,3), Axes(1,2), Q, R)
  \endcode
  calculates the following decomposition,
  \f[
  A_{abcd} = \sum_i Q_{adi} R_{ibc},
  \f]
  where the tensor Q satisfies
  \f[
  \sum_{ab} Q_{abi} Q_{abj}^{*} = \delta_{ij},
  \f]
  and the tensor R corresponds to the upper triangular matrix.

  \param[in] a A tensor to be decomposed.
  \param[in] axes_row Axes for the orthogonal matrix.
  \param[in] axes_col Axes for the upper triangular matrix.
  \param[out] q Decomposed tensor corresponds to orthogonal matrix
  \param[out] r Decomposed tensor corresponds to upper triangular matrix

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int qr(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
       Tensor<Matrix, C> &q, Tensor<Matrix, C> &r) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);
  assert(debug::check_svd_axes(axes_row, axes_col, a.rank()));

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();
  const Shape shape_a = a.shape();

  Axes axes = axes_row + axes_col;
  Shape shape;
  shape.resize(rank);
  for (size_t i = 0; i < rank; ++i) shape[i] = shape_a[axes[i]];

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;

  // rank-2 tensors (matrices)
  Tensor<Matrix, C> mat_q =
      reshape(transpose(a, axes, urank), Shape(d_row, d_col));
  Tensor<Matrix, C> mat_r(mat_q.get_comm(), mat_q.shape(), 1);

  // QR decomposition. Elements of mat_q change from a to q.
  int info;
  info = matrix_qr(mat_q.get_matrix(), mat_r.get_matrix());

  // Get shape of q and r.
  Shape shape_q;
  shape_q.resize(urank + 1);
  for (size_t i = 0; i < urank; ++i) shape_q[i] = shape[i];
  shape_q[urank] = size;

  Shape shape_r;
  shape_r.resize(rank - urank + 1);
  shape_r[0] = size;
  for (size_t i = urank; i < rank; ++i) shape_r[i - urank + 1] = shape[i];

  // Reshape q and r.
  if (d_row > d_col) {
    q = reshape(mat_q, shape_q);
    r = reshape(slice(mat_r, 0, 0, size), shape_r);
  } else if (d_row < d_col) {
    q = reshape(slice(mat_q, 1, 0, size), shape_q);
    r = reshape(mat_r, shape_r);
  } else {  // d_row==d_col
    q = reshape(mat_q, shape_q);
    r = reshape(mat_r, shape_r);
  }

  return info;
}

//! Compute the eigenvalues and eigenvectors of a complex Hermitian or real
//! symmetric matrix (rank-2 tensor)
/*!
  \param[in] a Rank-2 tensor
  \param[out] w Eigenvalues
  \param[out] z Eigenvectors

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int eigh(const Tensor<Matrix, C> &a, std::vector<double> &w,
         Tensor<Matrix, C> &z) {
  assert(a.rank() == 2);
  Shape shape = a.shape();
  assert(shape[0] == shape[1]);

  Tensor<Matrix, C> a_t = transpose(a, Axes(0, 1), 1);

  size_t n = shape[0];
  w.resize(n);
  z = Tensor<Matrix, C>(a.get_comm(), Shape(n, n), 1);

  int info;
  info = matrix_eigh(a_t.get_matrix(), w, z.get_matrix());
  return info;
};

//! Compute only the eigenvalues of a complex Hermitian or real symmetric matrix
//! (rank-2 tensor)
/*!
  \param[in] a Rank-2 tensor
  \param[out] w Eigenvalues

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int eigh(const Tensor<Matrix, C> &a, std::vector<double> &w) {
  assert(a.rank() == 2);
  Shape shape = a.shape();
  assert(shape[0] == shape[1]);

  Tensor<Matrix, C> a_t = transpose(a, Axes(0, 1), 1);

  size_t n = shape[0];
  w.resize(n);

  int info;
  info = matrix_eigh(a_t.get_matrix(), w);
  return info;
};

//! Compute the eigenvalues and eigenvectors of a complex Hermitian or real
//! symmetric tensor
/*!
  For example,
  \code
  eigh(A, Axes(0,3), Axes(1,2), w, U)
  \endcode
  calculates the following decomposition.
  \f[
  A_{abcd} = A_{(ad)(bc)} = \sum_i U_{adi} w_i (U^\dagger)_{ibc}.
  \f]

  \param[in] a A tensor
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] w Eigenvalues
  \param[out] z Tensor corresponds to Eigenvectors

  \return Information from linear-algebra library.

  \warning Input tensor should be Hermite.
*/
template <template <typename> class Matrix, typename C>
int eigh(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
         std::vector<double> &w, Tensor<Matrix, C> &z) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();

  Axes axes = axes_row + axes_col;

  Tensor<Matrix, C> a_t = transpose(a, axes, urank);
  const Shape &shape = a_t.shape();

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;

  assert(d_row == d_col);

  Shape shape_z;
  shape_z.resize(urank + 1);
  for (size_t i = 0; i < urank; ++i) shape_z[i] = shape[i];
  shape_z[urank] = size;

  z = Tensor<Matrix, C>(a.get_comm(), shape_z, urank);
  w.resize(size);

  int info;
  info = matrix_eigh(a_t.get_matrix(), w, z.get_matrix());
  return info;
};

//! Compute the eigenvalues of a complex Hermitian or real symmetric tensor
/*!
  \param[in] a A tensor
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] w Eigenvalues

  \return Information from linear-algebra library.

  \warning Input tensor should be Hermite.
*/
template <template <typename> class Matrix, typename C>
int eigh(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
         std::vector<double> &w) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();

  Axes axes = axes_row + axes_col;

  Tensor<Matrix, C> a_t = transpose(a, axes, urank);
  const Shape &shape = a_t.shape();

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;
  w.resize(size);

  assert(d_row == d_col);

  int info;
  info = matrix_eigh(a_t.get_matrix(), w);
  return info;
};

//! Solve a generalized eigenvalue problem \f$ A \vec{v} = \lambda B \vec{v} \f$
//! for a complex Hermitian or real symmetric tensor
/*!
  \param[in] a A complex Hermitian or real symmetric tensor.
  \param[in] axes_row_a Axes of tensor \c a to be row after matricization.
  \param[in] axes_col_a Axes of tensor \c a to be column after matricization.
  \param[in] b A complex Hermitian or real symmetric definitive positive tensor.
  \param[in] axes_row_b Axes of tensor \c b to be row after matricization.
  \param[in] axes_col_b Axes of tensor \c b to be column after matricization.
  \param[out] w Eigenvalues
  \param[out] z Tensor corresponds to Eigenvectors

  \return Information from linear-algebra library.

  For example,
  \code
  eigh(A, Axes(0,3), Axes(1,2), B, Axes(1, 3), Axes(2, 0), w, U)
  \endcode
  solve a generalized eigenvalue problem
  \f[
  \sum_{bc} A_{abcd} U_{bck} = \sum_{bc} B_{cabd} U_{bck} w_k.
  \f]
  In the matricized representation, it is written as
  \f[
  \sum_{bc} A'_{(ad)(bc)} U'_{(bc)(k)} = \sum_{bc} B_{(ad)(bc)} U'_{(bc)(k)}
  w_k, \f] where \f$ A \f$ and \f$ B \f$ are matricized as \f$ A_{abcd} =
  A'_{(ad)(bc)} \f$ and \f$ B_{cabd} = B'_{(ad)(bc)} \f$.

  \warning The matricies obtained by \c a and \c b should be square and have the
  same size.

  \note The shape of \c z is determined by \c axes_col_a.
*/
template <template <typename> class Matrix, typename C>
int eigh(const Tensor<Matrix, C> &a, const Axes &axes_row_a,
         const Axes &axes_col_a, const Tensor<Matrix, C> &b,
         const Axes &axes_row_b, const Axes &axes_col_b, std::vector<double> &w,
         Tensor<Matrix, C> &z) {
  assert(axes_row_a.size() > 0);
  assert(axes_col_a.size() > 0);
  assert(axes_row_b.size() > 0);
  assert(axes_col_b.size() > 0);

  const size_t rank_a = a.rank();
  const size_t rank_b = b.rank();
  const size_t urank_a = axes_row_a.size();
  const size_t urank_b = axes_row_b.size();
  Axes axes_a = axes_row_a + axes_col_a;
  Axes axes_b = axes_row_b + axes_col_b;
  Tensor<Matrix, C> a_t = transpose(a, axes_a, urank_a);
  Tensor<Matrix, C> b_t = transpose(b, axes_b, urank_b);
  const Shape &shape_a = a_t.shape();
  const Shape &shape_b = b_t.shape();
  size_t d_row_a(1), d_col_a(1);
  size_t d_row_b(1), d_col_b(1);
  for (size_t i = 0; i < urank_a; ++i) d_row_a *= shape_a[i];
  for (size_t i = 0; i < urank_b; ++i) d_row_b *= shape_b[i];
  for (size_t i = urank_a; i < rank_a; ++i) d_col_a *= shape_a[i];
  for (size_t i = urank_b; i < rank_b; ++i) d_col_b *= shape_b[i];

  assert(d_row_a == d_col_a);
  assert(d_row_b == d_col_b);
  assert(d_row_a == d_row_b);
  size_t size = d_row_a;

  Shape shape_z;
  shape_z.resize(rank_a - urank_a + 1);
  for (size_t i = urank_a; i < rank_a; ++i) shape_z[i - urank_a] = shape_a[i];
  shape_z[rank_a - urank_a] = size;

  z = Tensor<Matrix, C>(a.get_comm(), shape_z, urank_a);
  w.resize(size);

  int info;
  info = matrix_eigh(a_t.get_matrix(), b_t.get_matrix(), w, z.get_matrix());

  return info;
};

//! Compute the eigenvalues and eigenvectors of a general matrix (rank-2 tensor)
/*!
  \param[in] a Rank-2 tensor
  \param[out] w Eigenvalues
  \param[out] z Eigenvectors

  \return Information from linear-algebra library.
  \warning This function may require huge memory because it uses LAPACK routines.
*/
template <template <typename> class Matrix, typename C>
int eig(const Tensor<Matrix, C> &a, std::vector<complex> &w,
        Tensor<Matrix, complex> &z) {
  assert(a.rank() == 2);
  Shape shape = a.shape();
  assert(shape[0] == shape[1]);

  Tensor<lapack::Matrix, C> a_t = transpose(a, Axes(0, 1), 1).gather();

  size_t n = shape[0];
  w.resize(n);

  Tensor<lapack::Matrix, complex> z_t(a.get_comm(), Shape(n, n), 1);

  int info;
  info = matrix_eig(a_t.get_matrix(), w, z_t.get_matrix());

  z = Tensor<Matrix, complex>(a.get_comm(), z_t);

  return info;
};

//! Compute only the eigenvalues of a general matrix (rank-2 tensor)
/*!
  \param[in] a Rank-2 tensor
  \param[out] w Eigenvalues

  \return Information from linear-algebra library.
  \warning This function may require huge memory because it uses LAPACK routines.
*/
template <template <typename> class Matrix, typename C>
int eig(const Tensor<Matrix, C> &a, std::vector<complex> &w) {
  assert(a.rank() == 2);
  Shape shape = a.shape();
  assert(shape[0] == shape[1]);

  Tensor<lapack::Matrix, C> a_t = transpose(a, Axes(0, 1), 1).gather();

  size_t n = shape[0];
  w.resize(n);

  int info;
  info = matrix_eig(a_t.get_matrix(), w);
  return info;
};

//! Compute the eigenvalues and eigenvectors of a general tensor
/*!
  For example,
  \code
  eig(A, Axes(0,3), Axes(1,2), w, U)
  \endcode
  calculates the following decomposition.
  \f[
  A_{abcd} = A_{(ad)(bc)} = \sum_i U_{adi} w_i (U^\dagger)_{ibc}.
  \f]

  \param[in] a A tensor
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] w Eigenvalues
  \param[out] z Tensor corresponds to Eigenvectors

  \return Information from linear-algebra library.

  \warning This function may require huge memory because it uses LAPACK routines.
*/
template <template <typename> class Matrix, typename C>
int eig(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
        std::vector<complex> &w, Tensor<Matrix, complex> &z) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();

  Axes axes = axes_row + axes_col;

  Tensor<lapack::Matrix, C> a_t = transpose(a, axes, urank).gather();
  const Shape &shape = a_t.shape();

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;

  assert(d_row == d_col);

  Shape shape_z;
  shape_z.resize(urank + 1);
  for (size_t i = 0; i < urank; ++i) shape_z[i] = shape[i];
  shape_z[urank] = size;

  Tensor<lapack::Matrix, complex> z_t(a.get_comm(), shape_z, urank);

  w.resize(size);

  int info;
  info = matrix_eig(a_t.get_matrix(), w, z_t.get_matrix());

  z = Tensor<Matrix, complex>(a.get_comm(), z_t);

  return info;
};

//! Compute the eigenvalues of a general tensor
/*!
  \param[in] a A tensor
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] w Eigenvalues

  \return Information from linear-algebra library.

  \warning This function may require huge memory because it uses LAPACK routines.
*/
template <template <typename> class Matrix, typename C>
int eig(const Tensor<Matrix, C> &a, const Axes &axes_row, const Axes &axes_col,
        std::vector<complex> &w) {
  assert(axes_row.size() > 0);
  assert(axes_col.size() > 0);

  const size_t rank = a.rank();
  const size_t urank = axes_row.size();

  Axes axes = axes_row + axes_col;

  Tensor<lapack::Matrix, C> a_t = transpose(a, axes, urank).gather();
  const Shape &shape = a_t.shape();

  size_t d_row(1), d_col(1);
  for (size_t i = 0; i < urank; ++i) d_row *= shape[i];
  for (size_t i = urank; i < rank; ++i) d_col *= shape[i];
  size_t size = (d_row < d_col) ? d_row : d_col;
  w.resize(size);

  assert(d_row == d_col);

  int info;
  info = matrix_eig(a_t.get_matrix(), w);
  return info;
};


//! Solve linear equation \f$ A\vec{x}=\vec{b}\f$.
/*!
  \param[in] a The coefficient square matrix A.
  \param[in] b (global) The right hand side vector \f$ \vec{b}\f$.
  \param[out] x (global) The solution \f$ \vec{x}\f$.

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int solve(const Tensor<Matrix, C> &a, const std::vector<C> &b,
          std::vector<C> &x) {
  assert(a.rank() == 2);
  assert(a.shape()[0] == a.shape()[1]);
  assert(a.shape()[0] == b.size());

  Tensor<Matrix, C> x_t;
  int info;
  info = solve(a, Tensor<Matrix, C>(a.get_comm(), b), x_t, Axes(0), Axes(1),
               Axes(0), Axes());
  x = x_t.flatten();

  return info;
};

//! Solve linear equation \f$ AX=B\f$.
/*!
  \param[in] a The coefficient matrix A. a.shape() = (M,M)
  \param[in] b The right hand side matrix or vector B. b.shape() = (M,N) or (M)
  \param[out] x The solution X. Returned shape is identical to b.

  \return Information from linear-algebra library.
*/
template <template <typename> class Matrix, typename C>
int solve(const Tensor<Matrix, C> &a, const Tensor<Matrix, C> &b,
          Tensor<Matrix, C> &x) {
  assert(a.rank() == 2);
  assert(b.rank() == 2 || b.rank() == 1);
  assert(a.shape()[0] == a.shape()[1]);
  assert(a.shape()[0] == b.shape()[0]);
  int info;

  if (b.rank() == 2) {
    info = solve(a, b, x, Axes(0), Axes(1), Axes(0), Axes(1));
  } else {
    info = solve(a, b, x, Axes(0), Axes(1), Axes(0), Axes());
  }

  return info;
};

//! Solve linear equation \f$ AX=B\f$.
/*!
  \param[in] a The coefficient tensor A, which is a square matrix after
  matricization. \param[in] b The right hand side tensor B. \param[out] x The
  solution X. \param[in] axes_row_a Axes in A to reorder to the left. \param[in]
  axes_col_a Axes in A to reorder to the right. \param[in] axes_row_b Axes in B
  to reorder to the left. \param[in] axes_col_b Axes in B to reorder to the
  right.

  \return Information from linear-algebra library.

  The last four arguments define the way of matricization of tensors, A and B.
  The order of axes in solution X is deterimined by \c axes_col_a and \c
  axes_col_b.

  An equation \f$ \sum_{ik} A_{ijkl}X_{ikm} = B_{mlj} \f$ can be solved by
  \code
  solve(A,B,X,Axes(1,3),Axes(0,2),Axes(2,1),Axes(0)),
  \endcode
  where the equation after reordering is written as
  \f$\sum_{ik} A'_{(jl),(ik)}X_{(ik),(m)} = B'_{(jl),(m)} \f$.

  The length of \c axes_col_b may be zero. For example,
  \code
  solve(A,B,X,Axes(0,1),Axes(2,3),Axes(0,1),Axes()),
  \endcode
  solves \f$ \sum_{kl} A_{ijkl} X_{kl} = B_{ij} \f$.
*/
template <template <typename> class Matrix, typename C>
int solve(const Tensor<Matrix, C> &a, const Tensor<Matrix, C> &b,
          Tensor<Matrix, C> &x, const Axes &axes_row_a, const Axes &axes_col_a,
          const Axes &axes_row_b, const Axes &axes_col_b) {
  assert(a.rank() == axes_row_a.size() + axes_col_a.size());
  assert(b.rank() == axes_row_b.size() + axes_col_b.size());
  assert(axes_row_a.size() > 0);
  assert(axes_col_a.size() > 0);
  assert(axes_row_b.size() > 0);
  assert(axes_col_b.size() >= 0);

  // size_t rank_a = a.rank();
  // size_t rank_b = b.rank();
  size_t rank_row_a = axes_row_a.size();
  size_t rank_col_a = axes_col_a.size();
  size_t rank_row_b = axes_row_b.size();
  size_t rank_col_b = axes_col_b.size();
  Axes axes_a = axes_row_a + axes_col_a;
  Axes axes_b = axes_row_b + axes_col_b;

  Tensor<Matrix, C> a_t = transpose(a, axes_a, rank_row_a);
  Tensor<Matrix, C> b_t = transpose(b, axes_b, rank_row_b);
  const Shape &shape_a = a_t.shape();
  const Shape &shape_b = b_t.shape();

  assert(debug::check_square(shape_a, rank_row_a));

  Shape shape_x;
  shape_x.resize(rank_col_a + rank_col_b);
  for (size_t i = 0; i < rank_col_a; ++i) shape_x[i] = shape_a[i + rank_row_a];
  for (size_t i = 0; i < rank_col_b; ++i)
    shape_x[i + rank_col_a] = shape_b[i + rank_row_b];

  int info;
  info = matrix_solve(a_t.get_matrix(), b_t.get_matrix());
  x = reshape(b_t, shape_x);
  return info;
};

//! Unary plus
template <template <typename> class Matrix, typename C>
inline Tensor<Matrix, C> operator+(Tensor<Matrix, C> rhs) {
  return rhs;
}
//! Unary minus
template <template <typename> class Matrix, typename C>
inline Tensor<Matrix, C> operator-(Tensor<Matrix, C> rhs) {
  return (rhs *= -1.0);
}
//! Addition
template <template <typename> class Matrix, typename C>
inline Tensor<Matrix, C> operator+(Tensor<Matrix, C> lhs,
                                   const Tensor<Matrix, C> &rhs) {
  return (lhs += rhs);
}
//! Subtraction
template <template <typename> class Matrix, typename C>
inline Tensor<Matrix, C> operator-(Tensor<Matrix, C> lhs,
                                   const Tensor<Matrix, C> &rhs) {
  return (lhs -= rhs);
}
//! Tensor-scalar multiplication
template <template <typename> class Matrix, typename C, typename D>
inline Tensor<Matrix, C> operator*(Tensor<Matrix, C> lhs, D rhs) {
  return (lhs *= rhs);
}
//! Scalar division
template <template <typename> class Matrix, typename C, typename D>
inline Tensor<Matrix, C> operator/(Tensor<Matrix, C> lhs, D rhs) {
  return (lhs /= rhs);
}
//! Scalar-tensor multiplication
template <template <typename> class Matrix, typename C, typename D>
inline Tensor<Matrix, C> operator*(D lhs, Tensor<Matrix, C> rhs) {
  return (rhs *= lhs);
}

//! Return the maximum element. For complex, same as max_abs().
template <template <typename> class Matrix, typename C>
inline double max(const Tensor<Matrix, C> &t) {
  return max(t.get_matrix());
}
//! Return the minimum element. For complex, same as min_abs().
template <template <typename> class Matrix, typename C>
inline double min(const Tensor<Matrix, C> &t) {
  return min(t.get_matrix());
}
//! Return maximum of the absolute value of an element.
template <template <typename> class Matrix, typename C>
inline double max_abs(const Tensor<Matrix, C> &t) {
  return max_abs(t.get_matrix());
}
//! Return minimum of the absolute value of an element.
template <template <typename> class Matrix, typename C>
inline double min_abs(const Tensor<Matrix, C> &t) {
  return min_abs(t.get_matrix());
}

//! \cond
template <template <typename> class Matrix>
inline Tensor<Matrix, double> sqrt(Tensor<Matrix, double> t) {
  return t.map(static_cast<double (*)(double)>(&std::sqrt));
}
template <template <typename> class Matrix>
inline Tensor<Matrix, complex> sqrt(Tensor<Matrix, complex> t) {
  return t.map(static_cast<complex (*)(const complex &)>(&std::sqrt));
}
template <template <typename> class Matrix>
inline Tensor<Matrix, double> conj(Tensor<Matrix, double> t) {
  return t;
}
template <template <typename> class Matrix>
inline Tensor<Matrix, complex> conj(Tensor<Matrix, complex> t) {
  return t.map(static_cast<complex (*)(const complex &)>(&std::conj));
}
//! \endcond

//! Output elements of tensor in numpy style
/*!
  \param[out] out Output stream.
  \param[in] t Tensor to be output.
*/
template <template <typename> class Matrix, typename C>
std::ostream &operator<<(std::ostream &out, const Tensor<Matrix, C> &t) {
  std::size_t dim = t.ndim();
  if (t.get_comm_size() == 1) {
    std::size_t size = t.local_size();
    Shape shape = t.shape();
    std::vector<std::size_t> accum(dim + 1, 1);
    for (size_t d = dim; d != 0; --d) accum[d - 1] = accum[d] * shape[d - 1];

    std::vector<std::size_t> idx(dim);
    for (size_t i = 0; i < size; ++i) {
      for (size_t d = 0; d < dim; ++d) idx[d] = (i % accum[d]) / accum[d + 1];
      if (idx[dim - 1] == 0) {
        size_t count = 0;
        for (size_t d = dim; d != 0 && idx[d - 1] == 0; --d) ++count;
        for (size_t i = 0; i < dim - count; ++i) out << ' ';
        for (size_t i = 0; i < count; ++i) out << '[';
      }

      C val;
      t.get_value(idx, val);
      out << ' ' << val << ' ';

      if (idx[dim - 1] == shape[dim - 1] - 1) {
        size_t count = 0;
        for (size_t d = dim; d != 0 && idx[d - 1] == shape[d - 1] - 1; --d) ++count;
        for (size_t i = 0; i < count; ++i) out << ']';
        if (count < dim) {
          for (size_t i = 0; i < count; ++i) out << '\n';
        } else {
          out << '\n';
        }
      }
    }
  } else {
    if (t.get_comm_rank() == 0) {
      for (size_t d = 0; d < dim; ++d) out << '[';
      out << " distributed tensor ... ";
      for (size_t d = 0; d < dim; ++d) out << ']';
      out << '\n';
    }
  }
  return out;
}

}  // namespace mptensor

#endif  // _TENSOR_IMPL_HPP_
