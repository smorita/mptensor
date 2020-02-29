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
  \file   matrix_scalapack_impl.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jan 29 2015
  \brief  Implementation of scalapack::Matrix class
*/

#ifndef _MATRIX_SCALAPACK_IMPL_HPP_
#define _MATRIX_SCALAPACK_IMPL_HPP_

#include <mpi.h>
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <complex>
#include <iostream>
#include <vector>
#include "../complex.hpp"
#include "../mpi_wrapper.hpp"
#include "blacsgrid.hpp"

/* ---------- PBLAS, SCALAPACK ---------- */
extern "C" {
int numroc_(int* M, int* MB, int* prow, int* irsrc, int* nprow);
void descinit_(int desca[], int* M, int* N, int* MB, int* NB, int* irsrc,
               int* icsrc, int* ictxt, int* lld, int* info);
}

namespace mptensor {

//! Namespace for the distributed Matrix class with ScaLAPACK, PBLAS, and BLACS.
namespace scalapack {

//! Distributed matrix using ScaLAPACK.
/*!
  \class Matrix
*/

/* ---------- static member variables ---------- */
template <typename C>
const size_t Matrix<C>::BLOCK_SIZE = 16;

/* ---------- constructors ---------- */

template <typename C>
Matrix<C>::Matrix() : grid(MPI_COMM_WORLD), desc(9) {
  init(0, 0);
};

template <typename C>
Matrix<C>::Matrix(const MPI_Comm& comm) : grid(comm), desc(9) {
  init(0, 0);
};

template <typename C>
Matrix<C>::Matrix(size_t n_row, size_t n_col) : grid(MPI_COMM_WORLD), desc(9) {
  init(n_row, n_col);
};

template <typename C>
Matrix<C>::Matrix(const MPI_Comm& comm, size_t n_row, size_t n_col)
    : grid(comm), desc(9) {
  init(n_row, n_col);
};

/* ---------- member functions ---------- */

template <typename C>
inline const C& Matrix<C>::operator[](size_t i) const {
  return V[i];
}

template <typename C>
inline C& Matrix<C>::operator[](size_t i) {
  return V[i];
}

template <typename C>
inline const C* Matrix<C>::head() const {
  return &(V[0]);
}

template <typename C>
inline C* Matrix<C>::head() {
  return &(V[0]);
}

template <typename C>
inline size_t Matrix<C>::local_size() const {
  return V.size();
}

// template <typename C> inline
// const BlacsGrid& Matrix<C>::get_grid() const { return grid; }

template <typename C>
inline const MPI_Comm& Matrix<C>::get_comm() const {
  return grid.comm;
}

template <typename C>
inline int Matrix<C>::get_comm_size() const {
  return grid.mpisize;
}

template <typename C>
inline int Matrix<C>::get_comm_rank() const {
  return grid.myrank;
}

template <typename C>
inline const int* Matrix<C>::descriptor() const {
  return &(desc[0]);
}

// template <typename C> inline
// int Matrix<C>::blacs_context() const { return desc[1]; }

template <typename C>
inline int Matrix<C>::n_row() const {
  return desc[2];
}

template <typename C>
inline int Matrix<C>::n_col() const {
  return desc[3];
}

// template <typename C> inline
// int Matrix<C>::nb_row() const { return desc[4]; }

// template <typename C> inline
// int Matrix<C>::nb_col() const { return desc[5]; }

template <typename C>
inline int Matrix<C>::get_lld() const {
  return desc[8];
}

template <typename C>
inline void Matrix<C>::global_index(size_t i, size_t& g_row,
                                    size_t& g_col) const {
  const int lld = get_lld();
  const size_t l_row = i % lld;
  const size_t l_col = i / lld;
  prep_local_to_global();
  g_row = global_row[l_row];
  g_col = global_col[l_col];
};

template <typename C>
inline bool Matrix<C>::local_index(size_t g_row, size_t g_col,
                                   size_t& i) const {
  const int lld = get_lld();
  const int nprow = grid.nprow;
  const int npcol = grid.npcol;
  const int myprow = grid.myprow;
  const int mypcol = grid.mypcol;

  size_t b_row = g_row / BLOCK_SIZE;
  size_t p_row = b_row % nprow;
  size_t b_col = g_col / BLOCK_SIZE;
  size_t p_col = b_col % npcol;
  if (p_row != size_t(myprow) || p_col != size_t(mypcol)) return false;

  size_t l_row = (b_row / nprow) * BLOCK_SIZE + g_row % BLOCK_SIZE;
  size_t l_col = (b_col / npcol) * BLOCK_SIZE + g_col % BLOCK_SIZE;
  i = l_row + l_col * lld;  // column major
  return true;
};

template <typename C>
inline void Matrix<C>::local_position(size_t g_row, size_t g_col,
                                      int& comm_rank, size_t& lindex) const {
  prep_global_to_local();
  const int myprow = proc_row[g_row];
  const int mypcol = proc_col[g_col];
  comm_rank = grid.mpirank(myprow, mypcol);
  lindex = local_row[g_row] + local_col[g_col] * lld_list[myprow];
}

template <typename C>
inline size_t Matrix<C>::local_row_size() const {
  return local_row_size_;
}

template <typename C>
inline size_t Matrix<C>::local_col_size() const {
  return local_col_size_;
}

template <typename C>
inline size_t Matrix<C>::local_row_index(size_t lindex) const {
  return lindex % (static_cast<size_t>(get_lld()));
}

template <typename C>
inline size_t Matrix<C>::local_col_index(size_t lindex) const {
  return lindex / (static_cast<size_t>(get_lld()));
}

template <typename C>
inline size_t Matrix<C>::global_row_index(size_t l_row) const {
  return BLOCK_SIZE * ((l_row / BLOCK_SIZE) * grid.nprow + grid.myprow) +
         (l_row % BLOCK_SIZE);
}

template <typename C>
inline size_t Matrix<C>::global_col_index(size_t l_col) const {
  return BLOCK_SIZE * ((l_col / BLOCK_SIZE) * grid.npcol + grid.mypcol) +
         (l_col % BLOCK_SIZE);
}

//! Preprocess for fast conversion from local index to global one.
template <typename C>
inline void Matrix<C>::prep_local_to_global() const {
  if (has_local_to_global) return;

  const int nprow = grid.nprow;
  const int npcol = grid.npcol;
  const int myprow = grid.myprow;
  const int mypcol = grid.mypcol;
  const size_t size_row = get_lld();
  const size_t size_col = local_size() / size_row;
  global_row.resize(size_row);
  global_col.resize(size_col);

#pragma omp parallel default(shared)
  {
#pragma omp for
    for (size_t l_row = 0; l_row < size_row; ++l_row) {
      global_row[l_row] = BLOCK_SIZE * ((l_row / BLOCK_SIZE) * nprow + myprow) +
                          (l_row % BLOCK_SIZE);
    }
#pragma omp for
    for (size_t l_col = 0; l_col < size_col; ++l_col) {
      global_col[l_col] = BLOCK_SIZE * ((l_col / BLOCK_SIZE) * npcol + mypcol) +
                          (l_col % BLOCK_SIZE);
    }
  }

  has_local_to_global = true;
}

//! Preprocess for fast conversion from global index to local one.
template <typename C>
inline void Matrix<C>::prep_global_to_local() const {
  if (has_global_to_local) return;

  const int nprow = grid.nprow;
  const int npcol = grid.npcol;
  const size_t size_row = n_row();
  const size_t size_col = n_col();
  const size_t nblocks = size_row / BLOCK_SIZE;
  const size_t extra_blocks = nblocks % nprow;
  const size_t locr_offset = (nblocks / nprow) * BLOCK_SIZE;

  local_row.resize(size_row);
  local_col.resize(size_col);
  proc_row.resize(size_row);
  proc_col.resize(size_col);
  lld_list.resize(nprow);

#pragma omp parallel default(shared)
  {
#pragma omp for
    for (size_t g_row = 0; g_row < size_row; ++g_row) {
      size_t b_row = g_row / BLOCK_SIZE;
      proc_row[g_row] = b_row % nprow;
      local_row[g_row] = (b_row / nprow) * BLOCK_SIZE + g_row % BLOCK_SIZE;
    }

#pragma omp for
    for (size_t g_col = 0; g_col < size_col; ++g_col) {
      size_t b_col = g_col / BLOCK_SIZE;
      proc_col[g_col] = b_col % npcol;
      local_col[g_col] = (b_col / npcol) * BLOCK_SIZE + g_col % BLOCK_SIZE;
    }

#pragma omp for
    for (size_t myprow = 0; myprow < size_t(nprow); ++myprow) {
      // inline expansion of numroc_()
      size_t locr = locr_offset;
      if (myprow < extra_blocks)
        locr += BLOCK_SIZE;
      else if (myprow == extra_blocks)
        locr += size_row % BLOCK_SIZE;

      lld_list[myprow] = (locr < 1) ? 1 : locr;
    }
  }

  has_global_to_local = true;
};

template <typename C>
void Matrix<C>::init(size_t n_row, size_t n_col) {
  int xM = n_row;
  int xN = n_col;
  int xMB = BLOCK_SIZE;
  int xNB = BLOCK_SIZE;
  int irsrc = 0, icsrc = 0;
  int locr = numroc_(&xM, &xMB, &(grid.myprow), &irsrc, &(grid.nprow));
  int locc = numroc_(&xN, &xNB, &(grid.mypcol), &icsrc, &(grid.npcol));
  int lld = (locr < 1) ? 1 : locr;
  int info;
  descinit_(&(desc[0]), &xM, &xN, &xMB, &xNB, &irsrc, &icsrc, &(grid.ictxt),
            &lld, &info);
  V.resize(locr * locc);
  local_row_size_ = static_cast<size_t>(locr);
  local_col_size_ = static_cast<size_t>(locc);
  has_local_to_global = false;
  has_global_to_local = false;
};

// template <typename C>
// void Matrix<C>::set(C (*element)(size_t gi, size_t gj)) {
//   const size_t& length = V.size();
//   for (size_t i=0;i<length;++i) {
//     size_t gi, gj;
//     global_index(i, gi, gj);
//     V[i] = element(gi, gj);
//   }
// };

template <typename C>
inline void Matrix<C>::print_info(std::ostream& out) const {
  out << "Matrix: prow= " << grid.myprow << " pcol= " << grid.mypcol
      << " local_size= " << local_size() << " lld(locr)= " << get_lld()
      << " locc= " << local_size() / get_lld() << "\n";
};

template <typename C>
inline Matrix<C>& Matrix<C>::operator+=(const Matrix<C>& rhs) {
  assert(V.size() == rhs.local_size());
  for (size_t i = 0; i < V.size(); ++i) V[i] += rhs[i];
  return *this;
};

template <typename C>
inline Matrix<C>& Matrix<C>::operator-=(const Matrix<C>& rhs) {
  assert(V.size() == rhs.local_size());
  for (size_t i = 0; i < V.size(); ++i) V[i] -= rhs[i];
  return *this;
};

template <typename C>
inline Matrix<C>& Matrix<C>::operator*=(C rhs) {
  for (size_t i = 0; i < V.size(); ++i) V[i] *= rhs;
  return *this;
};

template <typename C>
inline Matrix<C>& Matrix<C>::operator/=(C rhs) {
  for (size_t i = 0; i < V.size(); ++i) V[i] /= rhs;
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

  /* create lists of local position and destination rank */
  std::vector<int> dest_rank(local_size());
  std::vector<unsigned long int> local_position(local_size());

  for (size_t i = 0; i < local_size(); i++) {
    int dest;
    size_t pos, g_col, g_row;

    global_index(i, g_row, g_col);
    M_new.local_position(g_col, g_row, dest, pos);

    local_position[i] = pos;
    dest_rank[i] = dest;
  }

  /* exchange data */
  replace_matrix_data((*this), dest_rank, local_position, M_new);

  return M_new;
}

template <typename C>
inline std::vector<C> Matrix<C>::flatten() {
  const size_t n = n_row() * n_col();
  const size_t nr = n_row();
  std::vector<C> vec(n);
  size_t g_row, g_col;
  for (size_t i = 0; i < local_size(); ++i) {
    global_index(i, g_row, g_col);
    vec[g_row + g_col * nr] = V[i];
  }
  return mpi_wrapper::allreduce_vec(vec, get_comm());
};

template <typename C>
inline void Matrix<C>::barrier() const {
  MPI_Barrier(get_comm());
}

template <typename C>
inline C Matrix<C>::allreduce_sum(C value) const {
  return mpi_wrapper::allreduce_sum(value, get_comm());
}

template <typename C>
template <typename D>
inline void Matrix<C>::bcast(D* buffer, int count, int root) const {
  mpi_wrapper::bcast(buffer, count, root, get_comm());
}

/* ---------- non-member functions ---------- */

template <typename C>
void replace_matrix_data(const Matrix<C>& M, const std::vector<int>& dest_rank,
                         const std::vector<size_t>& local_position,
                         Matrix<C>& M_new) {
  const MPI_Comm comm = M.get_comm();
  const int mpisize = M.get_comm_size();
  // const int mpirank = M.get_comm_rank();
  const size_t local_size = M.local_size();

  const C* mat = M.head();
  C* mat_new = M_new.head();

  // assert(send_size_list.size() == mpisize);
  assert(dest_rank.size() == local_size);
  assert(local_position.size() == local_size);

  const int proc_size = mpisize + 1;
  int* send_counts = new int[proc_size];
  int* send_displs = new int[proc_size];
  int* recv_counts = new int[proc_size];
  int* recv_displs = new int[proc_size];
  for (int rank = 0; rank < proc_size; ++rank) {
    send_counts[rank] = 0;
    send_displs[rank] = 0;
    recv_counts[rank] = 0;
    recv_displs[rank] = 0;
  }

  for (size_t i = 0; i < local_size; ++i) {
    send_counts[dest_rank[i]] += 1;
  }

  MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);

  for (int rank = 0; rank < mpisize; ++rank) {
    send_displs[rank + 1] = send_counts[rank] + send_displs[rank];
    recv_displs[rank + 1] = recv_counts[rank] + recv_displs[rank];
  }

  const int send_size = local_size;  // send_displs[mpisize];
  const int recv_size = recv_displs[mpisize];
  unsigned long int* send_pos = new unsigned long int[send_size];
  unsigned long int* recv_pos = new unsigned long int[recv_size];
  C* send_value = new C[send_size];
  C* recv_value = new C[recv_size];

  /* Pack */
  // std::vector<int> pack_idx = send_displs;
  int* pack_idx = new int[proc_size];
  for (int rank = 0; rank < proc_size; ++rank) {
    pack_idx[rank] = send_displs[rank];
  }
  for (size_t i = 0; i < local_size; ++i) {
    const int rank = dest_rank[i];
    const int idx = pack_idx[rank];
    send_value[idx] = mat[i];
    send_pos[idx] = local_position[i];
    pack_idx[rank] += 1;
  }
  delete[] pack_idx;

  /* Send and Recieve */
  mpi_wrapper::alltoallv(send_pos, send_counts, send_displs, recv_pos,
                         recv_counts, recv_displs, comm);
  mpi_wrapper::alltoallv(send_value, send_counts, send_displs, recv_value,
                         recv_counts, recv_displs, comm);

  /* Unpack */
  for (int i = 0; i < recv_size; ++i) {
    mat_new[recv_pos[i]] = recv_value[i];
  }

  delete[] send_pos;
  delete[] send_value;
  delete[] recv_pos;
  delete[] recv_value;

  delete[] send_counts;
  delete[] send_displs;
  delete[] recv_counts;
  delete[] recv_displs;
}

template <typename C>
void sum_matrix_data(const Matrix<C>& M, const std::vector<int>& dest_rank,
                     const std::vector<size_t>& local_position,
                     Matrix<C>& M_new) {
  const MPI_Comm comm = M.get_comm();
  const int mpisize = M.get_comm_size();
  const int mpirank = M.get_comm_rank();
  const size_t local_size = M.local_size();

  // assert(send_size_list.size() == mpisize);
  assert(dest_rank.size() == local_size);
  assert(local_position.size() == local_size);

  std::vector<int> send_size_list(mpisize, 0);
  for (size_t i = 0; i < local_size; ++i) {
    const int rank = dest_rank[i];
    if (rank >= 0 && rank < mpisize) send_size_list[dest_rank[i]] += 1;
  }

  const C* mat = M.head();
  C* mat_new = M_new.head();

  for (int step = 0; step < mpisize; ++step) {
    int dest = (mpirank + step) % mpisize;
    int source = (mpirank + mpisize - step) % mpisize;
    if (dest == mpirank) {
      for (size_t i = 0; i < local_size; i++) {
        if (dest_rank[i] == mpirank) {
          mat_new[local_position[i]] += mat[i];
        }
      }
    } else {
      /* Get recv_size */
      int send_size = send_size_list[dest];
      int recv_size;
      MPI_Status status;
      int tag = step;
      MPI_Sendrecv(&send_size, 1, MPI_INT, dest, tag, &recv_size, 1, MPI_INT,
                   source, tag, comm, &status);

      /* Pack */
      std::vector<unsigned long int> send_pos(send_size);
      std::vector<C> send_value(send_size);
      size_t k = 0;
      for (size_t i = 0; i < local_size; i++) {
        if (dest_rank[i] == dest) {
          send_pos[k] = local_position[i];
          send_value[k] = mat[i];
          k += 1;
        }
      }

      /* Send and Recieve */
      std::vector<unsigned long int> recv_pos(recv_size);
      std::vector<C> recv_value(recv_size);
      MPI_Sendrecv(&(send_pos[0]), send_size, MPI_UNSIGNED_LONG, dest, tag,
                   &(recv_pos[0]), recv_size, MPI_UNSIGNED_LONG, source, tag,
                   comm, &status);
      mpi_wrapper::send_recv_vector(send_value, dest, tag, recv_value, source,
                                    tag, comm, status);

      /* Unpack */
      for (int i = 0; i < recv_size; ++i) {
        mat_new[recv_pos[i]] += recv_value[i];
      }
    }
  }
}

template <typename C>
double max_abs(const Matrix<C>& a) {
  const size_t n = a.local_size();
  double send = 0.0;
  double recv;
  for (size_t i = 0; i < n; ++i) {
    send = std::max(send, std::abs(a[i]));
  }
  MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE, MPI_MAX, a.get_comm());
  return recv;
};

template <typename C>
double min_abs(const Matrix<C>& a) {
  const size_t n = a.local_size();
  double send = DBL_MAX;
  double recv;
  for (size_t i = 0; i < n; ++i) {
    send = std::min(send, std::abs(a[i]));
  }
  MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE, MPI_MIN, a.get_comm());
  return recv;
};

}  // namespace scalapack
}  // namespace mptensor

#endif  // _MATRIX_SCALAPACK_IMPL_HPP_
