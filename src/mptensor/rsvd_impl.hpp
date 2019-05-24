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
  \file   rsvd_impl.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Feb 3 2016

  \brief  Implementation of RSVD.
*/

#ifndef _TENSOR_RSVD_IMPL_HPP_
#define _TENSOR_RSVD_IMPL_HPP_

#include <vector>
#include <cassert>

#include "complex.hpp"
#include "index.hpp"
#include "matrix.hpp"
#include "tensor.hpp"
#include "rsvd.hpp"

namespace mptensor {

namespace random_tensor {

template <typename C> C uniform_dist();
void set_seed(unsigned int seed);

template <typename tensor_t> void fill(tensor_t &t) {
  const size_t n = t.local_size();
  for(size_t i=0;i<n;++i) {
    t[i] = uniform_dist<typename tensor_t::value_type>();
  }
}

} // namespace random_tensor


//! Singular value decomposition by randomized algorithm
/*!
  For example,
  \code
  rsvd(A, Axes(0,3), Axes(1,2), U, S, VT, k, p)
  \endcode
  calculates the following decomposition.
  \f[
  A_{abcd} \simeq \sum_{i=0}^{k-1} U_{adi} S_i (V^\dagger)_{ibc},
  \f]
  where \f$ k \f$ is the target rank.

  \param[in] a A tensor to be decomposed.
  \param[in] axes_row Axes for left singular vectors.
  \param[in] axes_col Axes for right singular vectors.
  \param[out] u Tensor \f$ U \f$ corresponds to left signular vectors.
  \param[out] s Singluar values.
  \param[out] vt Tensor \f$ V^\dagger \f$ corresponds to right singular vectors.
  \param[in] target_rank The number of singular values to be calculated
  \param[in] oversamp Oversampling parameter for randomized algorithm
  \return  Information from linear-algebra library.
*/
template <template<typename> class Matrix, typename C>
int rsvd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank, const size_t oversamp) {
  assert( axes_row.size() > 0 );
  assert( axes_col.size() > 0 );
  assert( debug::check_svd_axes(axes_row, axes_col, a.rank()) );

  const size_t rank = a.rank();
  const size_t rank_row = axes_row.size();
  const size_t rank_col = axes_col.size();
  int info;

  Axes axes = axes_row + axes_col;
  Tensor<Matrix,C> a_t = transpose(a, axes, rank_row);
  const Shape &shape = a_t.shape();

  Tensor<Matrix,C> q;
  {
    Shape shape_omega;
    shape_omega.resize(rank_col+1);
    for(int i=0;i<rank_col;++i) shape_omega[i] = shape[i+rank_row];
    shape_omega[rank_col] = target_rank + oversamp;

    // Tensor<Matrix,C> omega = random_tensor<C>(a.get_comm(), shape_omega);
    Tensor<Matrix,C> omega(a.get_comm(), shape_omega, rank_col);
    random_tensor::fill(omega);

    Tensor<Matrix,C> r;
    qr(tensordot(a_t, omega, range(rank_row, rank), range(0,rank_col)),
       range(0, rank_row), Axes(rank_row), q, r);
  }

  info = svd(tensordot(conj(q), a_t, range(0, rank_row), range(0, rank_row)),
             Axes(0), range(1,rank_col+1), u, s, vt);
  s.resize(target_rank);
  u = slice(u, 1, 0, target_rank);
  vt = slice(vt, 0, 0, target_rank);
  u = tensordot(q, u, Axes(rank_row), Axes(0));
  return info;
};


//! RSVD with \c oversamp = \c target_rank.
template <template<typename> class Matrix, typename C>
int rsvd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank) {
  return rsvd(a,axes_row,axes_col,u,s,vt,target_rank,target_rank);
};


//! Singular value decomposition by randomized algorithm
/*!
  \param[in] multiply_row A function object which takes a Tensor
  with <tt>Shape(shape_col+[target_rank+oversamp])</tt> and returns a Tensor
  with <tt>Shape(shape_row+[target_rank+oversamp])</tt>.
  \param[in] multiply_col A function object which takes a Tensor
  with <tt>Shape([target_rank+oversamp]+shape_row)</tt> and returns a Tensor
  with <tt>Shape([target_rank+oversamp]+shape_col)</tt>.
  \param[in] shape_row Shape of row indices.
  \param[in] shape_col Shape of column indices.
  \param[out] u Tensor \f$ U \f$ corresponds to left singular vectors.
  Its shape is <tt>Shape(shape_row+[target_rank])</tt>.
  \param[out] s Singular values.
  \param[out] vt Tensor \f$ V^\dagger \f$ corresponds to right singular vectors.
  Its shape is <tt>Shape([target_rank]+shape_col)</tt>.
  \param[in] target_rank The number of singular values to be calculated
  \param[in] oversamp Oversampling parameter for randomized algorithm
  \return  Information from linear-algebra library.
*/
template <template<typename> class Matrix, typename C, typename Func1, typename Func2>
int rsvd(Func1 &multiply_row, Func2 &multiply_col,
         const Shape &shape_row, const Shape &shape_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank, const size_t oversamp) {
  const size_t rank_row = shape_row.size();
  const size_t rank_col = shape_col.size();

  int info;
  Tensor<Matrix,C> q;
  {
    Shape shape_omega = shape_col;
    shape_omega.resize(rank_col+1);
    shape_omega[rank_col] = target_rank + oversamp;

    Tensor<Matrix,C> omega(u.get_comm(), shape_omega, rank_col);
    random_tensor::fill(omega);

    Tensor<Matrix,C> r;
    qr(multiply_col(omega), range(0, rank_row), Axes(rank_row), q, r);
  }

  info = svd(multiply_row(conj(q)), Axes(0), range(1,rank_col+1), u, s, vt);
  s.resize(target_rank);
  u = slice(u, 1, 0, target_rank);
  vt = slice(vt, 0, 0, target_rank);
  u = tensordot(q, u, Axes(rank_row), Axes(0));

  return info;
}


//! RSVD with \c oversamp = \c target_rank.
template <template<typename> class Matrix, typename C, typename Func1, typename Func2>
int rsvd(Func1 &multiply_row, Func2 &multiply_col,
         const Shape &shape_row, const Shape &shape_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank) {
  return rsvd(multiply_row, multiply_col, shape_row, shape_col,
              u, s, vt, target_rank, target_rank);
};


//! Set seed for random number generator.
/*!
  \param[in] seed Value of seed.
  \warning Set different seed values for each MPI process.
*/
inline void set_seed(unsigned int seed) {
  random_tensor::set_seed(seed);
};


} // namespace mptensor

#endif // _TENSOR_RSVD_IMPL_HPP_
