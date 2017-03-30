/*!
  \file   rsvd.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Feb 3 2016

  \brief  Randomized algorithm for singular value decomposition.
*/

#ifndef _TENSOR_RSVD_HPP_
#define _TENSOR_RSVD_HPP_

#include "tensor.hpp"

namespace mptensor {

template <template<typename> class Matrix, typename C>
int rsvd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank, const size_t oversamp);

template <template<typename> class Matrix, typename C>
int rsvd(const Tensor<Matrix,C> &a, const Axes &axes_row, const Axes &axes_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank);

template <template<typename> class Matrix, typename C, typename Func1, typename Func2>
int rsvd(Func1 &multiply_row, Func2 &multiply_col,
         const Shape &shape_row, const Shape &shape_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank, const size_t oversamp);

template <template<typename> class Matrix, typename C, typename Func1, typename Func2>
int rsvd(Func1 &multiply_row, Func2 &multiply_col,
         const Shape &shape_row, const Shape &shape_col,
         Tensor<Matrix,C> &u, std::vector<double> &s, Tensor<Matrix,C> &vt,
         const size_t target_rank);

inline void set_seed(unsigned int seed);

} // namespace mptensor

#include "rsvd_impl.hpp"
#endif // _TENSOR_RSVD_HPP_
