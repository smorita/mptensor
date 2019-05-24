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
  \file   matrix_interface_doc.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>

  \brief  Documentation of Matrix class interface.
*/

#ifndef _MATRIX_INTERFACE_DOC_HPP_
#define _MATRIX_INTERFACE_DOC_HPP_

namespace mptensor {

//! Nameclass for Matrix inteface.
/*!
  The list of member functions and non-member functions,
  which should be implemented in a Matrix class.
 */
namespace matrix_interface {


//! Inteface of Matrix class.
/*!
  This class shows the list of members, which should be implemented in a Matrix class.

  \warning
  A Matrix class is \e not necessary to inherint this class because we use duck typing.

  \tparam C type of elements (double or complex).
 */
template <typename C> class Matrix;


//! Constructor of a size-zero matrix.
/*!
  \note MPI communicator is set to MPI_COMM_WORLD or MPI_COMM_SELF.
*/
template <typename C> Matrix<C>::Matrix<C>();


//! Constructor of a size-zero matrix.
/*!
  \param[in] comm MPI communicator.
*/
template <typename C> explicit Matrix<C>::Matrix<C>(const comm_type& comm);


//! Constructor of a (n_row, n_col) matrix.
/*!
  \param[in] n_row size of row.
  \param[in] n_col size of column.
  \note MPI communicator is set to MPI_COMM_WORLD or MPI_COMM_SELF.
*/
template <typename C> Matrix<C>::Matrix<C>(size_t n_row, size_t n_col);


//! Constructor of a (n_row, n_col) matrix.
/*!
  \param[in] comm  MPI communicator.
  \param[in] n_row size of row.
  \param[in] n_col size of column.
*/
template <typename C> Matrix<C>::Matrix<C>(const comm_type& comm, size_t n_row, size_t n_col);


//! Const array subscript operator.
/*!
  \attention This function does not check validity of local index.
*/
template <typename C> const C& Matrix<C>::operator[](size_t i) const;


//! Array subscript operator.
/*!
  \attention This function does not check validity of local index.
*/
template <typename C> C& Matrix<C>::operator[](size_t i);


//! Return the number of elements in this process.
/*!
  \return Size of local storage.
*/
template <typename C> size_t Matrix<C>::local_size() const;


//! Return the MPI communicator.
/*!
  \return MPI communicator.
*/
template <typename C> const typename Matrix<C>::comm_type& Matrix<C>::get_comm() const;


//! Return the size of the MPI communicator.
/*!
  \return The size of the MPI communicator.
*/
template <typename C> int Matrix<C>::get_comm_size() const;


//! Return the MPI rank.
/*!
  \return the rank of process.
*/
template <typename C> int Matrix<C>::get_comm_rank() const;


//! Return the flattened vector.
/*!
  \return the flattened vector. (global)
*/
template <typename C> std::vector<C> Matrix<C>::flatten();

//! Wrapper of MPI_Barrier.
template <typename C> void Matrix<C>::barrier() const;

//! Return the summation of a scalar. Every processes returns the same value.
/*!
  \return The summation of val.
*/
template <typename C> C Matrix<C>::allreduce_sum(C val) const;


//! Preprocess for fast conversion from local index to global one.
template <typename C> void Matrix<C>::prep_local_to_global() const;


//! Preprocess for fast conversion from local index to global one.
template <typename C> void Matrix<C>::prep_global_to_local() const;


//! Eigenvalues of a hermite (symmetric) matrix.
/*!
  \param[in]  a The hermite or symmetric matrix. On exit, it may be destroyed.
  \param[out] s The eigenvalues in ascending order.

  \return information from the library.
  \relatesalso Matrix
*/
template <typename C> int matrix_eigh(Matrix<C>& a, std::vector<double>& s);


//! Solve linear equation \f$ AX=B\f$.
/*!
  \param[in] a The \f$ N\times N\f$ coefficient matrix A. On exit, it may be destroyed.
  \param[in,out]  b On entry, the \f$ N\times K\f$ right-hand side matrix B.
  On exit, the \f$ N\times K\f$ solution matrix X.

  \return information from the library.
*/
template <typename C> int matrix_solve(Matrix<C>& a, Matrix<C>& b);


//! Return the maximum element.
/*!
  For complex-valued matrix, this function is the same as max_abs();
  \param[in] a A matrix
  \return The maximum value in all elements. It's a global scalar.
*/
template <typename C> double max(const Matrix<C>& a);

//! Return the minimum element.
/*!
  For complex-valued matrix, this function is the same as min_abs();
  \param[in] a A matrix
  \return The minimum value in all elements. It's a global scalar.
*/
template <typename C> double min(const Matrix<C>& a);

//! Return the maximum of the absolute value of elements.
/*!
  \param[in] a A matrix
  \return The maximum of the absolute value of elements. It's a global scalar.
*/
template <typename C> double max_abs(const Matrix<C>& a);

//! Return the minimum of the absolute value of elements.
/*!
  \param[in] a A matrix
  \return The minimum of the absolute value of elements. It's a global scalar.
*/
template <typename C> double min_abs(const Matrix<C>& a);

} // namespace matrix_interface
} // namespace mptensor


#endif // _MATRIX_INTERFACE_DOC_HPP_
