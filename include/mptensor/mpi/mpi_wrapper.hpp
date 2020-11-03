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
  \file   mpi_wrapper.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jun 2 2015

  \brief  Wrapper functions of MPI communications

  This header file assumes the MPI environment.
*/

#ifndef _MPI_WRAPPER_HPP_
#define _MPI_WRAPPER_HPP_

#ifndef _NO_MPI

#include <mpi.h>

#include <vector>

namespace mptensor {

//! Wrappers of MPI library
namespace mpi_wrapper {

//! Template function for MPI Datatype.
template <typename C>
inline MPI_Datatype mpi_datatype();

template <>
inline MPI_Datatype mpi_datatype<char>() {
  return MPI_CHAR;
};
template <>
inline MPI_Datatype mpi_datatype<signed char>() {
  return MPI_SIGNED_CHAR;
};
template <>
inline MPI_Datatype mpi_datatype<unsigned char>() {
  return MPI_UNSIGNED_CHAR;
};
template <>
inline MPI_Datatype mpi_datatype<short>() {
  return MPI_SHORT;
};
template <>
inline MPI_Datatype mpi_datatype<unsigned short>() {
  return MPI_UNSIGNED_SHORT;
};
template <>
inline MPI_Datatype mpi_datatype<int>() {
  return MPI_INT;
};
template <>
inline MPI_Datatype mpi_datatype<unsigned int>() {
  return MPI_UNSIGNED;
};
template <>
inline MPI_Datatype mpi_datatype<long int>() {
  return MPI_LONG;
};
template <>
inline MPI_Datatype mpi_datatype<unsigned long int>() {
  return MPI_UNSIGNED_LONG;
};
template <>
inline MPI_Datatype mpi_datatype<long long int>() {
  return MPI_LONG_LONG;
};
template <>
inline MPI_Datatype mpi_datatype<unsigned long long int>() {
  return MPI_UNSIGNED_LONG_LONG;
};
template <>
inline MPI_Datatype mpi_datatype<double>() {
  return MPI_DOUBLE;
};
template <>
inline MPI_Datatype mpi_datatype<complex>() {
  return MPI_DOUBLE_COMPLEX;
};

//! Calculate a summation over MPI communicator.
/*!
  \param[in] val value to be summed.
  \param[in] comm MPI communicator.

  \return summation of val.
*/
template <typename C>
inline C allreduce_sum(C val, const MPI_Comm &comm) {
  C recv;
  MPI_Allreduce(&val, &recv, 1, mpi_datatype<C>(), MPI_SUM, comm);
  return recv;
};

//! Calculate a summation of each element of vector over MPI communicator.
/*!
  \param[in] vec vector to be summed.
  \param[in] comm MPI communicator.

  \return resulted vector.
*/
template <typename C>
inline std::vector<C> allreduce_vec(const std::vector<C> &vec,
                                    const MPI_Comm &comm) {
  size_t n = vec.size();
  std::vector<C> recv(n);
  MPI_Allreduce(const_cast<C *>(&(vec[0])), &(recv[0]), static_cast<int>(n),
                mpi_datatype<C>(), MPI_SUM, comm);
  return recv;
};

//! Wrapper of MPI_Allreduce
/*!
  \param[in] sendbuf send buffer
  \param[out] recvbuf receive buffer
  \param[in] count the number in send buffer
  \param[in] op Opreation
  \param[in] comm MPI Communicator
*/
template <typename C>
inline void allreduce(const C *sendbuf, C *recvbuf, int count, MPI_Op op,
                      const MPI_Comm &comm) {
  MPI_Allreduce(const_cast<C *>(sendbuf), recvbuf, count, mpi_datatype<C>(), op,
                comm);
};

//! Wrapper of MPI_Sendrecv
/*!
  \param[in] sendbuf send buffer
  \param[in] sendcount the number of elements to send
  \param[in] dest Rank of destination
  \param[in] sendtag Send tag
  \param[out] recvbuf receive buffer
  \param[in] recvcount the number of elements to receive
  \param[in] source Rank of source
  \param[in] recvtag Receive tag
  \param[in] comm MPI Communicator
*/
template <typename C>
inline void sendrecv(const C *sendbuf, int sendcount, int dest, int sendtag,
                     C *recvbuf, int recvcount, int source, int recvtag,
                     const MPI_Comm &comm) {
  MPI_Sendrecv(const_cast<C *>(sendbuf), sendcount, mpi_datatype<C>(), dest,
               sendtag, recvbuf, recvcount, mpi_datatype<C>(), source, recvtag,
               comm, MPI_STATUS_IGNORE);
};

//! Wrapper of MPI_Sendrecv for std::vector
/*!
  \param[in] send_vec send vector
  \param[in] dest Rank of destination
  \param[in] sendtag Send tag
  \param[out] recv_vec receive vector
  \param[in] source Rank of source
  \param[in] recvtag Receive tag
  \param[in] comm MPI Communicator
*/
template <typename C>
inline void sendrecv(const std::vector<C> &send_vec, int dest, int sendtag,
                     std::vector<C> &recv_vec, int source, int recvtag,
                     const MPI_Comm &comm) {
  MPI_Sendrecv(const_cast<C *>(&(send_vec[0])),
               static_cast<int>(send_vec.size()), mpi_datatype<C>(), dest,
               sendtag, &(recv_vec[0]), static_cast<int>(recv_vec.size()),
               mpi_datatype<C>(), source, recvtag, comm, MPI_STATUS_IGNORE);
};

//! Wrapper of MPI_Alltoall
/*!
  \param[in] sendbuf Starting address of send buffer.
  \param[in] sendcount The number of elements to send.
  \param[out] recvbuf Address of receive buffer.
  \param[in] recvcount The number of elements to receive.
  \param[in] comm Communicator over which data is to be exchanged.
*/
template <typename C>
inline void alltoall(const C *sendbuf, int sendcount, C *recvbuf, int recvcount,
                     const MPI_Comm &comm) {
  MPI_Alltoall(const_cast<C *>(sendbuf), sendcount, mpi_datatype<C>(), recvbuf,
               recvcount, mpi_datatype<C>(), comm);
};

//! Wrapper of MPI_Alltoallv
/*!
  \param[in] sendbuf Starting address of send buffer.
  \param[in] sendcounts Integer array, where entry i specifies the number of
  elements to send to rank i.
  \param[in] sdispls Integer array, where entry i specifies the displacement
  (offset from sendbuf, in units of sendtype) from which to send data to rank i.
  \param[out] recvbuf Address of receive buffer.
  \param[in] recvcounts Integer array, where entry j specifies the number of
  elements to receive from rank j.
  \param[in] rdispls Integer array, where entry j specifies the displacement
  (offset from recvbuf, in units of recvtype) to which data from rank j should
  be written.
  \param[in] comm Communicator over which data is to be exchanged.
*/
template <typename C>
inline void alltoallv(const C *sendbuf, const int *sendcounts,
                      const int *sdispls, C *recvbuf, const int *recvcounts,
                      const int *rdispls, const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<C *>(sendbuf), const_cast<int *>(sendcounts),
                const_cast<int *>(sdispls), mpi_datatype<C>(), recvbuf,
                const_cast<int *>(recvcounts), const_cast<int *>(rdispls),
                mpi_datatype<C>(), comm);
};

//! Wrapper of MPI_Bcast
/*!
  \param buffer Starting address of buffer.
  \param count Number of entries in buffer.
  \param root Rank of broadcast root.
  \param comm Communicator.
*/
template <typename C>
inline void bcast(C *buffer, int count, int root, const MPI_Comm &comm) {
  MPI_Bcast(buffer, count, mpi_datatype<C>(), root, comm);
};

}  // namespace mpi_wrapper
}  // namespace mptensor

#endif  // _NO_MPI
#endif  // _MPI_WRAPPER_HPP_
