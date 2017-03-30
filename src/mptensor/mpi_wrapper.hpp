/*
  Jun. 2, 2015
  Copyright (C) 2015 Satoshi Morita
*/

#ifndef _MPI_WRAPPER_HPP_
#define _MPI_WRAPPER_HPP_

#ifndef _NO_MPI

#include <vector>
#include <mpi.h>

namespace mptensor {

//! Wrappers of MPI library
namespace mpi_wrapper {


//! Calculate a summation over MPI communicator.
/*!
  \param[in] val value to be summed.
  \param[in] comm MPI communicator.

  \return summation of val.
*/
template <typename C> C allreduce_sum(C val, const MPI_Comm &comm);


//! Calculate a summation of each element of vector over MPI communicator.
/*!
  \param[in] vec vector to be summed.
  \param[in] comm MPI communicator.

  \return resulted vector.
*/
template <typename C> std::vector<C> allreduce_vec(const std::vector<C> &vec, const MPI_Comm &comm);


//! Wrapper of MPI_Sendrecv
/*!
  \param[in] send_vec send vector
  \param[in] dest Rank of destination
  \param[in] sendtag Send tag
  \param[out] recv_vec receive vector
  \param[in] source Rank of source
  \param[in] recvtag Receive tag
  \param[in] comm MPI Comunicator
  \param[out] status Status object
*/
template <typename C>
void send_recv_vector(const std::vector<C> &send_vec, int dest, int sendtag,
                      std::vector<C> &recv_vec, int source, int recvtag,
                      const MPI_Comm &comm, MPI_Status &status);


//! Wrapper of MPI_Alltoallv
/*!
  \param[in] sendbuf Starting address of send buffer.
  \param[in] sendcounts Integer array, where entry i specifies the number of elements to send to rank i.
  \param[in] sdispls Integer array, where entry i specifies the displacement (offset from sendbuf, in units of sendtype) from which to send data to rank i.
  \param[out] recvbuf Address of receive buffer.
  \param[in] recvcounts Integer array, where entry j specifies the number of elements to receive from rank j.
  \param[in] rdispls Integer array, where entry j specifies the displacement (offset from recvbuf, in units of recvtype) to which data from rank j should be written.
  \param[in] comm Communicator over which data is to be exchanged.
*/
template <typename C>
void alltoallv(const C *sendbuf, const int *sendcounts, const int *sdispls,
               C *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm);


//! Wrapper of MPI_Bcast
/*!
  \param buffer Starting address of buffer.
  \param count Number of entries in buffer.
  \param root Rank of broadcast root.
  \param comm Communicator.
*/
template <typename C>
void bcast(C *buffer, int count, int root, const MPI_Comm &comm);


} // namespace mpi_wrapper
} // namespace mptensor

#endif // _NO_MPI
#endif // _MPI_WRAPPER_HPP_
