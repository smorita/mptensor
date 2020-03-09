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
  \file   mpi_wrapper.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jun 2 2015

  \brief  Wrapper functions of MPI communications
*/

#ifndef _NO_MPI
#include <mpi.h>

#include "mptensor/complex.hpp"
#include "mptensor/mpi_wrapper.hpp"

namespace mptensor {
namespace mpi_wrapper {
//! @cond

template <>
double allreduce_sum(double val, const MPI_Comm &comm) {
  double recv;
  MPI_Allreduce(&val, &recv, 1, MPI_DOUBLE, MPI_SUM, comm);
  return recv;
}

template <>
complex allreduce_sum(complex val, const MPI_Comm &comm) {
  complex recv;
  MPI_Allreduce(&val, &recv, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
  return recv;
}

template <>
std::vector<double> allreduce_vec(const std::vector<double> &vec,
                                  const MPI_Comm &comm) {
  size_t n = vec.size();
  std::vector<double> recv(n);
  MPI_Allreduce(const_cast<double *>(&(vec[0])), &(recv[0]),
                static_cast<int>(n), MPI_DOUBLE, MPI_SUM, comm);
  return recv;
}

template <>
std::vector<complex> allreduce_vec(const std::vector<complex> &vec,
                                   const MPI_Comm &comm) {
  size_t n = vec.size();
  std::vector<complex> recv(n);
  MPI_Allreduce(const_cast<complex *>(&(vec[0])), &(recv[0]),
                static_cast<int>(n), MPI_DOUBLE_COMPLEX, MPI_SUM, comm);
  return recv;
}

template <>
void send_recv_vector(const std::vector<double> &send_vec, int dest,
                      int sendtag, std::vector<double> &recv_vec, int source,
                      int recvtag, const MPI_Comm &comm, MPI_Status &status) {
  MPI_Sendrecv(const_cast<double *>(&(send_vec[0])),
               static_cast<int>(send_vec.size()), MPI_DOUBLE, dest, sendtag,
               &(recv_vec[0]), static_cast<int>(recv_vec.size()), MPI_DOUBLE,
               source, sendtag, comm, &status);
};

template <>
void send_recv_vector(const std::vector<complex> &send_vec, int dest,
                      int sendtag, std::vector<complex> &recv_vec, int source,
                      int recvtag, const MPI_Comm &comm, MPI_Status &status) {
  MPI_Sendrecv(const_cast<complex *>(&(send_vec[0])),
               static_cast<int>(send_vec.size()), MPI_DOUBLE_COMPLEX, dest,
               sendtag, &(recv_vec[0]), static_cast<int>(recv_vec.size()),
               MPI_DOUBLE_COMPLEX, source, sendtag, comm, &status);
};

template <>
void alltoallv(const int *sendbuf, const int *sendcounts, const int *sdispls,
               int *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<int *>(sendbuf), const_cast<int *>(sendcounts),
                const_cast<int *>(sdispls), MPI_INT, recvbuf,
                const_cast<int *>(recvcounts), const_cast<int *>(rdispls),
                MPI_INT, comm);
};

template <>
void alltoallv(const unsigned long int *sendbuf, const int *sendcounts,
               const int *sdispls, unsigned long int *recvbuf,
               const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<unsigned long int *>(sendbuf),
                const_cast<int *>(sendcounts), const_cast<int *>(sdispls),
                MPI_UNSIGNED_LONG, recvbuf, const_cast<int *>(recvcounts),
                const_cast<int *>(rdispls), MPI_UNSIGNED_LONG, comm);
};

template <>
void alltoallv(const double *sendbuf, const int *sendcounts, const int *sdispls,
               double *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<double *>(sendbuf), const_cast<int *>(sendcounts),
                const_cast<int *>(sdispls), MPI_DOUBLE, recvbuf,
                const_cast<int *>(recvcounts), const_cast<int *>(rdispls),
                MPI_DOUBLE, comm);
};

template <>
void alltoallv(const complex *sendbuf, const int *sendcounts,
               const int *sdispls, complex *recvbuf, const int *recvcounts,
               const int *rdispls, const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<complex *>(sendbuf), const_cast<int *>(sendcounts),
                const_cast<int *>(sdispls), MPI_DOUBLE_COMPLEX, recvbuf,
                const_cast<int *>(recvcounts), const_cast<int *>(rdispls),
                MPI_DOUBLE_COMPLEX, comm);
};

template <>
void bcast(size_t *buffer, int count, int root, const MPI_Comm &comm) {
  MPI_Bcast(buffer, count, MPI_UNSIGNED_LONG, root, comm);
};

template <>
void bcast(double *buffer, int count, int root, const MPI_Comm &comm) {
  MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm);
};

template <>
void bcast(complex *buffer, int count, int root, const MPI_Comm &comm) {
  MPI_Bcast(buffer, count, MPI_DOUBLE_COMPLEX, root, comm);
};

//! @endcond
}  // namespace mpi_wrapper
}  // namespace mptensor

#endif  // _NO_MPI
