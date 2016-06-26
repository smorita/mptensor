/*
  Jun. 2, 2015
  Copyright (C) 2015 Satoshi Morita
*/

#ifndef _NO_MPI
#include <mpi.h>
#include "complex.hpp"
#include "mpi_wrapper.hpp"

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
void send_recv_vector(const std::vector<double> &send_vec, int dest, int sendtag,
                      std::vector<double> &recv_vec, int source, int recvtag,
                      const MPI_Comm &comm, MPI_Status &status) {
  MPI_Sendrecv( const_cast<double*>(&(send_vec[0])),
                static_cast<int>(send_vec.size()),
                MPI_DOUBLE, dest, sendtag,
                &(recv_vec[0]),
                static_cast<int>(recv_vec.size()),
                MPI_DOUBLE, source, sendtag,
                comm, &status);
};

template <>
void send_recv_vector(const std::vector<complex> &send_vec, int dest, int sendtag,
                      std::vector<complex> &recv_vec, int source, int recvtag,
                      const MPI_Comm &comm, MPI_Status &status) {
  MPI_Sendrecv( const_cast<complex*>(&(send_vec[0])),
                static_cast<int>(send_vec.size()),
                MPI_DOUBLE_COMPLEX, dest, sendtag,
                &(recv_vec[0]),
                static_cast<int>(recv_vec.size()),
                MPI_DOUBLE_COMPLEX, source, sendtag,
                comm, &status);
};

template <>
void alltoallv(const int *sendbuf, const int *sendcounts, const int *sdispls,
               int *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<int*>(sendbuf),
                const_cast<int*>(sendcounts),
                const_cast<int*>(sdispls), MPI_INT,
                recvbuf,
                const_cast<int*>(recvcounts),
                const_cast<int*>(rdispls), MPI_INT,
                comm);
};

template <>
void alltoallv(const unsigned long int *sendbuf, const int *sendcounts, const int *sdispls,
               unsigned long int *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<unsigned long int*>(sendbuf),
                const_cast<int*>(sendcounts),
                const_cast<int*>(sdispls), MPI_UNSIGNED_LONG,
                recvbuf,
                const_cast<int*>(recvcounts),
                const_cast<int*>(rdispls), MPI_UNSIGNED_LONG,
                comm);
};

template <>
void alltoallv(const double *sendbuf, const int *sendcounts, const int *sdispls,
               double *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<double*>(sendbuf),
                const_cast<int*>(sendcounts),
                const_cast<int*>(sdispls), MPI_DOUBLE,
                recvbuf,
                const_cast<int*>(recvcounts),
                const_cast<int*>(rdispls), MPI_DOUBLE,
                comm);
};

template <>
void alltoallv(const complex *sendbuf, const int *sendcounts, const int *sdispls,
               complex *recvbuf, const int *recvcounts, const int *rdispls,
               const MPI_Comm &comm) {
  MPI_Alltoallv(const_cast<complex*>(sendbuf),
                const_cast<int*>(sendcounts),
                const_cast<int*>(sdispls), MPI_DOUBLE_COMPLEX,
                recvbuf,
                const_cast<int*>(recvcounts),
                const_cast<int*>(rdispls), MPI_DOUBLE_COMPLEX,
                comm);
};


//! @endcond
}
}

#endif // _NO_MPI
