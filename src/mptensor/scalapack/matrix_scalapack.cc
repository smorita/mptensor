/*
  Dec. 12, 2014
  Copyright (C) 2014 Satoshi Morita
 */

#ifndef _NO_MPI
#include <cassert>
#include <cfloat>
#include <complex>
#include <algorithm>
#include "matrix_scalapack.hpp"

typedef std::complex<double> complex;

/* PBLAS, SCALAPACK */
extern "C" {
  int numroc_(int *M, int *MB, int *prow, int *irsrc, int *nprow);
  void descinit_(int desca[], int *M, int *N, int *MB, int *NB, int *irsrc,
                 int *icsrc, int *ictxt, int *lld, int *info);

  void pdgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
               double a[], int *ia, int *ja, int desca[], double b[], int *ib,
               int *jb, int descb[], double *beta, double c[], int *ic, int *jc,
               int descc[]);
  void pdgesvd_(char *jobu, char *jobvt, int *m, int *n,
                double a[], int *ia, int *ja, int desca[],
                double s[],
                double u[], int *iu, int *ju, int descu[],
                double vt[], int *ivt, int *jvt, int descvt[],
                double work[], int *lwork, int *info);
  void pdsyev_(char *jobz, char *uplo, int *n,
               double a[], int *ia, int *ja, int desca[], double w[],
               double z[], int *iz, int *jz, int descz[],
               double work[], int *lwork, int *info);
  void pdgeqrf_(int *m, int *n, double a[], int *ia, int *ja, int desca[],
                double tau[], double work[], int *lwork, int *info);
  void pdorgqr_(int *m, int *n, int *k, double a[], int *ia, int *ja, int desca[],
                double tau[], double work[], int *lwork, int *info);
  void pdgesv_(int *n, int *nrhs, double a[], int *ia, int *ja, int desca[],
               int ipiv[], double b[], int *ib, int *jb, int descb[], int *info);

  void pzgemm_(char *transa, char *transb, int *m, int *n, int *k, complex *alpha,
               complex a[], int *ia, int *ja, int desca[], complex b[], int *ib,
               int *jb, int descb[], complex *beta, complex c[], int *ic, int *jc,
               int descc[]);
  void pzgesvd_(char *jobu, char *jobvt, int *m, int *n,
                complex a[], int *ia, int *ja, int desca[],
                double s[],
                complex u[], int *iu, int *ju, int descu[],
                complex vt[], int *ivt, int *jvt, int descvt[],
                complex work[], int *lwork, double rwork[], int *info);
  void pzheev_(char *jobz, char *uplo, int *n,
               complex a[], int *ia, int *ja, int desca[], double w[],
               complex z[], int *iz, int *jz, int descz[],
               complex work[], int *lwork, double rwork[], int *lrwork, int *info);
  void pzgeqrf_(int *m, int *n, complex a[], int *ia, int *ja, int desca[],
                complex tau[], complex work[], int *lwork, int *info);
  void pzungqr_(int *m, int *n, int *k, complex a[], int *ia, int *ja, int desca[],
                complex tau[], complex work[], int *lwork, int *info);
  void pzgesv_(int *n, int *nrhs, complex a[], int *ia, int *ja, int desca[],
               int ipiv[], complex b[], int *ib, int *jb, int descb[], int *info);
}


namespace mptensor {
namespace scalapack {
//! \cond

template <>
void matrix_product(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
  assert( A.n_col()==B.n_row() );
  assert( A.n_row()==C.n_row() );
  assert( B.n_col()==C.n_col() );
  int M = A.n_row();
  int N = B.n_col();
  int K = A.n_col();

  char transa = 'N';
  char transb = 'N';
  double alpha = 1.0;
  double beta = 0.0;
  int ia, ja;
  int ib, jb;
  int ic, jc;
  ia = ib = ic = ja = jb = jc = 1;

  pdgemm_(&transa, &transb, &M, &N, &K, &alpha,
          const_cast<double*>(A.head()), &ia, &ja, const_cast<int*>(A.descriptor()),
          const_cast<double*>(B.head()), &ib, &jb, const_cast<int*>(B.descriptor()),
          &beta,
          C.head(), &ic, &jc, const_cast<int*>(C.descriptor()));

};

template <>
void matrix_product(const Matrix<complex>& A, const Matrix<complex>& B, Matrix<complex>& C) {
  assert( A.n_row()==C.n_row() );
  assert( A.n_col()==B.n_row() );
  assert( B.n_col()==C.n_col() );
  int M = A.n_row();
  int N = B.n_col();
  int K = A.n_col();

  char transa = 'N';
  char transb = 'N';
  complex alpha = 1.0;
  complex beta = 0.0;
  int ia, ja;
  int ib, jb;
  int ic, jc;
  ia = ib = ic = ja = jb = jc = 1;

  pzgemm_(&transa, &transb, &M, &N, &K, &alpha,
          const_cast<complex*>(A.head()), &ia, &ja, const_cast<int*>(A.descriptor()),
          const_cast<complex*>(B.head()), &ib, &jb, const_cast<int*>(B.descriptor()),
          &beta,
          C.head(), &ic, &jc, const_cast<int*>(C.descriptor()));

};

template <>
int matrix_svd(Matrix<double>& A, Matrix<double>& U,
               std::vector<double>& S, Matrix<double>& VT) {
  assert(A.n_row() == U.n_row());
  assert(A.n_col() == VT.n_col());
  size_t size = (A.n_row() < A.n_col()) ? A.n_row() : A.n_col();
  assert(U.n_col() == size);
  assert(S.size() == size);
  assert(VT.n_row() == size);

  int M = A.n_row();
  int N = A.n_col();

  char jobu = 'V';
  char jobvt = 'V';
  int ia, ja, iu, ju, ivt, jvt, lwork, info;
  std::vector<double> work;
  double work_size;
  ia = ja = iu = ju = ivt = jvt = 1;
  lwork = -1;

  /* Get the size of workspace */
  pdgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           U.head(), &iu, &ju, const_cast<int*>(U.descriptor()),
           VT.head(), &ivt, &jvt, const_cast<int*>(VT.descriptor()),
           &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  pdgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           U.head(), &iu, &ju, const_cast<int*>(U.descriptor()),
           VT.head(), &ivt, &jvt, const_cast<int*>(VT.descriptor()),
           &(work[0]), &lwork, &info);

  /* for debug */
  // std::cerr << "matrix_svd: M= " << M << " N= " << N
  //           << " SIZE= " << size << " lwork= " << lwork
  //           << " info = " << info << "\n";

  return info;
}

template <>
int matrix_svd(Matrix<complex>& A, Matrix<complex>& U,
               std::vector<double>& S, Matrix<complex>& VT) {
  assert(A.n_row() == U.n_row());
  assert(A.n_col() == VT.n_col());
  size_t size = (A.n_row() < A.n_col()) ? A.n_row() : A.n_col();
  assert(U.n_col() == size);
  assert(S.size() == size);
  assert(VT.n_row() == size);

  int M = A.n_row();
  int N = A.n_col();

  char jobu = 'V';
  char jobvt = 'V';
  int ia, ja, iu, ju, ivt, jvt, lwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  ia = ja = iu = ju = ivt = jvt = 1;
  lwork = -1;

  /* Get the size of workspace */
  pzgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           U.head(), &iu, &ju, const_cast<int*>(U.descriptor()),
           VT.head(), &ivt, &jvt, const_cast<int*>(VT.descriptor()),
           &work_size, &lwork, &rwork_size, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  rwork.resize(static_cast<int>(rwork_size));

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  pzgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           U.head(), &iu, &ju, const_cast<int*>(U.descriptor()),
           VT.head(), &ivt, &jvt, const_cast<int*>(VT.descriptor()),
           &(work[0]), &lwork, &(rwork[0]), &info);

  /* for debug */
  // std::cerr << "matrix_svd<complex>: M= " << M << " N= " << N
  //           << " SIZE= " << size << " lwork= " << lwork
  //           << " rwork_size= " << rwork.size()
  //           << " info = " << info << "\n";

  return info;
}


template <>
int matrix_svd(Matrix<double>& A, std::vector<double>& S) {
  size_t size = (A.n_row() < A.n_col()) ? A.n_row() : A.n_col();
  assert(S.size() == size);

  int M = A.n_row();
  int N = A.n_col();

  char jobu = 'N';
  char jobvt = 'N';
  int ia, ja, iu, ju, ivt, jvt, lwork, info;
  std::vector<double> work;
  double work_size;
  ia = ja = iu = ju = ivt = jvt = 1;
  lwork = -1;
  double d_dummy;
  int i_dummy;

  /* Get the size of workspace */
  pdgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           &(d_dummy), &iu, &ju, &(i_dummy),
           &(d_dummy), &ivt, &jvt, &(i_dummy),
           &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  pdgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           &(d_dummy), &iu, &ju, &(i_dummy),
           &(d_dummy), &ivt, &jvt, &(i_dummy),
           &(work[0]), &lwork, &info);

  /* for debug */
  // std::cerr << "matrix_svd: M= " << M << " N= " << N
  //           << " SIZE= " << size << " lwork= " << lwork
  //           << " info = " << info << "\n";

  return info;
}

template <>
int matrix_svd(Matrix<complex>& A, std::vector<double>& S) {
  size_t size = (A.n_row() < A.n_col()) ? A.n_row() : A.n_col();
  assert(S.size() == size);

  int M = A.n_row();
  int N = A.n_col();

  char jobu = 'N';
  char jobvt = 'N';
  int ia, ja, iu, ju, ivt, jvt, lwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  ia = ja = iu = ju = ivt = jvt = 1;
  lwork = -1;
  complex c_dummy;
  int i_dummy;

  /* Get the size of workspace */
  pzgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           &(c_dummy), &iu, &ju, &(i_dummy),
           &(c_dummy), &ivt, &jvt, &(i_dummy),
           &work_size, &lwork, &rwork_size, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  rwork.resize(static_cast<int>(rwork_size));

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  pzgesvd_(&jobu, &jobvt, &M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(S[0]),
           &(c_dummy), &iu, &ju, &(i_dummy),
           &(c_dummy), &ivt, &jvt, &(i_dummy),
           &(work[0]), &lwork, &(rwork[0]), &info);

  /* for debug */
  // std::cerr << "matrix_svd<complex>: M= " << M << " N= " << N
  //           << " SIZE= " << size << " lwork= " << lwork
  //           << " rwork_size= " << rwork.size()
  //           << " info = " << info << "\n";

  return info;
}

template <>
int matrix_qr(Matrix<double>& A, Matrix<double>& R) {
  assert(A.n_row() == R.n_row());
  assert(A.n_col() == R.n_col());
  assert(A.local_size() == R.local_size());

  int M = A.n_row();
  int N = A.n_col();
  int K = (M<N) ? M : N;

  int ia, ja, lwork, info;
  std::vector<double> tau(K);
  std::vector<double> work;
  double work_size, work_size_1, work_size_2;
  ia = ja = 1;
  lwork = -1;

  /* Get the size of workspace */
  pdgeqrf_(&M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &work_size_1, &lwork, &info);
  pdorgqr_(&M, &K, &K,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &work_size_2, &lwork, &info);
  work_size = (work_size_1 > work_size_2) ? work_size_1 : work_size_2;
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* QR decomposition */
  pdgeqrf_(&M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &(work[0]), &lwork, &info);

  assert(info==0);

  // copy trianglar matrix R
  for(size_t i=0;i<R.local_size();++i) {
    size_t g_row, g_col;
    R.global_index(i,g_row,g_col);
    R[i] = (g_row > g_col) ? 0.0 : A[i];
  }

  // create orthogonal matrix
  pdorgqr_(&M, &K, &K,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &(work[0]), &lwork, &info);

  return info;
}


template <>
int matrix_qr(Matrix<complex>& A, Matrix<complex>& R) {
  assert(A.n_row() == R.n_row());
  assert(A.n_col() == R.n_col());
  assert(A.local_size() == R.local_size());

  int M = A.n_row();
  int N = A.n_col();
  int K = (M<N) ? M : N;

  int ia, ja, lwork, info;
  std::vector<complex> tau(K);
  std::vector<complex> work;
  complex work_size_1, work_size_2;
  ia = ja = 1;
  lwork = -1;

  /* Get the size of workspace */
  pzgeqrf_(&M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &work_size_1, &lwork, &info);
  pzungqr_(&M, &K, &K,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &work_size_2, &lwork, &info);
  int n1 = static_cast<int>( work_size_1.real() );
  int n2 = static_cast<int>( work_size_2.real() );
  lwork = (n1 > n2) ? n1 : n2;
  work.resize(lwork);

  /* QR decomposition */
  pzgeqrf_(&M, &N,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &(work[0]), &lwork, &info);

  assert(info==0);

  // copy trianglar matrix R
  for(size_t i=0;i<R.local_size();++i) {
    size_t g_row, g_col;
    R.global_index(i,g_row,g_col);
    R[i] = (g_row > g_col) ? 0.0 : A[i];
  }

  // create orthogonal matrix
  pzungqr_(&M, &K, &K,
           A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
           &(tau[0]), &(work[0]), &lwork, &info);

  return info;
}


template <>
int matrix_eigh(Matrix<double>& A, std::vector<double>& W, Matrix<double>& Z) {
  assert(A.n_row() == A.n_col());
  assert(A.n_row() == Z.n_row());
  assert(Z.n_row() == Z.n_col());
  assert(W.size() == A.n_row());

  char jobz = 'V';
  char uplo = 'U';
  int n = A.n_row();
  int ia, ja, iz, jz, lwork, info;
  std::vector<double> work;
  double work_size;
  ia = ja = iz = jz = 1;
  lwork = -1;

  /* Get the size of workspace */
  pdsyev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          Z.head(), &iz, &jz, const_cast<int*>(Z.descriptor()),
          &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* Get eigenvalues and eigenvectors */
  pdsyev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          Z.head(), &iz, &jz, const_cast<int*>(Z.descriptor()),
          &(work[0]), &lwork, &info);

  return info;
};

template <>
int matrix_eigh(Matrix<complex>& A, std::vector<double>& W, Matrix<complex>& Z) {
  assert(A.n_row() == A.n_col());
  assert(A.n_row() == Z.n_row());
  assert(Z.n_row() == Z.n_col());
  assert(W.size() == A.n_row());

  char jobz = 'V';
  char uplo = 'U';
  int n = A.n_row();
  int ia, ja, iz, jz, lwork, lrwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  ia = ja = iz = jz = 1;
  lwork = lrwork = -1;

  /* Get the size of workspace */
  pzheev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          Z.head(), &iz, &jz, const_cast<int*>(Z.descriptor()),
          &work_size, &lwork, &rwork_size, &lrwork, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  lrwork = static_cast<int>(rwork_size);
  rwork.resize(lrwork*2); // ?? bug of SCALAPACK

  /* Get eigenvalues and eigenvectors */
  pzheev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          Z.head(), &iz, &jz, const_cast<int*>(Z.descriptor()),
          &(work[0]), &lwork, &(rwork[0]), &lrwork, &info);

  return info;
};

template <>
int matrix_eigh(Matrix<double>& A, std::vector<double>& W) {
  assert(A.n_row() == A.n_col());
  assert(W.size() == A.n_row());

  char jobz = 'N';
  char uplo = 'U';
  int n = A.n_row();
  int ia, ja, iz, jz, lwork, info;
  std::vector<double> work;
  double work_size;
  ia = ja = iz = jz = 1;
  lwork = -1;
  double d_dummy;
  int i_dummy;

  /* Get the size of workspace */
  pdsyev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          &(d_dummy), &iz, &jz, &(i_dummy),
          &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* Get eigenvalues and eigenvectors */
  pdsyev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          &(d_dummy), &iz, &jz, &(i_dummy),
          &(work[0]), &lwork, &info);

  return info;
};

template <>
int matrix_eigh(Matrix<complex>& A, std::vector<double>& W) {
  assert(A.n_row() == A.n_col());
  assert(W.size() == A.n_row());

  char jobz = 'N';
  char uplo = 'U';
  int n = A.n_row();
  int ia, ja, iz, jz, lwork, lrwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  ia = ja = iz = jz = 1;
  lwork = lrwork = -1;
  complex c_dummy;
  int i_dummy;

  /* Get the size of workspace */
  pzheev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          &(c_dummy), &iz, &jz, &(i_dummy),
          &work_size, &lwork, &rwork_size, &lrwork, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  lrwork = static_cast<int>(rwork_size);
  rwork.resize(lrwork);

  /* Get eigenvalues and eigenvectors */
  pzheev_(&jobz, &uplo, &n,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(W[0]),
          &(c_dummy), &iz, &jz, &(i_dummy),
          &(work[0]), &lwork, &(rwork[0]), &lrwork, &info);

  return info;
};


template <>
int matrix_solve(Matrix<double>& A, Matrix<double>& B) {
  assert(A.n_row() == A.n_col());
  assert(A.n_row() == B.n_row());

  int n = A.n_row();
  int nrhs = B.n_col();
  int ia, ja, ib, jb, info;
  std::vector<int> ipiv(n);
  ia = ja = ib = jb = 1;

  /* Solve linear equation */
  pdgesv_(&n, &nrhs,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(ipiv[0]),
          B.head(), &ib, &jb, const_cast<int*>(B.descriptor()),
          &info);

  return info;
};


template <>
int matrix_solve(Matrix<complex>& A, Matrix<complex>& B) {
  assert(A.n_row() == A.n_col());
  assert(A.n_row() == B.n_row());

  int n = A.n_row();
  int nrhs = B.n_col();
  int ia, ja, ib, jb, info;
  std::vector<int> ipiv(n);
  ia = ja = ib = jb = 1;

  /* Solve linear equation */
  pzgesv_(&n, &nrhs,
          A.head(), &ia, &ja, const_cast<int*>(A.descriptor()),
          &(ipiv[0]),
          B.head(), &ib, &jb, const_cast<int*>(B.descriptor()),
          &info);

  return info;
};


template <>
double matrix_trace(const Matrix<double>& A) {
  const size_t n = A.local_size();
  size_t g_row, g_col;
  double val = 0.0;

  for(size_t i=0;i<n;++i) {
    A.global_index(i, g_row, g_col);
    if(g_row==g_col) val += A[i];
  }

  double recv;
  MPI_Allreduce(&val, &recv, 1, MPI_DOUBLE, MPI_SUM, A.get_comm());
  return recv;
};

template <>
complex matrix_trace(const Matrix<complex>& A) {
  const size_t n = A.local_size();
  size_t g_row, g_col;
  complex val = 0.0;

  for(size_t i=0;i<n;++i) {
    A.global_index(i, g_row, g_col);
    if(g_row==g_col) val += A[i];
  }

  complex recv;
  MPI_Allreduce(&val, &recv, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, A.get_comm());
  return recv;
};

template <> double max(const Matrix<double>& a) {
  const size_t n = a.local_size();
  double send = -DBL_MAX;
  double recv;
  if(n>0) send = *(std::max_element(a.head(), a.head() + n));
  MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE, MPI_MAX, a.get_comm());
  return recv;
};

template <> double min(const Matrix<double>& a) {
  const size_t n = a.local_size();
  double send = DBL_MAX;
  double recv;
  if(n>0) send = *(std::min_element(a.head(), a.head() + n));
  MPI_Allreduce(&send, &recv, 1, MPI_DOUBLE, MPI_MIN, a.get_comm());
  return recv;
};

template <> double max(const Matrix<complex>& a) {return max_abs(a);};
template <> double min(const Matrix<complex>& a) {return min_abs(a);};

//! \endcond
} // namespace scalapack
} // namespace mptensor

#endif // _NO_MPI
