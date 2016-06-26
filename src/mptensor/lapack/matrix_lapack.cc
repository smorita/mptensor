/*!
  \file   matrix_lapack.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>

  \brief  Definition of functions which call LAPACK and BLAS routines.

  Copyright (C) 2015 Satoshi Morita.
*/

#include <cassert>
#include <complex>
#include <algorithm>
#include "../complex.hpp"
#include "matrix_lapack.hpp"

typedef std::complex<double> complex;

/* BLAS, LAPACK */
extern "C" {
  void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
              double *alpha, double a[], int *lda, double b[], int *ldb,
              double *beta, double c[], int *ldc);
  void dgesvd_(char *jobu, char *jobvt, int *m, int *n,
               double a[], int *lda, double s[],
               double u[], int *ldu, double vt[], int *ldvt,
               double work[], int *lwork, int *info);
  void dgeqrf_(int *m, int *n, double a[], int *lda,
                double tau[], double work[], int *lwork, int *info);
  void dorgqr_(int *m, int *n, int *k, double a[], int *lda,
                double tau[], double work[], int *lwork, int *info);
  void dsyevd_(char *jobz, char *uplo, int *n,
               double a[], int *lda, double w[],
               double work[], int *lwork,
               int iwork[], int *liwork, int *info);


  void zgemm_(char *transa, char *transb, int *m, int *n, int *k,
              complex *alpha, complex a[], int *lda, complex b[], int *ldb,
              complex *beta, complex c[], int *ldc);
  void zgesvd_(char *jobu, char *jobvt, int *m, int *n,
               complex a[], int *lda, double s[],
               complex u[], int *ldu, complex vt[], int *ldvt,
               complex work[], int *lwork, double rwork[], int *info);
  void zgeqrf_(int *m, int *n, complex a[], int *lda,
               complex tau[], complex work[], int *lwork, int *info);
  void zungqr_(int *m, int *n, int *k, complex a[], int *lda,
               complex tau[], complex work[], int *lwork, int *info);
  void zheevd_(char *jobz, char *uplo, int *n,
               complex a[], int *lda, double w[],
               complex work[], int *lwork, double rwork[], int *lrwork,
               int iwork[], int *liwork, int *info);

}

namespace mptensor {
namespace lapack {
//! \cond

template <>
void matrix_product(const Matrix<double>& A, const Matrix<double>& B, Matrix<double>& C) {
  assert( A.n_col()==B.n_row() );
  assert( A.n_row()==C.n_row() );
  assert( B.n_col()==C.n_col() );
  int m = A.n_row();
  int n = B.n_col();
  int k = A.n_col();

  char transa = 'N';
  char transb = 'N';
  double alpha = 1.0;
  double beta = 0.0;
  int lda = A.n_row();
  int ldb = B.n_row();
  int ldc = C.n_row();

  dgemm_(&transa, &transb, &m, &n, &k, &alpha,
         const_cast<double*>(A.head()), &lda,
         const_cast<double*>(B.head()), &ldb,
         &beta, C.head(), &ldc);
};

template <>
void matrix_product(const Matrix<complex>& A, const Matrix<complex>& B, Matrix<complex>& C) {
  assert( A.n_row()==C.n_row() );
  assert( A.n_col()==B.n_row() );
  assert( B.n_col()==C.n_col() );
  int m = A.n_row();
  int n = B.n_col();
  int k = A.n_col();

  char transa = 'N';
  char transb = 'N';
  complex alpha = 1.0;
  complex beta = 0.0;
  int lda = A.n_row();
  int ldb = B.n_row();
  int ldc = C.n_row();

  zgemm_(&transa, &transb, &m, &n, &k, &alpha,
         const_cast<complex*>(A.head()), &lda,
         const_cast<complex*>(B.head()), &ldb,
         &beta, C.head(), &ldc);
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

  int m = A.n_row();
  int n = A.n_col();

  char jobu = 'S';
  char jobvt = 'S';
  int lda = A.n_row();
  int ldu = U.n_row();
  int ldvt = VT.n_row();
  int lwork, info;
  std::vector<double> work;
  double work_size;
  lwork = -1;

  /* Get the size of workspace */
  dgesvd_(&jobu, &jobvt, &m, &n,
          A.head(), &lda,
          &(S[0]),
          U.head(), &ldu,
          VT.head(), &ldvt,
          &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  dgesvd_(&jobu, &jobvt, &m, &n,
          A.head(), &lda,
          &(S[0]),
          U.head(), &ldu,
          VT.head(), &ldvt,
          &(work[0]), &lwork, &info);

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

  int m = A.n_row();
  int n = A.n_col();

  char jobu = 'S';
  char jobvt = 'S';
  int lda = A.n_row();
  int ldu = U.n_row();
  int ldvt = VT.n_row();
  int lwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork(5*size); // 5*min(m,n)
  double d_dummy;
  lwork = -1;

  /* Get the size of workspace */
  zgesvd_(&jobu, &jobvt, &m, &n,
          A.head(), &lda,
          &(S[0]),
          U.head(), &ldu,
          VT.head(), &ldvt,
          &work_size, &lwork, &(d_dummy), &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  zgesvd_(&jobu, &jobvt, &m, &n,
          A.head(), &lda,
          &(S[0]),
          U.head(), &ldu,
          VT.head(), &ldvt,
          &(work[0]), &lwork, &(rwork[0]), &info);

  return info;
}

template <>
int matrix_svd(Matrix<double>& A, std::vector<double>& S) {
  size_t size = (A.n_row() < A.n_col()) ? A.n_row() : A.n_col();
  assert(S.size() == size);

  int m = A.n_row();
  int n = A.n_col();

  char jobu = 'N';
  char jobvt = 'N';
  int lda = A.n_row();
  int lwork, info;
  std::vector<double> work;
  double work_size;
  lwork = -1;
  double d_dummy;
  int i_dummy;

  /* Get the size of workspace */
  dgesvd_(&jobu, &jobvt, &m, &n,
          A.head(), &lda,
          &(S[0]),
          &(d_dummy), &(i_dummy),
          &(d_dummy), &(i_dummy),
          &work_size, &lwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  dgesvd_(&jobu, &jobvt, &m, &n,
          A.head(), &lda,
          &(S[0]),
          &(d_dummy), &(i_dummy),
          &(d_dummy), &(i_dummy),
          &(work[0]), &lwork, &info);

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
  int lda = A.n_row();
  int lwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  lwork = -1;
  complex c_dummy;
  int i_dummy;

  /* Get the size of workspace */
  zgesvd_(&jobu, &jobvt, &M, &N,
          A.head(), &lda,
          &(S[0]),
          &(c_dummy), &(i_dummy),
          &(c_dummy), &(i_dummy),
          &work_size, &lwork, &rwork_size, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  rwork.resize(static_cast<int>(rwork_size));

  /* SVD: A(m,n) = U(m,i) * S(i) * VT(i,n) */
  zgesvd_(&jobu, &jobvt, &M, &N,
          A.head(), &lda,
          &(S[0]),
          &(c_dummy), &(i_dummy),
          &(c_dummy), &(i_dummy),
          &(work[0]), &lwork, &(rwork[0]), &info);

  return info;
}

template <>
int matrix_qr(Matrix<double>& A, Matrix<double>& R) {
  assert(A.n_row() == R.n_row());
  assert(A.n_col() == R.n_col());
  assert(A.local_size() == R.local_size());

  int m = A.n_row();
  int n = A.n_col();
  int k = (m<n) ? m : n;
  int lda = A.n_row();
  int lwork, info;
  std::vector<double> tau(k);
  std::vector<double> work;
  double work_size, work_size_1, work_size_2;
  lwork = -1;

  /* Get the size of workspace */
  dgeqrf_(&m, &n, A.head(), &lda,
          &(tau[0]), &work_size_1, &lwork, &info);
  dorgqr_(&m, &k, &k, A.head(), &lda,
          &(tau[0]), &work_size_2, &lwork, &info);
  work_size = (work_size_1 > work_size_2) ? work_size_1 : work_size_2;
  lwork = static_cast<int>(work_size);
  work.resize(lwork);

  /* QR decomposition */
  dgeqrf_(&m, &n, A.head(), &lda,
          &(tau[0]), &(work[0]), &lwork, &info);

  assert(info==0);

  // copy trianglar matrix R
  for(size_t i=0;i<R.local_size();++i) {
    size_t g_row, g_col;
    R.global_index(i,g_row,g_col);
    R[i] = (g_row > g_col) ? 0.0 : A[i];
  }

  // create orthogonal matrix
  dorgqr_(&m, &k, &k, A.head(), &lda,
          &(tau[0]), &(work[0]), &lwork, &info);

  return info;
}


template <>
int matrix_qr(Matrix<complex>& A, Matrix<complex>& R) {
  assert(A.n_row() == R.n_row());
  assert(A.n_col() == R.n_col());
  assert(A.local_size() == R.local_size());

  int m = A.n_row();
  int n = A.n_col();
  int k = (m<n) ? m : n;
  int lda = A.n_row();

  int ia, ja, lwork, info;
  std::vector<complex> tau(k);
  std::vector<complex> work;
  complex work_size_1, work_size_2;
  ia = ja = 1;
  lwork = -1;

  /* Get the size of workspace */
  zgeqrf_(&m, &n, A.head(), &lda,
          &(tau[0]), &work_size_1, &lwork, &info);
  zungqr_(&m, &k, &k, A.head(), &lda,
          &(tau[0]), &work_size_2, &lwork, &info);
  int n1 = static_cast<int>( work_size_1.real() );
  int n2 = static_cast<int>( work_size_2.real() );
  lwork = (n1 > n2) ? n1 : n2;
  work.resize(lwork);

  /* QR decomposition */
  zgeqrf_(&m, &n, A.head(), &lda,
          &(tau[0]), &(work[0]), &lwork, &info);

  assert(info==0);

  // copy trianglar matrix R
  for(size_t i=0;i<R.local_size();++i) {
    size_t g_row, g_col;
    R.global_index(i,g_row,g_col);
    R[i] = (g_row > g_col) ? 0.0 : A[i];
  }

  // create orthogonal matrix
  zungqr_(&m, &k, &k, A.head(), &lda,
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
  int lda = A.n_row();
  int lwork, liwork, info;
  std::vector<double> work;
  std::vector<int> iwork;
  double work_size;
  int iwork_size;
  lwork = -1;
  liwork = -1;

  /* Get the size of workspace */
  dsyevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &work_size, &lwork, &iwork_size, &liwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  liwork = iwork_size;
  iwork.resize(liwork);

  /* Get eigenvalues and eigenvectors */
  dsyevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &(work[0]), &lwork, &(iwork[0]), &liwork, &info);

  Z = A;

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
  int lda = A.n_row();
  int lwork, lrwork, liwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  std::vector<int> iwork;
  int iwork_size;
  lwork = lrwork = liwork = -1;

  /* Get the size of workspace */
  zheevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &work_size, &lwork, &rwork_size, &lrwork,
          &iwork_size, &liwork, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  lrwork = static_cast<int>(rwork_size);
  rwork.resize(lrwork*2); // ?? bug of SCALAPACK
  liwork = iwork_size;
  iwork.resize(liwork);

  /* Get eigenvalues and eigenvectors */
  zheevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &(work[0]), &lwork, &(rwork[0]), &lrwork,
          &(iwork[0]), &liwork, &info);
  Z = A;

  return info;
};


template <>
int matrix_eigh(Matrix<double>& A, std::vector<double>& W) {
  assert(A.n_row() == A.n_col());
  assert(W.size() == A.n_row());

  char jobz = 'N';
  char uplo = 'U';
  int n = A.n_row();
  int lda = A.n_row();
  int lwork, liwork, info;
  std::vector<double> work;
  std::vector<int> iwork;
  double work_size;
  int iwork_size;
  lwork = -1;
  liwork = -1;

  /* Get the size of workspace */
  dsyevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &work_size, &lwork, &iwork_size, &liwork, &info);
  lwork = static_cast<int>(work_size);
  work.resize(lwork);
  liwork = iwork_size;
  iwork.resize(liwork);

  /* Get eigenvalues */
  dsyevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
         &(work[0]), &lwork, &(iwork[0]), &liwork, &info);

  return info;
};


template <>
int matrix_eigh(Matrix<complex>& A, std::vector<double>& W) {
  assert(A.n_row() == A.n_col());
  assert(W.size() == A.n_row());

  char jobz = 'N';
  char uplo = 'U';
  int n = A.n_row();
  int lda = A.n_row();
  int lwork, lrwork, liwork, info;
  std::vector<complex> work;
  complex work_size;
  std::vector<double> rwork;
  double rwork_size;
  std::vector<int> iwork;
  int iwork_size;
  lwork = lrwork = liwork = -1;

  /* Get the size of workspace */
  zheevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &work_size, &lwork, &rwork_size, &lrwork,
          &iwork_size, &liwork, &info);
  lwork = static_cast<int>(work_size.real());
  work.resize(lwork);
  lrwork = static_cast<int>(rwork_size);
  rwork.resize(lrwork*2); // ?? bug of SCALAPACK
  liwork = iwork_size;
  iwork.resize(liwork);

  /* Get eigenvalues */
  zheevd_(&jobz, &uplo, &n, A.head(), &lda, &(W[0]),
          &(work[0]), &lwork, &(rwork[0]), &lrwork,
          &(iwork[0]), &liwork, &info);

  return info;
};


template <> double max(const Matrix<double>& a) {
  return *(std::max_element(a.head(), a.head() + a.local_size()));

};
template <> double min(const Matrix<double>& a) {
  return *(std::min_element(a.head(), a.head() + a.local_size()));
};

template <> double max(const Matrix<complex>& a) {return max_abs(a);};
template <> double min(const Matrix<complex>& a) {return min_abs(a);};


//! \endcond
} // namespace lapack
} // namespace mptensor
