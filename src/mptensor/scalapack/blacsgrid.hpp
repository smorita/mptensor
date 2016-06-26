/*
  Dec. 12, 2014
  Copyright (C) 2014 Satoshi Morita
 */

#ifndef _BLACSGRID_HPP_
#define _BLACSGRID_HPP_
#ifndef _NO_MPI

#include <vector>
#include <mpi.h>

namespace mptensor {
namespace scalapack {


class BlacsGrid {
private:
  // std::vector<int> pnum2mpirank;
  // std::vector<int> mpirank2pnum;
  void init_grid(const MPI_Comm &comm, int nprow, int npcol);
  static bool is_initialized;

public:
  BlacsGrid(const MPI_Comm &comm);

  static void init();
  static void exit();

  MPI_Comm comm;
  int ictxt;
  int nprow, npcol;
  int myprow, mypcol, mypnum;
  int mpisize, myrank;
  int mpirank(int prow, int pcol) const;
  int mpirank() const;
  void show() const;
};

inline int BlacsGrid::mpirank(int prow, int pcol) const {
  return prow * npcol + pcol;
  // return Cblacs_pnum(ictxt, prow, pcol);
  // This equibalence is checked in BlacsGrid::init_grid()

  // int pnum = Cblacs_pnum(ictxt, prow, pcol);
  // return pnum2mpirank[pnum];
}

inline int BlacsGrid::mpirank() const {return myrank;}


} // namespace scalapack
} // namespace mptensor

#endif // _NO_MPI
#endif // _BLACSGRID_HPP_
