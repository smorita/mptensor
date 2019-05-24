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
  \file   blacsgrid.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Dec 12 2014

  \brief  BlacsGrid class
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
