/*
  Dec. 12, 2014
  Copyright (C) 2014 Satoshi Morita
 */

#ifndef _NO_MPI

#include <cstdlib>
#include <vector>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include "blacsgrid.hpp"

/* BLACS */
extern "C" {
  void Cblacs_pinfo(int *mypnum, int *nprocs);
  void Cblacs_exit(int NotDone);
  int Cblacs_pnum (int ictxt, int prow, int pcol);
  int Cblacs_gridinit(int *ictxt, char *order, int nprow, int npcol);
  int Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *myprow, int *mypcol);
  int Csys2blacs_handle(MPI_Comm comm);
  MPI_Comm Cblacs2sys_handle(int ictxt);
}


namespace mptensor {
namespace scalapack {

bool BlacsGrid::is_initialized = false;

BlacsGrid::BlacsGrid(const MPI_Comm &comm) {
  if(!is_initialized) BlacsGrid::init();
  int mpisize;
  int dims[2]={0,0};
  MPI_Comm_size(comm, &mpisize);
  MPI_Dims_create(mpisize,2,dims);
  init_grid(comm,dims[0],dims[1]);
}

void BlacsGrid::init_grid(const MPI_Comm &newComm, int nr, int nc) {
  ictxt = Csys2blacs_handle(newComm);
  Cblacs_gridinfo(ictxt, &nprow, &npcol, &myprow, &mypcol);

  if(nprow==nr && npcol==nc) {
    comm = newComm;
  } else {
    if(nprow<0) comm=newComm;
    else {
      /* If the shape of blacsgrid is specified from the size of MPI communicator,
         the program does not enter here. */
      std::cerr << "Warning (BlacsGrid): MPI Communicator was already stored in blacs_handle "
                << "but grid shape is different.";
      MPI_Comm_dup(newComm, &(comm));
    }
    ictxt = Csys2blacs_handle(comm);
    char order = 'R';
    Cblacs_gridinit(&ictxt, &order, nr, nc);
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myprow, &mypcol);

  }

  // comm=Cblacs2sys_handle(ictxt);
  MPI_Comm_size(comm, &mpisize);
  MPI_Comm_rank(comm, &myrank);
  mypnum = Cblacs_pnum(ictxt, myprow, mypcol);

  if( (myrank != mypnum) || (myrank != myprow*npcol+mypcol) ) {
    /* If a process BLACS grid is row-major ordering,
       the program does not enter here. */
    std::cerr << "grid_init: myrank= " << myrank << " myprow= " << mypnum << "\t"
              << "(prow, pcol) = (" << myprow << "," << mypcol << ")\n";
    assert(myrank == mypnum);
    assert(myrank == myprow * npcol + mypcol);
  }

  // pnum2mpirank.resize(mpisize);
  // mpirank2pnum.resize(mpisize);
  // MPI_Allgather(&mypnum,1,MPI_INT,&(mpirank2pnum[0]),1,MPI_INT,comm);
  // for(int i=0;i<mpisize;++i) {
  //   pnum2mpirank[ mpirank2pnum[i] ] = i;
  // }
}

void BlacsGrid::init() {
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if(!mpi_initialized) {
    int argc = 0;
    char **argv=NULL;
    MPI_Init(&argc,&argv);
  }

  int mypnum, nprocs;
  Cblacs_pinfo(&mypnum, &nprocs);
  int ictxt=Csys2blacs_handle(MPI_COMM_WORLD);
  int nprow, npcol, myprow, mypcol;
  Cblacs_gridinfo(ictxt, &nprow, &npcol, &myprow, &mypcol);

  if(nprow<0) {
    int dims[2]={0,0};
    MPI_Dims_create(nprocs,2,dims);
    nprow = dims[0];
    npcol = dims[1];
    char order = 'R';
    Cblacs_gridinit(&ictxt, &order, nprow, npcol);

    std::atexit(BlacsGrid::exit);
  }

  is_initialized = true;

  return;
}

void BlacsGrid::exit() {
  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if(!mpi_finalized) {
    Cblacs_exit(1);
    MPI_Finalize();
  }
 return;
}

void BlacsGrid::show() const {
  std::cout << "BlacsGrid: rank= " << myrank
            << " ictxt= " << ictxt
            << " nprow= " << nprow
            << " npcol= " << npcol
            << " myprow= " << myprow
            << " mypcol= " << mypcol << std::endl;
}


} // namespace scalapack
} // namespace mptensor

#endif // _NO_MPI
