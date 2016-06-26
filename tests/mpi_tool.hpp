/*
  Jan. 14, 2015
  Copyright (C) 2015 Satoshi Morita
 */
#ifndef _MPI_TOOL_HPP_
#define _MPI_TOOL_HPP_

#include <mpi.h>

void mpi_finalize();
void mpi_init(int argc, char **argv);
void mpi_info(const MPI_Comm &comm, int &rank, int &size, bool &is_root);

#endif // _MPI_TOOL_HPP_
