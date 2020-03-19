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
  \file   load.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 18 2020

  \brief  Header file of load function.
*/

#ifndef _MPTENSOR_LOAD_HPP_
#define _MPTENSOR_LOAD_HPP_

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>

#include "mptensor/file_io/io_helper.hpp"
#include "mptensor/tensor.hpp"

namespace mptensor {

//! Load a tensor from files.
/*!
  Rank-0 process reads an ASCII file with shape information whose
  name is given by \c filename .  Every process restores tensor elements
  from a binary file \c filename.[rank_no].bin and an ASCII file \c filename.[rank_no].idx .

  \param filename Name of the base file.
  \note \c [rank_no] in the name of binary files has at least 4 digit, (ex.
  filename.0001.bin).
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::load(const std::string &filename) {
  const bool comm_root = (get_comm_rank() == 0);
  std::ifstream fin;
  std::string dummy;
  std::string version;
  size_t ibuf[8] = {0};
  size_t& loaded_version_major = ibuf[0];
  size_t& loaded_version_minor = ibuf[1];
  size_t& loaded_version_patch = ibuf[2];
  size_t& loaded_matrix_type = ibuf[3];
  size_t& loaded_value_type = ibuf[4];
  size_t& loaded_comm_size = ibuf[5];
  size_t& loaded_ndim = ibuf[6];
  size_t& loaded_urank = ibuf[7];
  const size_t this_matrix_type = Matrix<C>::matrix_type_tag;
  const size_t this_value_type = value_type_tag<C>();
  const size_t this_comm_size = get_comm_size();
  Shape loaded_shape;
  Axes loaded_map;

  // Read the base file
  {
    if (comm_root) {
      fin.open(filename);
      fin >> dummy >> version;

      if (dummy != "mptensor") {
        ibuf[0] = 0;
        ibuf[1] = 2;
        ibuf[2] = 0;
        fin.close();
      } else {
        sscanf(version.c_str(), "%lu.%lu.%lu", &(ibuf[0]), &(ibuf[1]),
               &(ibuf[2]));
        fin >> dummy >> ibuf[3] >> dummy;  // matrix_type
        fin >> dummy >> ibuf[4] >> dummy;  // value_type
        fin >> dummy >> ibuf[5];           // comm_size
        fin >> dummy >> ibuf[6];           // ndim
        fin >> dummy >> ibuf[7];           // upper_rank
      }
    }
    Mat.bcast(ibuf, 8, 0);

    if (ibuf[0] == 0 && ibuf[1] <= 2) {
      load_ver_0_2(filename.c_str());
      return;
    }

    const size_t count = 2 * loaded_ndim;
    size_t* buffer = new size_t[count];

    if (comm_root) {
      size_t k = 0;
      fin >> dummy;
      for (size_t i = 0; i < loaded_ndim; ++i) fin >> buffer[k++];
      fin >> dummy;
      for (size_t i = 0; i < loaded_ndim; ++i) fin >> buffer[k++];
      fin.close();
    }
    Mat.bcast(buffer, count, 0);

    loaded_shape.assign(loaded_ndim, buffer);
    loaded_map.assign(loaded_ndim, (buffer + loaded_ndim));

    delete[] buffer;
  }

  assert(loaded_value_type == this_value_type);

  // Initialize tensor shape
  init(loaded_shape, loaded_urank, loaded_map);

  // Read tensor elements
  if (loaded_matrix_type == this_matrix_type &&
      loaded_comm_size == this_comm_size) {
    if (io_helper::debug && comm_root) {
      std::clog << "Info: Load a tensor directly." << std::endl;
    }
    io_helper::load_binary(filename, get_comm_rank(), get_matrix().head(),
                           local_size());
    return;
  } else if (loaded_matrix_type == MATRIX_TYPE_TAG_LAPACK) {
    if (io_helper::debug && comm_root) {
      std::clog << "Info: Load a non-distributed tensor." << std::endl;
    }
    io_helper::load_scalapack(filename, loaded_comm_size, Mat);
    return;
  } else if (loaded_matrix_type == MATRIX_TYPE_TAG_SCALAPACK) {
    if (io_helper::debug && comm_root) {
      std::clog
          << "Info: Load a tensor distributed on different-size communicator."
          << std::endl;
    }
    io_helper::load_scalapack(filename, loaded_comm_size, Mat);
    return;
  }
}

//! Load a tensor from files using the old version interface (<= 0.2).
/*!
  Rank-0 process reads an ASCII file with shape information whose
  name is given by \c filename .  Every process restores tensor elements
  from a binary file whose name is \c filename.rank_no .

  \param filename Name of the base file.
  \note \c rank_no in the name of binary files has at least 4 digit, (ex.
  filename.0001).
*/
template <template <typename> class Matrix, typename C>
void Tensor<Matrix, C>::load_ver_0_2(const char* filename) {
  std::ifstream fin;
  size_t n;
  size_t urank;
  Shape shape;
  Axes map;

  if (get_comm_rank() == 0) {
    std::clog << "Warning: \"" << filename
              << "\" will be loaded using v0.2 interface." << std::endl;
  }

  // Read the base file
  {
    if (get_comm_rank() == 0) {
      fin.open(filename);
      fin >> n;
    }
    Mat.bcast(&n, 1, 0);

    const size_t count = 2 * n + 1;
    size_t* buffer = new size_t[count];

    if (get_comm_rank() == 0) {
      for (size_t i = 0; i < count; ++i) fin >> buffer[i];
      fin.close();
    }
    Mat.bcast(buffer, count, 0);

    urank = buffer[0];
    shape.assign(n, (buffer + 1));
    map.assign(n, (buffer + n + 1));

    delete[] buffer;
  }

  // Initialize tensor shape
  init(shape, urank, map);

  // Read tensor elements
  {
    char* datafile = new char[std::strlen(filename) + 16];
    sprintf(datafile, "%s.%04d", filename, get_comm_rank());

    // load_binary(datafile,get_matrix().head(),local_size());
    fin.open(datafile, std::ofstream::binary);
    fin.read(reinterpret_cast<char*>(get_matrix().head()),
             sizeof(C) * local_size());
    fin.close();

    delete[] datafile;
  }
}

}  // namespace mptensor

#endif  // _MPTENSOR_LOAD_HPP_
