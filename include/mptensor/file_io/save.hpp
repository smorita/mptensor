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
  \file   save.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 18 2020

  \brief  Header file of saver.
*/

#ifndef _MPTENSOR_SAVE_HPP_
#define _MPTENSOR_SAVE_HPP_

#include <cstdio>
#include <fstream>
#include <vector>

#include "mptensor/file_io/io_helper.hpp"
#include "mptensor/tensor.hpp"
#include "mptensor/version.hpp"

namespace mptensor {

//! Save a tensor to files.
/*!
  Rank-0 process creates an ASCII file with shape information whose
  name is given by \c filename .  Every process saves tensor elements
  to a binary file \c filename.[rank_no].bin and an ASCII file \c filename.[rank_no].idx .

  \param filename Name of the base file.
  \note \c rank_no in the name of binary files has at least 4 digit, (ex.
  filename.0001.bin).
*/
template <typename MatrixType>
void Tensor<MatrixType>::save(const std::string &filename) const {
  // Create the base file
  if (get_comm_rank() == 0) {
    size_t n = ndim();
    std::ofstream fout(filename);
    fout << "mptensor " << MPTENSOR_VERSION_STRING << "\n";
    fout << "matrix_type= " << MatrixType::matrix_type_tag;
    fout << " (" << MatrixType::matrix_type_name << ")\n";
    fout << "value_type= " << value_type_tag<value_type>();
    fout << " (" << value_type_name<value_type>() << ")\n";
    fout << "comm_size= " << get_comm_size() << "\n";
    fout << "ndim= " << n << "\n";
    fout << "upper_rank= " << upper_rank << "\n";

    fout << "shape=";
    for (size_t i = 0; i < n; ++i) fout << " " << Dim[i];
    fout << "\n";

    fout << "axes_map=";
    for (size_t i = 0; i < n; ++i) fout << " " << axes_map[i];
    fout << "\n";

    fout.close();
  }

  // Save tensor elements
  {
    std::ofstream fout(io_helper::binary_filename(filename, get_comm_rank()),
                       std::ofstream::binary);
    fout.write(reinterpret_cast<const char *>(get_matrix().head()),
               sizeof(value_type) * local_size());
    fout.close();
  }

  // Save additional information of matrix
  {
    std::string indexfile;
    indexfile = io_helper::index_filename(filename, get_comm_rank());
    get_matrix().save_index(indexfile.c_str());
  }
}

//! Save a tensor to files using the old version interface (<= 0.2).
/*!
  Rank-0 process creates an ASCII file with shape information whose
  name is given by \c filename .  Every process saves tensor elements
  to a binary file whose name is \c filename.rank_no .

  \param filename Name of the base file.
  \note \c rank_no in the name of binary files has at least 4 digit, (ex.
  filename.0001).
*/
template <typename MatrixType>
void Tensor<MatrixType>::save_ver_0_2(const char *filename) const {
  std::ofstream fout;

  // Create the base file
  if (get_comm_rank() == 0) {
    fout.open(filename);
    size_t n = ndim();
    fout << n << "\n" << upper_rank << "\n";
    for (size_t i = 0; i < n - 1; ++i) fout << Dim[i] << " ";
    fout << Dim[n - 1] << "\n";
    for (size_t i = 0; i < n - 1; ++i) fout << axes_map[i] << " ";
    fout << axes_map[n - 1] << "\n";
    fout.close();
  }

  // Save tensor elements
  {
    std::vector<char> datafile(std::strlen(filename) + 16);
    std::sprintf(datafile.data(), "%s.%04d", filename, get_comm_rank());

    // save_binary(datafile,get_matrix().head(),local_size());
    std::ofstream fout(datafile.data(), std::ofstream::binary);
    fout.write(reinterpret_cast<const char *>(get_matrix().head()),
               sizeof(value_type) * local_size());
    fout.close();
  }
}

}  // namespace mptensor

#endif  // _MPTENSOR_SAVE_HPP_
