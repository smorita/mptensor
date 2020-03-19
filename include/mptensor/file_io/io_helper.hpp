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
  \file   io_helper.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Mar 18 2020

  \brief  Header file of helper functions for file io.
*/

#ifndef _MPTENSOR_LOAD_HELPER_HPP_
#define _MPTENSOR_LOAD_HELPER_HPP_

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "mptensor/matrix.hpp"

namespace mptensor {
namespace io_helper {

#if defined(NDEBUG)
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

inline std::string binary_filename(const std::string& prefix, int comm_rank) {
  std::ostringstream ss;
  ss << prefix << ".";
  ss << std::setw(4) << std::setfill('0') << comm_rank;
  ss << ".bin";
  return ss.str();
}

inline std::string index_filename(const std::string& prefix, int comm_rank) {
  std::ostringstream ss;
  ss << prefix << ".";
  ss << std::setw(4) << std::setfill('0') << comm_rank;
  ss << ".idx";
  return ss.str();
}

template <typename C>
void load_binary(const std::string& prefix, int comm_rank, C* data_head,
                 std::size_t local_size) {
  std::string filename;
  filename = io_helper::binary_filename(prefix, comm_rank);
  std::ifstream fin(filename, std::ofstream::binary);
  assert(fin.is_open());
  fin.read(reinterpret_cast<char*>(data_head), sizeof(C) * local_size);
  fin.close();
}

template <template <typename> class Matrix, typename C>
void load_local_files(const std::string& prefix, int comm_rank,
                      size_t& local_size, std::vector<int>& dest_rank,
                      std::vector<size_t>& local_idx, std::vector<C>& data,
                      const Matrix<C>& mat) {
  std::vector<size_t> g_row;
  std::vector<size_t> g_col;
  size_t local_n_row;
  size_t local_n_col;

  {
    std::string dummy;
    std::string filename;
    filename = io_helper::index_filename(prefix, comm_rank);
    std::ifstream fin(filename);
    assert(fin.is_open());

    fin >> dummy >> local_size;
    fin >> dummy >> local_n_row;
    fin >> dummy >> local_n_col;
    assert(local_size == local_n_row * local_n_col);

    g_row.resize(local_n_row);
    g_col.resize(local_n_col);

    fin >> dummy;
    for (size_t i = 0; i < local_n_row; ++i) {
      fin >> g_row[i];
    }
    fin >> dummy;
    for (size_t i = 0; i < local_n_col; ++i) {
      fin >> g_col[i];
    }
    fin.close();
  }

  dest_rank.resize(local_size);
  local_idx.resize(local_size);
  for (size_t i = 0; i < local_size; ++i) {
    size_t i_row = i % local_n_row;
    size_t i_col = i / local_n_row;
    mat.local_position(g_row[i_row], g_col[i_col], dest_rank[i], local_idx[i]);
  }

  data.resize(local_size);
  load_binary(prefix, comm_rank, &(data[0]), local_size);
}

template <template <typename> class Matrix, typename C>
void load_scalapack(const std::string& prefix, int loaded_comm_size,
                    Matrix<C>& mat) {
  const size_t this_comm_size = mat.get_comm_size();
  const size_t this_comm_rank = mat.get_comm_rank();
  size_t local_size;
  std::vector<int> dest_rank;
  std::vector<size_t> local_idx;
  std::vector<C> data;
  int n = loaded_comm_size / this_comm_size;
  if (loaded_comm_size % this_comm_size > 0) n += 1;
  for (int i = 0; i < n; ++i) {
    int comm_rank = this_comm_rank + i * this_comm_size;
    if (comm_rank < loaded_comm_size) {
      load_local_files(prefix, comm_rank, local_size, dest_rank, local_idx,
                       data, mat);
    } else {
      local_size = 0;
      local_idx.resize(local_size);
      dest_rank.resize(local_size);
      data.resize(local_size);
    }
    replace_matrix_data(data, dest_rank, local_idx, mat);
  }
}

}  // namespace io_helper
}  // namespace mptensor
#endif  // _MPTENSOR_IO_HELPER_HPP_
