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
  \file   index.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jan 08 2015

  \brief  header file of Index class
*/

#ifndef _INDEX_HPP_
#define _INDEX_HPP_

#include <iostream>
#include <vector>

namespace mptensor {
//! \ingroup Index
//! \{

class Index {
public:
  typedef std::vector<size_t> index_t;
  Index();
  Index(const index_t &index);

  // constructors like as Index(size_t j0,size_t j1,size_t j2);
  #include "index_constructor.hpp"

  const size_t& operator[](size_t i) const;
  size_t& operator[](size_t i);
  size_t size() const;
  void push(size_t i);
  void resize(size_t n);

  void assign(size_t n, size_t j[]);
  void sort();
  Index inverse();

  bool operator==(const Index&) const;
  Index& operator+=(const Index&);

private:
  index_t idx;
};

//! \{
std::ostream& operator<<(std::ostream& os, const Index& idx);
Index operator+(const Index&, const Index&);
//! \}

/* ---------- constructors ---------- */

inline Index::Index() : idx() {};
inline Index::Index(const index_t &index) : idx(index) {};


/* ---------- inline member functions ---------- */

inline const size_t& Index::operator[](size_t i) const {return idx[i];};
inline size_t& Index::operator[](size_t i) {return idx[i];};
inline size_t Index::size() const {return idx.size();};
inline void Index::push(size_t i) {idx.push_back(i);};
inline void Index::resize(size_t n) {idx.resize(n);};


/* ---------- non-member functions ---------- */
Index range(const size_t start, const size_t stop);
inline Index range(const size_t stop) {return range(0,stop);};

//! \}
} // namespace mptensor

#endif //  _INDEX_HPP_
