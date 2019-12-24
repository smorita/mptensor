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
  \file   index.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jan 08 2015

  \brief  Index class
*/

#include <iostream>
#include <algorithm>
#include <cassert>
#include "index.hpp"

namespace mptensor {


void Index::assign(size_t n, size_t j[]) {
  idx.resize(n);
  idx.assign(j,j+n);
};

void Index::sort() {
  std::sort(idx.begin(), idx.end());
}

Index Index::inverse() {
  Index inv;
  inv.resize(size());
  for(int i=0;i<size();++i) {
    inv[(*this)[i]] = i;
  }
  return inv;
};

bool Index::operator==(const Index& rhs) const {
  if( size() != rhs.size() ) return false;
  for(int i=0;i<size();++i) {
    if(idx[i] != rhs[i]) return false;
  }
  return true;
}

Index& Index::operator+=(const Index& rhs) {
  idx.insert(idx.end(), rhs.idx.begin(), rhs.idx.end());
  return (*this);
}

/* ---------- non-member functions ---------- */

/*! The format is the same as a list of python, for example "[0, 1, 2]".
  \relates Index
*/
std::ostream& operator<<(std::ostream& os, const Index& idx) {
  os << "[";
  if(idx.size()>0) os << idx[0];
  for(int i=1;i<idx.size();++i) {
    os << ", " << idx[i];
  }
  os << "]";
  return os;
};

//! Joint two indices
/*!
  Index(0,1) + Index(2,3) = Index(0,1,2,3)
  \relates Index
*/
Index operator+(const Index& lhs, const Index& rhs) {
  return (Index(lhs) += rhs);
};


//! Create an increasing sequence. it is similar to range() in python
Index range(const size_t start, const size_t stop) {
  assert(start <= stop);
  Index index;
  index.resize(stop-start);
  for(size_t i=start;i<stop;++i) index[i-start] = i;
  return index;
}

} // namespace mptensor
