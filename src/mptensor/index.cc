/*
  Jan. 08, 2015
  Copyright (C) 2015 Satoshi Morita
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
  bool same=true;
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
