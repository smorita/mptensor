/*
  Jan. 08, 2015
  Copyright (C) 2015 Satoshi Morita
*/

#ifndef _INDEX_HPP_
#define _INDEX_HPP_

#include <iostream>
#include <vector>

namespace mptensor {

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

} // namespace mptensor

#endif //  _INDEX_HPP_
