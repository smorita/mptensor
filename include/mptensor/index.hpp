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

  \brief  header file of BasicIndex class template
*/

#ifndef _INDEX_HPP_
#define _INDEX_HPP_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mptensor {
//! \ingroup Index
//! \{

namespace detail {
//! Convert an integer to \c T, or throw std::out_of_range if it does not fit.
template <typename T, typename I>
T checked_index_cast(I value) {
  static_assert(std::is_integral_v<I>, "index values must be integers");
  bool fits = true;
  if constexpr (std::is_signed_v<I> && std::is_unsigned_v<T>) {
    fits = (value >= 0);
  } else if constexpr (std::is_unsigned_v<I> && std::is_signed_v<T>) {
    fits = (value <= static_cast<std::make_unsigned_t<T>>(
                         std::numeric_limits<T>::max()));
  }
  if (!fits) {
    std::ostringstream ss;
    ss << "mptensor: index value " << +value << " is out of range for "
       << (std::is_signed_v<T> ? "a signed" : "an unsigned") << " index";
    throw std::out_of_range(ss.str());
  }
  return static_cast<T>(value);
}
}  // namespace detail

//! List of non-negative or signed integers used as an index, axes, or shape.
/*!
  \c Index (= \c Axes = \c Shape) is the public type. \c detail::UIndex is
  used inside the library.
*/
template <typename T>
class BasicIndex {
 public:
  using value_type = T;
  using index_t = std::vector<T>;

  BasicIndex() = default;
  BasicIndex(const index_t& index) : idx(index) {}
  BasicIndex(std::initializer_list<T> list) : idx(list) {}

  //! Python-like list literal, e.g. <tt>Index(0, 1, -1)</tt>.
  /*! Not explicit: <tt>Axes a = 2;</tt> creates <tt>[2]</tt>. */
  template <typename... Ints,
            typename = std::enable_if_t<(sizeof...(Ints) > 0) &&
                                        (std::is_integral_v<Ints> && ...)>>
  BasicIndex(Ints... js) : idx{detail::checked_index_cast<T>(js)...} {}

  const T& operator[](size_t i) const { return idx[i]; }
  T& operator[](size_t i) { return idx[i]; }
  size_t size() const { return idx.size(); }
  void push(T i) { idx.push_back(i); }
  void resize(size_t n) { idx.resize(n); }
  void assign(size_t n, const T j[]) { idx.assign(j, j + n); }
  void sort() { std::sort(idx.begin(), idx.end()); }

  //! Inverse permutation: <tt>inv[(*this)[i]] = i</tt>.
  BasicIndex inverse() const {
    BasicIndex inv;
    inv.resize(size());
    for (size_t i = 0; i < size(); ++i) {
      inv[static_cast<size_t>(idx[i])] = static_cast<T>(i);
    }
    return inv;
  }

  bool operator==(const BasicIndex& rhs) const { return idx == rhs.idx; }

  BasicIndex& operator+=(const BasicIndex& rhs) {
    idx.insert(idx.end(), rhs.idx.begin(), rhs.idx.end());
    return *this;
  }

 private:
  index_t idx;
};

/*! The format is the same as a list of python, for example "[0, 1, 2]". */
template <typename T>
std::ostream& operator<<(std::ostream& os, const BasicIndex<T>& idx) {
  os << "[";
  if (idx.size() > 0) os << idx[0];
  for (size_t i = 1; i < idx.size(); ++i) os << ", " << idx[i];
  os << "]";
  return os;
}

//! Joint two indices: Index(0,1) + Index(2,3) = Index(0,1,2,3)
template <typename T>
BasicIndex<T> operator+(const BasicIndex<T>& lhs, const BasicIndex<T>& rhs) {
  return (BasicIndex<T>(lhs) += rhs);
}

using Index = BasicIndex<std::size_t>;

namespace detail {
using UIndex = BasicIndex<std::size_t>;
}  // namespace detail

//! Create an increasing sequence. It is similar to range() in python.
inline Index range(const size_t start, const size_t stop) {
  assert(start <= stop);
  Index index;
  index.resize(stop - start);
  for (size_t i = start; i < stop; ++i) index[i - start] = i;
  return index;
}
inline Index range(const size_t stop) { return range(0, stop); }

//! \}
}  // namespace mptensor

#endif  //  _INDEX_HPP_
