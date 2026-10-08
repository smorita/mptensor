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
//! True for unscoped enumerations (implicitly convertible to their underlying type).
template <typename I, bool = std::is_enum_v<I>>
struct is_unscoped_enum : std::false_type {};
template <typename I>
struct is_unscoped_enum<I, true>
    : std::bool_constant<std::is_convertible_v<I, std::underlying_type_t<I>>> {};

//! True for integer types and unscoped enumerations (accepted as index values).
template <typename I>
inline constexpr bool is_index_value_v =
    std::is_integral_v<I> || is_unscoped_enum<I>::value;

//! Convert an integer to \c T, or throw std::out_of_range if it does not fit.
template <typename T, typename I>
T checked_index_cast(I value) {
  static_assert(is_index_value_v<I>, "index values must be integers");
  if constexpr (std::is_enum_v<I>) {
    return checked_index_cast<T>(static_cast<std::underlying_type_t<I>>(value));
  } else {
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
}
}  // namespace detail

//! List of non-negative or signed integers used as an index, axes, or shape.
/*!
  \c Index (= \c Axes = \c Shape) is the public type. \c detail::UIndex
  (= \c detail::UAxes = \c detail::UShape) is
  used inside the library.
*/
template <typename T>
class BasicIndex {
 public:
  using value_type = T;
  using index_t = std::vector<T>;

  BasicIndex() = default;
  BasicIndex(const index_t& index) : idx(index) {}
  //! Converting constructor from a vector of another integer type.
  template <typename U,
            typename = std::enable_if_t<detail::is_index_value_v<U> &&
                                        !std::is_same_v<U, T>>>
  BasicIndex(const std::vector<U>& index) {
    idx.reserve(index.size());
    for (const U& v : index) idx.push_back(detail::checked_index_cast<T>(v));
  }
  BasicIndex(std::initializer_list<T> list) : idx(list) {}

  //! Python-like list literal, e.g. <tt>Index(0, 1, -1)</tt>.
  /*! Not explicit: <tt>Axes a = 2;</tt> creates <tt>[2]</tt>. */
  template <typename... Ints,
            typename = std::enable_if_t<(sizeof...(Ints) > 0) &&
                                        (detail::is_index_value_v<Ints> && ...)>>
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

using Index = BasicIndex<std::ptrdiff_t>;  //!< Public element index.
using Axes = Index;                        //!< Public axes.
using Shape = Index;                       //!< Public shape.

namespace detail {
using UIndex = BasicIndex<std::size_t>;  //!< Internal element index.
using UAxes = UIndex;                    //!< Internal axes.
using UShape = UIndex;                   //!< Internal shape.

//! Normalize one value into [0, n) (or [0, n] if \c end_inclusive).
/*!
  \param position Element number shown in the error message; negative for a
  scalar argument.
*/
inline size_t normalize_value(std::ptrdiff_t v, size_t n, bool end_inclusive,
                              const char* kind, std::ptrdiff_t position) {
  const std::ptrdiff_t sn = static_cast<std::ptrdiff_t>(n);
  const bool ok = end_inclusive ? (v >= -sn && v <= sn) : (v >= -sn && v < sn);
  if (!ok) {
    std::ostringstream ss;
    ss << "mptensor: " << kind << " " << v;
    if (position >= 0) ss << " (position " << position << ")";
    ss << " is out of range [" << -sn << ", " << sn
       << (end_inclusive ? "]" : ")");
    throw std::out_of_range(ss.str());
  }
  return static_cast<size_t>(v < 0 ? v + sn : v);
}

//! [0, 1, ..., n-1]
inline UAxes identity_axes(size_t n) {
  UAxes axes;
  axes.resize(n);
  for (size_t i = 0; i < n; ++i) axes[i] = i;
  return axes;
}

//! Normalize a scalar slice [begin, end) on an axis of size \c n.
/*! \throw std::out_of_range if a bound is out of range or the slice is empty. */
inline std::pair<size_t, size_t> normalize_slice_range(std::ptrdiff_t begin,
                                                       std::ptrdiff_t end,
                                                       size_t n, size_t axis) {
  const std::ptrdiff_t pos = static_cast<std::ptrdiff_t>(axis);
  const size_t b = normalize_value(begin, n, false, "slice begin", pos);
  const size_t e = normalize_value(end, n, true, "slice end", pos);
  if (b >= e) {
    std::ostringstream ss;
    ss << "mptensor: slice [" << begin << ", " << end << ") on axis " << axis
       << " is empty (normalized to [" << b << ", " << e << "))";
    throw std::out_of_range(ss.str());
  }
  return {b, e};
}

//! Normalize Index-style slice bounds; raw begin[r] == end[r] means the full axis.
inline void normalize_slice_ranges(const Index& begin, const Index& end,
                                   const UShape& shape, UIndex& ubegin,
                                   UIndex& uend) {
  const size_t rank = shape.size();
  if (begin.size() != rank || end.size() != rank) {
    std::ostringstream ss;
    ss << "mptensor: slice bounds " << begin << " and " << end
       << " do not match the tensor rank " << rank;
    throw std::invalid_argument(ss.str());
  }
  ubegin.resize(rank);
  uend.resize(rank);
  for (size_t r = 0; r < rank; ++r) {
    if (begin[r] == end[r]) {
      ubegin[r] = 0;
      uend[r] = shape[r];
    } else {
      const std::pair<size_t, size_t> be =
          normalize_slice_range(begin[r], end[r], shape[r], r);
      ubegin[r] = be.first;
      uend[r] = be.second;
    }
  }
}
}  // namespace detail

//! Normalize an axis: [-rank, rank) -> [0, rank).
inline size_t normalize_axis(std::ptrdiff_t a, size_t rank) {
  return detail::normalize_value(a, rank, false, "axis", -1);
}

//! Normalize each axis: [-rank, rank) -> [0, rank).
inline detail::UAxes normalize_axes(const Axes& axes, size_t rank) {
  detail::UAxes result;
  result.resize(axes.size());
  for (size_t i = 0; i < axes.size(); ++i) {
    result[i] = detail::normalize_value(axes[i], rank, false, "axis",
                                        static_cast<std::ptrdiff_t>(i));
  }
  return result;
}

//! Normalize an element index: [-n, n) -> [0, n).
inline size_t normalize_index(std::ptrdiff_t i, size_t n) {
  return detail::normalize_value(i, n, false, "index", -1);
}

//! Normalize a global element index against \c shape.
inline detail::UIndex normalize_index(const Index& idx,
                                      const detail::UShape& shape) {
  if (idx.size() != shape.size()) {
    std::ostringstream ss;
    ss << "mptensor: index " << idx << " has " << idx.size()
       << " elements, but the tensor has rank " << shape.size();
    throw std::invalid_argument(ss.str());
  }
  detail::UIndex result;
  result.resize(idx.size());
  for (size_t k = 0; k < idx.size(); ++k) {
    result[k] = detail::normalize_value(idx[k], shape[k], false, "index",
                                        static_cast<std::ptrdiff_t>(k));
  }
  return result;
}

//! Normalize an exclusive slice end: [-n, n] -> [0, n].
inline size_t normalize_slice_end(std::ptrdiff_t e, size_t n) {
  return detail::normalize_value(e, n, true, "slice end", -1);
}

//! Convert a public shape to the internal type. Negative sizes are invalid.
inline detail::UShape to_internal_shape(const Shape& s) {
  detail::UShape result;
  result.resize(s.size());
  for (size_t k = 0; k < s.size(); ++k) {
    if (s[k] < 0) {
      std::ostringstream ss;
      ss << "mptensor: shape " << s << " has a negative size at position " << k;
      throw std::invalid_argument(ss.str());
    }
    result[k] = static_cast<size_t>(s[k]);
  }
  return result;
}

//! Convert an internal index to the public type.
inline Index to_public(const detail::UIndex& u) {
  Index result;
  result.resize(u.size());
  for (size_t k = 0; k < u.size(); ++k) {
    result[k] = detail::checked_index_cast<std::ptrdiff_t>(u[k]);
  }
  return result;
}

//! Arithmetic sequence start, start+step, ... up to (not including) stop.
/*!
  Same as range() in python: the result is empty if \c stop is not reached in
  the direction of \c step, e.g. <tt>range(3, 2)</tt> or <tt>range(0, 5, -1)</tt>.
  Negative values are allowed, e.g. <tt>range(-1, -5, -1)</tt> is
  <tt>[-1, -2, -3, -4]</tt>, which reverses the axes of a rank-4 tensor.

  \throw std::invalid_argument if <tt>step == 0</tt>.
*/
inline Index range(const std::ptrdiff_t start, const std::ptrdiff_t stop,
                   const std::ptrdiff_t step) {
  if (step == 0) {
    throw std::invalid_argument("mptensor: range() step must not be zero");
  }
  // Number of elements, computed in unsigned arithmetic to avoid overflow.
  size_t n = 0;
  if (step > 0 && start < stop) {
    const size_t diff = static_cast<size_t>(stop) - static_cast<size_t>(start);
    n = (diff - 1) / static_cast<size_t>(step) + 1;
  } else if (step < 0 && start > stop) {
    const size_t diff = static_cast<size_t>(start) - static_cast<size_t>(stop);
    const size_t ustep = size_t(0) - static_cast<size_t>(step);
    n = (diff - 1) / ustep + 1;
  }
  Index index;
  index.resize(n);
  std::ptrdiff_t v = start;
  for (size_t i = 0; i < n; ++i) {
    index[i] = v;
    if (i + 1 < n) v += step;
  }
  return index;
}

//! Same as <tt>range(start, stop, 1)</tt>.
inline Index range(const std::ptrdiff_t start, const std::ptrdiff_t stop) {
  return range(start, stop, 1);
}

//! Same as <tt>range(0, stop, 1)</tt>.
inline Index range(const std::ptrdiff_t stop) { return range(0, stop, 1); }

//! \}
}  // namespace mptensor

#endif  //  _INDEX_HPP_
