/*
  Jan. 13, 2015
  Copyright (C) 2015 Satoshi Morita
*/

#include <vector>
#include <algorithm>
#include "tensor.hpp"

namespace mptensor {

bool is_no_transpose(const Axes& axes, const Axes& axes_map, size_t rank) {
  for(size_t i=0;i<rank;++i) {
    if(axes[i] != i) return false;
    if(axes_map[i] != i) return false;
  }
  return true;
}


//! Namespace for debugging
/*!
  Functions in this namespace are called in assert().
 */
namespace debug {

bool check_total_size(const Shape& s1, const Shape& s2) {
  size_t n1=1;
  for(int i=0;i<s1.size();++i) n1 *= s1[i];
  size_t n2=1;
  for(int i=0;i<s2.size();++i) n2 *= s2[i];
  return n1==n2;
}

bool check_extend(const Shape& s_old, const Shape& s_new) {
  const size_t n = s_old.size();
  bool check = true;
  for(size_t i=0;i<n;++i) {
    check = check && (s_old[i] <= s_new[i]);
  }
  return check;
}

bool check_transpose_axes(const Axes& axes, size_t rank) {
  if(axes.size()!=rank) return false;
  Axes v = axes;
  v.sort();
  for(int i=0;i<rank;++i) {
    if(v[i] != i) return false;
  }
  return true;
}

bool check_svd_axes(const Axes& a_row, const Axes& a_col, size_t rank) {
  if(a_row.size()+a_col.size() != rank) return false;
  Axes v = a_row + a_col;
  v.sort();
  for(int i=0;i<rank;++i) {
    if(v[i] != i) return false;
  }
  return true;
}

bool check_trace_axes(const Axes& axes_1, const Axes& axes_2, size_t rank) {
  Axes v = axes_1 + axes_2;
  v.sort();
  for(int i=0;i<rank;++i) {
    if(v[i] != i) return false;
  }
  return true;
}

bool check_trace_axes(const Axes& axes_a, const Axes& axes_b,
                      const Shape& shape_a, const Shape& shape_b) {
  for(size_t i=0;i<axes_a.size();++i) {
    if(shape_a[axes_a[i]] != shape_b[axes_b[i]]) return false;
  }
  Axes axes;
  axes = axes_a;
  axes.sort();
  for(size_t i=0;i<axes.size();++i) {
    if(axes[i] != i) return false;
  }
  axes = axes_b;
  axes.sort();
  for(size_t i=0;i<axes.size();++i) {
    if(axes[i] != i) return false;
  }
  return true;
}

bool check_contract_axes(const Axes& axes_1, const Axes& axes_2, size_t rank) {
  Axes v = axes_1 + axes_2;
  v.sort();
  const size_t n = v.size();
  for(size_t i=0;i<n-1;++i) {
    if(v[i] >= v[i+1]) return false;
  }
  if(v[n-1] >= rank) return false;
  return true;
}

} // nemespace debug

} // namespace mptensor
