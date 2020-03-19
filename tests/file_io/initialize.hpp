#pragma once

#include <mptensor/mptensor.hpp>

template <typename tensor>
void initialize(tensor &t) {
  using namespace mptensor;
  for (size_t i = 0; i < t.local_size(); ++i) {
    Index idx = t.global_index(i);
    t[i] = idx[0] + idx[1] * 100 + idx[2] * 10000 + idx[3] * 1000000;
  }
  t.transpose({3, 1, 0, 2});
}

template <typename tensor>
tensor initialize2(size_t n) {
  using namespace mptensor;
  tensor t(Shape(n, n + 1, n + 2, n + 3));
  for (size_t i = 0; i < t.local_size(); ++i) {
    Index idx = t.global_index(i);
    t[i] = idx[0] + idx[1] * 100 + idx[2] * 10000 + idx[3] * 1000000;
  }
  t.transpose({3, 1, 0, 2});
  return t;
}
