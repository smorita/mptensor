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
  \file   timer.hpp
  \author Satoshi Morita <morita@morita-epson3>
  \date   Sep 5 2019
  \brief  Timer
*/

#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#if __cplusplus >= 201103L
#include <chrono>
using namespace std::chrono;

class Timer {
 public:
  Timer(){};
  void now() { t = system_clock::now(); };

 private:
  system_clock::time_point t;

  friend double operator-(Timer& t1, Timer& t0);
};

inline double operator-(Timer& t1, Timer& t0) {
  return double(duration_cast<microseconds>(t1.t - t0.t).count()) * 1.0e-6;
}

#else
#include <ctime>

class Timer {
 public:
  Timer(){};
  void now() { t = std::clock(); };

 private:
  clock_t t;

  friend double operator-(Timer& t1, Timer& t0);
};

inline double operator-(Timer& t1, Timer& t0) {
  return double(t1.t - t0.t) / CLOCKS_PER_SEC;
}

#endif

#endif  // _MPI_TOOL_HPP_
