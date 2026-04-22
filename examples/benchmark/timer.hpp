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
  \file   benchmark/timer.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Thu Aug 25
  \brief  Timer class
*/

#ifndef _TIMER_HPP_
#define _TIMER_HPP_

#include <chrono>

namespace examples {
namespace benchmark {

using namespace std::chrono;

class Timer {
 public:
  Timer(){};
  void start() { t_start = system_clock::now(); };
  void stop() { t_end = system_clock::now(); };
  double result() {
    return double(duration_cast<microseconds>(t_end - t_start).count()) * 1.0e-6;
  };

 private:
  system_clock::time_point t_start;
  system_clock::time_point t_end;
};

}  // namespace benchmark
}  // namespace examples

#endif  // _TIMER_HPP_
