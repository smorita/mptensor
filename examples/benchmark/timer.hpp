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

#include <mpi.h>

#ifndef _TIMER_HPP_
#define _TIMER_HPP_

namespace examples {
namespace benchmark {

class Timer {
public:
  Timer() {};
  void start() {t_start = MPI_Wtime();};
  void stop() {t_end = MPI_Wtime();};
  double result() {return t_end - t_start;};
private:
  double t_start;
  double t_end;
};

} // namespace benchmark
} // namespace examples

#endif // _TIMER_HPP_
