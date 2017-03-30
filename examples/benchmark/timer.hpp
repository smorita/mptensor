/*!
  \file   timer.hpp
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
