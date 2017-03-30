/*!
  \file   example.cc
  \author Synge Todo <wistaria@phys.s.u-tokyo.ac.jp>
  \date   February 16 2017
  \brief  Print out of mptensor
*/

#include <iostream>
#include <mpi.h>
#include <mptensor.hpp>

/* Main function */
int main(int argc, char **argv) {
  using namespace mptensor;
  typedef Tensor<scalapack::Matrix,double> ptensor;

  MPI_Init(&argc, &argv);

  Shape shape = Shape(2,3);
  ptensor A2(shape);
  for (int i0=0;i0<shape[0];++i0)
    for (int i1=0;i1<shape[1];++i1)
      A2.set_value(Index(i0,i1),i0*3+i1);
  std::cout << A2;

  shape = Shape(2,3,4);
  ptensor A3(shape);
  for (int i0=0;i0<shape[0];++i0)
    for (int i1=0;i1<shape[1];++i1)
      for (int i2=0;i2<shape[2];++i2)
        A3.set_value(Index(i0,i1,i2),i0*12+i1*4+i2);
  std::cout << A3;

  shape = Shape(2,3,4,5);
  ptensor A4(shape);
  for (int i0=0;i0<shape[0];++i0)
    for (int i1=0;i1<shape[1];++i1)
      for (int i2=0;i2<shape[2];++i2)
        for (int i3=0;i3<shape[3];++i3)
          A4.set_value(Index(i0,i1,i2,i3),i0*60+i1*20+i2*5+i3);
  std::cout << A4;

  MPI_Finalize();
}
