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
  \file   ising.cc
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Thu Nov 5 2015
  \brief  Two-dimensional Ising model
*/

#include <cmath>

#include "ising.hpp"

namespace examples {
namespace Ising_2D {

double exact_free_energy_integrand(double beta, double x) {
  double k = 2.0 * sinh(2.0 * beta) / (cosh(2.0 * beta) * cosh(2.0 * beta));
  return log(1.0 + sqrt(fabs(1.0 - k * k * cos(x) * cos(x))));
}

double exact_free_energy(double temp) {
  const double pi = M_PI;
  const double beta = 1.0 / temp;

  // Simpson's rule
  // integral(0.0, pi/2.0, func(x));
  double integral = 0.0;
  {
    const int n = 500000;
    const double x_start = 0.0;
    const double x_end = 0.5 * pi;
    const double h = (x_end - x_start) / static_cast<double>(2 * n);
    double sum_even = 0.0;
    double sum_odd = 0.0;
    for (int i = 0; i < n; ++i) {
      sum_odd += exact_free_energy_integrand(beta, h * (2 * i + 1));
    }
    for (int i = 1; i < n; ++i) {
      sum_even += exact_free_energy_integrand(beta, h * (2 * i));
    }
    integral += exact_free_energy_integrand(beta, x_start);
    integral += exact_free_energy_integrand(beta, x_end);
    integral += 2.0 * sum_even + 4.0 * sum_odd;
    integral *= h / 3.0;
  }

  return -temp * (log(sqrt(2.0) * cosh(2.0 * beta)) + integral / pi);
}

}  // namespace Ising_2D
}  // namespace examples
