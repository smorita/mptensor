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
  \file   mptensor.hpp
  \author Satoshi Morita <morita@issp.u-tokyo.ac.jp>
  \date   Jun 26 2016

  \brief  Top header file of mptensor.
*/

#ifndef _MPTENSOR_HPP_
#define _MPTENSOR_HPP_

#include "mptensor/complex.hpp"
#include "mptensor/index.hpp"
#include "mptensor/tensor.hpp"

#ifndef _NO_RANDOM
#include "mptensor/rsvd.hpp"
#endif // _NO_RANDOM

#endif // _MPTENSOR_HPP_
