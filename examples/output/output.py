# -*- coding: utf-8 -*-

# mptensor - Parallel Library for Tensor Network Methods
#
# Copyright 2016 Satoshi Morita
#
# mptensor is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# mptensor is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with mptensor.  If not, see
# <https://www.gnu.org/licenses/>.

#  \file   output.py
#  \author Synge Todo <wistaria@phys.s.u-tokyo.ac.jp>
#  \date   February 16 2017
#  \brief  Print out of mptensor

import numpy as np

shape = (2, 3)
A2 = np.zeros(shape)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        A2[i0, i1] = i0 * 3 + i1
print A2

shape = (2, 3, 4)
A3 = np.zeros(shape)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            A3[i0, i1, i2] = i0 * 12 + i1 * 4 + i2
print A3

shape = (2, 3, 4, 5)
A4 = np.zeros(shape)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                A4[i0, i1, i2, i3] = i0 * 60 + i1 * 20 + i2 * 5 + i3
print A4
