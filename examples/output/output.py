# -*- coding: utf-8 -*-
#  \file   example.py
#  \author Synge Todo <wistaria@phys.s.u-tokyo.ac.jp>
#  \date   February 16 2017
#  \brief  Print out of mptensor

import numpy as np

shape = (2,3)
A2 = np.zeros(shape)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        A2[i0,i1] = i0 * 3 + i1
print A2

shape = (2,3,4)
A3 = np.zeros(shape)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            A3[i0,i1,i2] = i0 * 12 + i1 * 4 + i2
print A3

shape = (2,3,4,5)
A4 = np.zeros(shape)
for i0 in range(shape[0]):
    for i1 in range(shape[1]):
        for i2 in range(shape[2]):
            for i3 in range(shape[3]):
                A4[i0,i1,i2,i3] = i0 * 60 + i1 * 20 + i2 * 5 + i3
print A4
