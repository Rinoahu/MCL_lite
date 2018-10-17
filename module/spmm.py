#!usr/bin/env python
# optimized spmm for pypy

import sys
from time import time
import os
import gc
from struct import pack, unpack
from math import sqrt
import mimetypes
import gzip
import bz2 as bzip2
import bz2
from itertools import izip
from scipy import sparse as sps
import tempfile
import cPickle
import mmap
try:
    from _numpypy import multiarray as npy
except:
    npy = np



# mmap based array
class darray:

    def __init__(self, fn, size, dtype='float32'):
        self.fn = fn
        self.size = int(size)
        self.dtype = dtype

        if self.dtype == 'float8' or self.dtype == 'int8':
            self.stride = 1
        elif self.dtype == 'float16' or self.dtype == 'int16':
            self.stride = 2
        elif self.dtype == 'float32' or self.dtype == 'int32':
            self.stride = 4
        else:
            self.stride = 8

        self.size = size

        self.f = open(self.fn, "w+b")
        self.f.seek(self.stride * self.size - 1)
        self.f.write('\x00')
        self.f.flush()

        self.buf = mmap.mmap(self.f.fileno(), self.stride *
                             self.size, prot=mmap.ACCESS_WRITE)
        self.dat = npy.frombuffer(self.buf, self.dtype)

    # resize the array
    def resize(self, size=-1):
        L = size > 0 and size * self.stride - \
            1 or self.stride * (self.dat.size + 10**6) - 1
        L = int(L)
        self.f.seek(L)
        self.f.write('\x00')
        self.f.close()
        self.f = open(self.fn, "r+b")
        L += 1
        self.buf = mmap.mmap(self.f.fileno(), L, prot=mmap.ACCESS_WRITE)
        self.dat = npy.frombuffer(self.buf, self.dtype)


# a x b = z
def csrmm(x0, y0, z0, x1, y1, z1, z2n='tmp'):

    R = x0.size
    C = (y0.size + y1.size) * 10
    x2d = darray(z2n + '_x.npz', R, dtype=x0.dtype)
    y2d = darray(z2n + '_y.npz', C, dtype=y0.dtype)
    z2d = darray(z2n + '_z.npz', C, dtype=z0.dtype)

    x2, y2, z2 = x2d.dat, y2d.dat, z2d.dat
    # row value
    row_i = npy.zeros(R, z0.dtype)
    visit = npy.zeros(R, 'int8')
    key = npy.zeros(R, 'int32')
    z2_ptr = 0
    for i in xrange(x0.size-1):
        ist, ied = x0[i:i+2]
        if ist == ied:
            continue

        # get a[i,j], j
        i2 = 0
        for i0 in xrange(ist, ied):
            k = y0[i0]
            aik = z0[i0]
            kst, ked = x1[k:k+1]
            #bi = z1[x1st: x1ed]
            for i1 in xrange(kst, ked):
                j = y1[i1]
                bkj = z1[i1]
                zij = aik * bkj
                row_i[j] += zij

                if visit[j] == 0:
                    key[i2] = j
                    i2 += 1
                    visit[j] = 1
        # add to z
        for i3 in xrange(i2):
            j = key[i3]
            zij = row_i[j]
            if zij > 0:

                if z2_ptr > z2.size:
                    y2d.resize()
                    z2d.resize()
                    y2, z2 = x2d.dat, z2d.dat

                y2[z2_ptr] = j
                z2[z2_ptr] = zij
                visit[j] = 0
                z2_ptr += 1

        x2[x2_ptr] = z2_ptr
        x2_ptr += 1


