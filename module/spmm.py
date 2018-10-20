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
#from scipy import sparse as sps
import tempfile
import cPickle
import mmap
try:
    from _numpypy import multiarray as npy
except:
    import numpy as npy
#import numpy as npy

try:
    from numba import njit
except:
    njit = lambda x: x

from array import array
#import math
from random import random
from time import time

# mmap based array
class darray:

    def __init__(self, fn, size, dtype='float32', chk=10**6):
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
        self.chk = chk

        self.buf = mmap.mmap(self.f.fileno(), self.stride *
                             self.size, prot=mmap.ACCESS_WRITE)

        #print 'buf length', len(self.buf), size, len(self.buf) / self.stride
        self.dat = npy.frombuffer(self.buf, self.dtype)

    # resize the array
    def resize(self, size=-1):
        L = size > 0 and size * self.stride - \
            1 or self.stride * (self.dat.size + self.chk) - 1
        L = int(L)
        self.f.seek(L)
        self.f.write('\x00')
        self.f.close()
        self.f = open(self.fn, "r+b")
        L += 1
        self.buf = mmap.mmap(self.f.fileno(), L, prot=mmap.ACCESS_WRITE)
        self.dat = npy.frombuffer(self.buf, self.dtype)


# a x b = z
def csrmm0(x0, y0, z0, x1, y1, z1, z2n='tmp'):

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


# int: int hash table
class ht:
    def __init__(self, size=100):
        self.size = size
        self.key = array('i', [0]) * self.size
        self.value = array('i', [0]) * self.size
        self.ptr = 0


    def __getitem__(self, i):
        self.value


# a0 x a1
def csrmm(r0, c0, d0, r1, c1, d1, fn='tmp'):
    # r: row index 
    # c: col index
    # d: data

    R = r0.size
    C = (c0.size + c1.size) * 10
    x2d = darray(fn + '_r.npz', R, dtype=x0.dtype)
    y2d = darray(fn + '_c.npz', C, dtype=y0.dtype)
    z2d = darray(fn + '_d.npz', C, dtype=z0.dtype)

    r2, c2, d2 = x2d.dat, y2d.dat, z2d.dat
    # values of ith row
    di = npy.zeros(R, z0.dtype)

    # set
    visit = set()

    for i in xrange(r0.size-1):
        k00, k01 = r0[i:i+2]
        if k00 == k01:
            continue

        for k0i in xrange(k00, k01):
            k = c0[k0i]
            a0ik = d0[k0i]

            k10, k11 = r1[k:k+1]
            for k1i in xrange(k10, k11):
                j = c1[k1i]
                a1kj = d1[k1i]
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


class nd:
    def __init__(self, shape, dtype='f'):
        self.shape = shape
        if dtype == 'float32':
            self.dtype = 'f'
        elif dtype == 'float64':
            self.dtype = 'd'
        else:
            self.dtype = dtype
        N = 1
        for i in shape:
            N *= i
        self.N = N
        self.data = [array(dtype, [0]) * self.shape[1] for elem in xrange(shape[0])]

    def __getitem__(self, (i, j)):
        return self.data[i][j]

    def __setitem__(self, (i, j), v):
        self.data[i][j] = v

    def shape(self):
        return shape


def vec(x, y):
    n = len(x)
    d = 0
    for i in xrange(n):
        d += x[i] * y[i]
    return d


def dot(x, y):
    r, d = x.shape
    #r, d = len(x), len(x[1])
    d, c = y.shape
    #d, c = len(y), len(y[1])

    #z = npy.empty((r, c), dtype='float32')
    #z = [array('d', [0]) * c for elem in xrange(r)]
    z = nd((r, c), 'd')
    for i in xrange(r):
        for k in xrange(d):
            xik = x.data[i][k]
            #xik = x[i][k]
            zi = z.data[i]
            yk = y.data[k]
            for j in xrange(c):
                #z[i][j] += xik * y[k][j]
                #z[i, j] += xik * y[k, j]
                #ori = z[i, j]
                #orj = zi[j]
                zi[j] += xik * yk[j]
                #z.data[i][j] += xik * y.data[k][j]
                #if z[i, j] != zi[j]:
                #    print 'fuckyou'
                #else:
                #    print z[i, j], zi[j], ori, orj

                #z[i][j] += x[i][k] * y[k][j]

    return z

@njit
def ndot(x, y):
    r, d = x.shape
    #r, d = len(x), len(x[1])
    d, c = y.shape
    #d, c = len(y), len(y[1])

    z = npy.empty((r, c), dtype=x.dtype)
    #z = nd((r, c), 'float32')
    for i in xrange(r):
        for k in xrange(d):
            #xik = x[i, k]
            #xik = x[i][k]
            #zi = z[i]
            #yk = y[k]
            for j in xrange(c):
                z[i, j] += x[i, k] * y[k, j]
                #z[i, j] += xik * y[k, j]
                #zi[j] += xik * yk[j]

    return z



if __name__ == '__main__':
    n = int(eval(sys.argv[1]))
    #x = darray('test.npy', n)
    try:
        x = npy.random.randn(n, n)
        x.astype('float32')
        #z = npy.empty((n, n), dtype='float32')
        st = time()
        ndot(x, x)
        print 'numba jit', time() - st


    except:
        '''
        x = []
        for i in xrange(n):
            elem = array('f')
            #elem = []
            for j in xrange(n):
                elem.append(random())
            x.append(elem)
        '''
        #x = npy.empty((n, n), dtype='float32')
        x = nd((n, n))
        for i in xrange(n):
            for j in xrange(n):
                x[i,j] = random()
      
        #z = [array('d', [0])*n for elem in xrange(n)]
        #z = [[0]*n for elem in xrange(n)]

        st = time()
        dot(x, x)

        '''
        x = npy.empty((n, n), dtype='float32')
        z = npy.empty((n, n), dtype='float32')

        st = time()
        ndot(x, x, z)
        '''

        print 'pypy', time() - st

