#!usr/bin/env python
#import scipy as np
import scipy as sp
import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy import stats
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
#import numpy as np
from scipy import sparse as sps
import tempfile
import cPickle

from threading import Thread
from sklearn.externals.joblib import Parallel, delayed


import mmap
try:
    from _numpypy import multiarray as npy
except:
    npy = np


try:
    import sharedmem as sm
except:
    sm = np


# try:
#    import cupy as cp
#    has_cupy = has_gpu = True
# except:
#    cp = np
#    has_cupy = has_gpu = False

try:
    import pyculib
    has_gpu = True
except:
    has_gpu = False


import multiprocessing as mp
from multiprocessing import Manager, Array

try:
    from numba import jit, njit, cuda
except:
    njit = jit = lambda x: x


# the sparse matrix add matrix on gpu
if has_gpu:
    # if 1:
    def csrgeam_ez(matA, matB, alpha=1, beta=1, transA='N', transB='N', descrA=None,
                   descrB=None, descrC=None, clf=None):

        if type(clf) == type(None):
            clf = pyculib.sparse.Sparse()

        tmpdescr = pyculib.sparse.Sparse().matdescr()
        descrA = descrA or tmpdescr
        descrB = descrB or tmpdescr
        descrC = descrC or tmpdescr

        dtype = matA.dtype
        m, ka = matA.shape
        kb, n = matB.shape
        if ka != kb:
            raise ValueError("incompatible matrices")
        k = ka

        indptrC = pyculib.cuda.device_array(m + 1, dtype='int32')
        nnz = clf.XcsrgeamNnz(m, n, descrA, matA.nnz,
                              matA.indptr, matA.indices, descrB, matB.nnz,
                              matB.indptr, matB.indices, descrC, indptrC)

        if nnz == 0:
            raise ValueError("result is entirely zero")

        dataC = pyculib.cuda.device_array(nnz, dtype=dtype)
        indicesC = pyculib.cuda.device_array(nnz, dtype='int32')
        clf.csrgeam(m, n, alpha, descrA, matA.nnz, matA.data,
                    matA.indptr, matA.indices, beta, descrB, matB.nnz, matB.data,
                    matB.indptr, matB.indices, descrC, dataC, indptrC,
                    indicesC)

        return pyculib.sparse.CudaCSRMatrix().from_attributes(data=dataC, indices=indicesC,
                                                              indptr=indptrC, shape=(
                                                                  m, n),
                                                              dtype=dtype, nnz=nnz)

    #csrgemm_ez = pyculib.sparse.Sparse().csrgemm_ez
else:

    def csrgeam_ez(x, y, clf=None):
        return x + y


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


# worker of thread
class worker(Thread):

    def __init__(self, func, args=()):
        super(worker, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None





# a + b
@njit(fastmath=True, nogil=True, cache=True)
def csram_ms(xr, xc, x, yr, yc, y, zr, zc, z):

    R = xr.size
    D = yr.size
    nnz = z.size
    data = np.zeros(D-1, dtype=x.dtype)
    visit = np.zeros(D, dtype=np.int8)
    index = np.zeros(D, yr.dtype)
    zptr = 0
    for i in xrange(R - 1):

        ks = 0
        # get ith row of a
        ast, aed = xr[i], xr[i+1]
        bst, bed = yr[i], yr[i+1]
        for j in xrange(ast, aed):
            col, val = xc[j], x[j]

            if val != 0:
                pass
            else:
                continue

            data[col] += val
            if visit[col] == 0:
                index[ks] = col
                ks += 1
                visit[col] = 1
            else:
                continue

        for j in xrange(bst, bed):
            col, val = yc[j], y[j]

            if val != 0:
                pass
            else:
                continue

            data[col] += val
            if visit[col] == 0:
                index[ks] = col
                ks += 1
                visit[col] = 1
            else:
                continue

        for pt in xrange(ks):
            col = index[pt]
            visit[col] = 0
            val = data[col]
            if val != 0:
                zc[zptr], z[zptr] = col, val
                zptr += 1
                data[col] = 0

        zr[i+1] = zptr

    #print 'the zptr hello', zptr
    flag = zptr
    return zptr, flag







# a + b
def csram_ez_ms(a, b, cpu=1, prefix=None, tmp_path=None, disk=False):
    assert a.shape == b.shape
    np.nan_to_num(a.data, False)
    np.nan_to_num(b.data, False)

    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    R = xr.shape[0]
    nnz = a.nnz + b.nnz

    if prefix == None:
        tmpfn = tempfile.mktemp('tmp', dir=tmp_path)

    else:
        tmpfn = prefix

    zr = np.zeros(R, xr.dtype)

    if disk:
        #zr = np.memmap(tmpfn + '_zr_ms.npy', mode='w+', shape=R,  dtype=xr.dtype)
        zc = np.memmap(tmpfn + '_zc_ms.npy', mode='w+', shape=nnz,  dtype=xc.dtype)
        z = np.memmap(tmpfn + '_z_ms.npy', mode='w+', shape=nnz, dtype=x.dtype)

    else:
        #zr = np.zeros(R, xr.dtype)
        zc = np.empty(nnz,  dtype=xc.dtype)
        z = np.empty(nnz, dtype=x.dtype)


    zptr, flag = csram_ms(xr, xc, x, yr, yc, y, zr, zc, z)

    # truncate
    if disk:
        print 'before truncate', zc.size, zptr
        zc.flush()
        N = zptr * zc.strides[0]
        fn = zc.filename
        _dtype = zc.dtype
        del zc
        f = open(fn, 'r+')
        f.truncate(N)
        f.close()
        zc = np.memmap(fn, mode='r+', dtype=_dtype)
        print 'after truncate', zc.size, zptr


        z.flush()
        N = zptr * z.strides[0]
        fn = z.filename
        _dtype = z.dtype
        del z
        f = open(fn, 'r+')
        f.truncate(N)
        f.close()
        z = np.memmap(fn, mode='r+', dtype=_dtype)


    shape = a.shape
    if disk:
        zmtx = sparse.csr_matrix(shape, dtype=z.dtype)
        zmtx.indptr, zmtx.indices, zmtx.data = zr, zc, z

        save_npz_disk(zmtx, tmpfn + '.npy')
        del zmtx
        os.system('rm %s_z*_ms.npy'%tmpfn)
        zmtx = load_npz_disk(tmpfn + '.npy') 



    else:
        indptr = zr
        indices = zc
        data = z
        zmtx = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=z.dtype)

    gc.collect()

    return zmtx




@njit
def resize(a, new_size):
    new = np.empty(new_size, a.dtype)
    new[:a.size] = a
    return new


@njit
def resize_mmp(a, new_size):
    new = np.asarray(np.memmap('tmp.npy', mode='w+',
                               shape=new_size, dtype=a.dtype), dtype=a.dtype)
    new[:a.size] = a
    return new


# csr matrix by matrix
# original version
@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ori(xr, xc, x, yr, yc, y, visit):

    R = xr.shape[0]
    D = yr.shape[0]
    nnz = int(1. * x.size * y.size / (D - 1))
    #nnz = x.size + y.size
    print 'nnz size', nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    n_size = nnz
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(
        n_size, xc.dtype), np.empty(n_size, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)
    # print 'zr init', zr[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]

            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]
                data[y_col] += x_val * y_val

                if visit[y_col] == 0:
                    visit[y_col] = 1
                    index[ks] = y_col
                    ks += 1
                    flag += 3
                #nz += 1
                flag += 3
            flag += 2

        zend = zr[i] + nz
        if zend > n_size:
            n_size += nnz
            print 'resize sparse matrix', n_size
            zc = resize(zc, n_size)
            z = resize(z, n_size)
            flag += 2

        for pt in xrange(ks):
            idx = index[pt]
            #mx_col = max(mx_col, idx)
            val = data[idx]
            visit[idx] = 0
            if val > 0:
                zc[zptr], z[zptr] = idx, val
                zptr += 1
                data[idx] = 0
                flag += 5

            flag += 1

        zr[i + 1] = zptr

    return zr, zc[:zptr], z[:zptr], flag
    #zmtx = sps.csr_matrix((z[:zptr], zc[:zptr], zr), shape=(a.shape[0], b.shape[1]))
    # return zmtx


# memory save version
@njit(fastmath=True, nogil=True, cache=True)
def csrmm_msav(xr, xc, x, yr, yc, y, visit):

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    #nnz = chk
    nnz = min(max(int(1. * x.size * y.size / (D - 1)), chk * 33), chk * 50)
    print 'nnz size', chk, nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(
        nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)
    # print 'zr init', zr[:5], zc[:5], z[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    #index = np.zeros(yr.size, yr.dtype)
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        i_sz = index.size
        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2
            for j in xrange(jst, jed):
                # for j in prange(jst, jed):
                y_col, y_val = yc[j], y[j]
                # print 'before', ks, len(index), i_sz
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    if ks < i_sz:
                        index[ks] = y_col
                    else:
                        i_sz += (jed - jst) * 2
                        index = resize(index, i_sz)
                        index[ks] = y_col
                    ks += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3
                # print 'end', ks, len(index), i_sz

            #print(k, jst, jed, len(yr))

        zend = zr[i] + nz
        if zend > nnz:
            print'resize estimate', nnz, nnz + chk * 15, nnz * R / i
            #nnz = max(chk+nnz, R/i*nnz)
            nnz += chk * 15

            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
            # for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0
                flag += 3

            flag += 1

        zr[i + 1] = zptr
    print 'the zptr', zptr
    return zr, zc[:zptr], z[:zptr], flag


@njit(fastmath=True, nogil=True, cache=True)
def csrmm_msav1(xr, xc, x, yr, yc, y):

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    nnz = chk
    print 'nnz size', nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(
        nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)
    print 'zr init', zr[:5], zc[:5], z[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    #index = np.zeros(yr.size, yr.dtype)
    index = np.zeros(yr.size, yr.dtype)
    index_tmp = np.zeros(yr.size, yr.dtype)
    index_mg = np.zeros(yr.size, yr.dtype)

    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        index[0], index_tmp[0] = -1, -1
        nz = 0
        ks = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2
            ks_tmp = 0
            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    index_tmp[ks_tmp] = y_col
                    ks_tmp += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3
                # print 'end', ks, len(index), i_sz
            if index[0] == -1:
                index, index_tmp = index_tmp, index
                ks = ks_tmp

            if index_tmp[0] != -1:
                #ks = merge_index(index, index_tmp)
                ks_mg = p0 = p1 = 0
                while p0 < ks and p1 < ks_tmp:
                    idx0 = index[p0]
                    idx1 = index_tmp[p1]
                    if idx0 < idx1:
                        index_mg[ks_mg] = idx0
                        p0 += 1
                    else:
                        p1 += 1
                        index_mg[ks_mg] = idx1
                    if ks_mg <= 0 or index_mg[ks_mg - 1] != index_mg[ks_mg]:
                        ks_mg += 1
                    else:
                        continue
                index, index_mg = index_mg, index
                ks, ks_mg = ks_mg, ks
            #print(k, jst, jed, len(yr))

        print index[:ks + 1]
        zend = zr[i] + nz
        if zend > nnz:
            nnz += chk
            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
            # for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0
                flag += 3

            flag += 1

        zr[i + 1] = zptr

    return zr, zc[:zptr], z[:zptr], flag
    #zmtx = sps.csr_matrix((z[:zptr], zc[:zptr], zr), shape=(a.shape[0], b.shape[1]))
    # return zmtx


@njit(fastmath=True, nogil=True, cache=True)
def csrmm_msav2(xr, xc, x, yr, yc, y, visit):

    R = xr.shape[0]
    D = yr.shape[0]
    chk = (x.size + y.size)
    #nnz = int(1. * x.size * y.size / (D-1))
    nnz = chk
    print 'nnz size', nnz
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(
        nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)
    #visit = np.zeros(D-1, 'int8')
    # print 'zr init', zr[:5], zc[:5], z[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    #index = np.zeros(yr.size, yr.dtype)
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        nz = 0
        ks = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2
            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0 and visit[y_col] == 0:
                    # if visit[y_col] == 0:
                    index[ks] = y_col
                    visit[y_col] = 1
                    ks += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3
                # print 'end', ks, len(index), i_sz

        zend = zr[i] + nz
        if zend > nnz:
            print'resize estimate', nnz, nnz + chk, nnz * R / i
            nnz = max(chk + nnz, R / i * nnz)
            #nnz += chk
            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
            # for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = visit[y_col] = 0
                flag += 3

            flag += 1

        zr[i + 1] = zptr

    return zr, zc[:zptr], z[:zptr], flag



# memory saved version
@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ms0(xr, xc, x, yr, yc, y, zr, zc, z, visit):

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    #nnz = chk
    #nnz = min(max(int(1. * x.size * y.size / (D - 1)), chk * 33), chk * 50)
    nnz = z.size
    #print 'nnz size', chk, nnz
    #zr, zc, z = np.zeros(R, xr.dtype), np.empty(nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)

    #visit1 = np.zeros(yr.size, dtype=x.dtype)

    #zr = np.zeros(R, xr.dtype)
    #zc = np.asarray(np.memmap('zc_tmp.npy', mode='w', shape=nnz,  dtype=xc.dtype))
    #z = np.asarray(np.memmap('zc_tmp.npy', mode='w', shape=nnz, dtype=x.dtype))
    #fq = np.memmap('zc_tmp.npy', mode='w', shape=nnz, dtype=x.dtype)


    # print 'zr init', zr[:5], zc[:5], z[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    #index = np.zeros(yr.size, yr.dtype)
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        i_sz = index.size
        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2
            for j in xrange(jst, jed):
                # for j in prange(jst, jed):
                y_col, y_val = yc[j], y[j]
                # print 'before', ks, len(index), i_sz
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    if ks < i_sz:
                        index[ks] = y_col
                    else:
                        i_sz += (jed - jst) * 2
                        index = resize(index, i_sz)
                        index[ks] = y_col
                    ks += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3
                # print 'end', ks, len(index), i_sz

            #print(k, jst, jed, len(yr))

        zend = zr[i] + nz
        if zend > nnz:
            print'resize estimate', nnz, nnz + chk * 15, nnz * R / i
            #nnz = max(chk+nnz, R/i*nnz)
            nnz += chk * 15

            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
            # for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0
                flag += 3

            flag += 1

        zr[i + 1] = zptr

    #zr.flush()
    #zc.flush()
    #z.flush()
    print 'the zptr', zptr
    #return zr, zc[:zptr], z[:zptr], flag
    return zptr, flag



@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ms_1pass_fast(xr, xc, x, yr, yc, y):

    R = xr.shape[0]
    D = yr.shape[0]
    visit = np.zeros(yr.size, dtype=np.int8)
    index = np.zeros(yr.size, yr.dtype)
    zptr = 0
    for i in xrange(R-1):

        # get ith row of a
        kst, ked = xr[i], xr[i+1]
        if kst == ked:
            continue

        ks = 0
        for k in xrange(kst, ked):
            x_col = xc[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col+1]
            if jst == jed:
                continue

            for j in xrange(jst, jed):
                y_col = yc[j]
                if visit[y_col] == 0:
                    index[ks] = y_col
                    ks += 1
                    visit[y_col] = 1
                else:
                    continue
    
        for pt in xrange(ks):
            y_col = index[pt]
            visit[y_col] = 0

        zptr += ks

    return zptr


@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ms_1pass(xr, xc, x, yr, yc, y):

    R = xr.size
    D = yr.size
    visit = np.zeros(D, dtype=np.int8)
    index = np.zeros(D, yr.dtype)
    data = np.zeros(D, y.dtype)

    zptr = 0
    for i in xrange(R-1):

        # get ith row of a
        kst, ked = xr[i], xr[i+1]
        if kst == ked:
            continue

        ks = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col+1]
            if jst == jed:
                continue

            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]
                data[y_col] += x_val * y_val
                if visit[y_col] == 0:
                    index[ks] = y_col
                    ks += 1
                    visit[y_col] = 1
                else:
                    continue
    
        for pt in xrange(ks):
            y_col = index[pt]
            visit[y_col] = 0
            if data[y_col] != 0:
                data[y_col] = 0
                zptr += 1

    return zptr






@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ms_2pass0(xr, xc, x, yr, yc, y, zr, zc, z):

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    #nnz = chk
    #nnz = min(max(int(1. * x.size * y.size / (D - 1)), chk * 33), chk * 50)
    nnz = z.size
    #print 'nnz size', chk, nnz
    #zr, zc, z = np.zeros(R, xr.dtype), np.empty(nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)
    visit = np.zeros(yr.size, dtype=np.int8)
    index = np.zeros(yr.size, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        i_sz = index.size
        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2
            for j in xrange(jst, jed):
                # for j in prange(jst, jed):
                y_col, y_val = yc[j], y[j]
                # print 'before', ks, len(index), i_sz
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    if visit[y_col] == 0:
                        index[ks] = y_col
                        ks += 1
                        visit[y_col] = 1
                        flag += 2

                    data[y_col] = y_col_val
                    flag += 3
                else:
                    continue
                # print 'end', ks, len(index), i_sz

            #print(k, jst, jed, len(yr))

        zend = zr[i] + nz
        if zend > nnz:
            print'resize estimate', nnz, nnz + chk * 15, nnz * R / i
            #nnz = max(chk+nnz, R/i*nnz)
            nnz += chk * 15

            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
            # for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = visit[y_col] = 0
                flag += 3

            flag += 1

        zr[i + 1] = zptr

    print 'the zptr hello', zptr
    return zptr, flag


@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ms_2pass1(xr, xc, x, yr, yc, y, zr, zc, z):

    R = xr.shape[0]
    D = yr.shape[0]
    nnz = z.size
    data = np.zeros(D - 1, dtype=x.dtype)
    visit = np.zeros(yr.size, dtype=np.int8)
    index = np.zeros(yr.size, yr.dtype)
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        #i_sz = index.size
        ks = 0
        #nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]
            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            #nz += jed - jst
            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    if visit[y_col] == 0:
                        index[ks] = y_col
                        ks += 1
                        visit[y_col] = 1

                    data[y_col] = y_col_val
                else:
                    continue
    
        for pt in xrange(ks):
            y_col = index[pt]
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = visit[y_col] = 0


        zr[i + 1] = zptr

    #print 'the zptr hello', zptr
    flag = zptr
    return zptr, flag




@njit(fastmath=True, nogil=True, cache=True)
def csrmm_ms_2pass(xr, xc, x, yr, yc, y, zr, zc, z):

    R = xr.size
    D = yr.size
    nnz = z.size
    data = np.zeros(D-1, dtype=x.dtype)
    visit = np.zeros(yr.size, dtype=np.int8)
    index = np.zeros(yr.size, yr.dtype)
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        #i_sz = index.size
        ks = 0
        #nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]

            if x_val != 0:
                pass
            else:
                continue

            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            #nz += jed - jst
            for j in xrange(jst, jed):
                y_col, y_val = yc[j], y[j]

                if y_val != 0:
                    pass
                else:
                    continue

                data[y_col] += x_val * y_val
                if visit[y_col] == 0:
                    index[ks] = y_col
                    ks += 1
                    visit[y_col] = 1
                else:
                    continue
    
        for pt in xrange(ks):
            y_col = index[pt]
            visit[y_col] = 0
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0


        zr[i+1] = zptr

    #print 'the zptr hello', zptr
    flag = zptr
    return zptr, flag






def csrmm_ez_ms(a, b, mm='msav', cpu=1, prefix=None, tmp_path=None, disk=False):
    np.nan_to_num(a.data, False)
    np.nan_to_num(b.data, False)

    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    R = xr.shape[0]
    D = yr.shape[0]
    nnz = csrmm_ms_1pass_fast(xr, xc, x, yr, yc, y)
    print '1st pass', nnz


    if prefix == None:
        tmpfn = tempfile.mktemp('tmp', dir=tmp_path)

    else:
        tmpfn = prefix

    zr = np.zeros(R, xr.dtype)
    if disk:
        #zr = np.memmap(tmpfn + '_zr_ms.npy', mode='w+', shape=R,  dtype=xr.dtype)
        zc = np.memmap(tmpfn + '_zc_ms.npy', mode='w+', shape=nnz,  dtype=xc.dtype)
        z = np.memmap(tmpfn + '_z_ms.npy', mode='w+', shape=nnz, dtype=x.dtype)

    else:
        #zr = np.zeros(R, xr.dtype)
        zc = np.empty(nnz,  dtype=xc.dtype)
        z = np.empty(nnz, dtype=x.dtype)

    print 'a nnz', a.nnz, 'b nnz', b.nnz

    zptr, flag = csrmm_ms_2pass(xr, xc, x, yr, yc, y, zr, zc, z)

    # truncate
    if disk:
        print 'before truncate', zc.size, zptr
        zc.flush()
        N = zptr * zc.strides[0]
        fn = zc.filename
        _dtype = zc.dtype
        del zc
        f = open(fn, 'r+')
        f.truncate(N)
        f.close()
        zc = np.memmap(fn, mode='r+', dtype=_dtype)
        print 'after truncate', zc.size, zptr


        z.flush()
        N = zptr * z.strides[0]
        fn = z.filename
        _dtype = z.dtype
        del z
        f = open(fn, 'r+')
        f.truncate(N)
        f.close()
        z = np.memmap(fn, mode='r+', dtype=_dtype)


    shape = (a.shape[0], b.shape[1])
    if disk:
        zmtx = sparse.csr_matrix(shape, dtype=z.dtype)
        zmtx.indptr, zmtx.indices, zmtx.data = zr, zc, z
        save_npz_disk(zmtx, tmpfn + '.npy')
        del zmtx
        os.system('rm %s_z*_ms.npy'%tmpfn)
        zmtx = load_npz_disk(tmpfn + '.npy') 

    else:
        indptr = zr
        indices = zc
        data = z
        zmtx = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=z.dtype)

    gc.collect()

    return zmtx





def csrmm_ez_ms1(a, b, mm='msav', cpu=1, prefix=None, tmp_path=None, disk=False):
    np.nan_to_num(a.data, False)
    np.nan_to_num(b.data, False)

    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    R = xr.shape[0]
    D = yr.shape[0]
    #chk = x.size + y.size
    #nnz = chk
    #nnz = min(max(int(1. * x.size * y.size / (D - 1)), chk * 33), chk * 50)
    nnz = csrmm_ms_1pass(xr, xc, x, yr, yc, y, zr, zc, z)


    if prefix == None:
        #tmpfn = tempfile.mktemp('tmp', dir='./tmp/')
        tmpfn = tempfile.mktemp('tmp', dir=tmp_path)

    else:
        #tmpfn = tempfile.mktemp(prefix, dir=tmp_path)
        tmpfn = prefix

    print 'the_tmpfn_fk', tmp_path, prefix




    zr = np.zeros(R, xr.dtype)

    #zc = np.asarray(np.memmap('zc_tmp.npy', mode='w+', shape=nnz,  dtype=xc.dtype))
    zc = np.memmap(tmpfn + '_zc_ms.npy', mode='w+', shape=nnz,  dtype=xc.dtype)

    #z = np.asarray(np.memmap('z_tmp.npy', mode='w+', shape=nnz, dtype=x.dtype))
    z = np.memmap(tmpfn + '_z_ms.npy', mode='w+', shape=nnz, dtype=x.dtype)


    print 'a nnz', a.nnz, 'b nnz', b.nnz
    st = time()
    # if cpu > 1 and x.size > 5e8:
    #    csrmm = csrmm_sp
    # if cpu > 1 and x.size < 5e8:
    # if cpu > 1:
    csrmm = csrmm_ms

    nnzs = x.size + y.size
    #visit = np.zeros(yr.size, 'int8')
    #Zr, Zc, Z, flag = csrmm(xr, xc, x, yr, yc, y, zr, zc, z, visit)
    #zptr, flag = csrmm(xr, xc, x, yr, yc, y, zr, zc, z, visit)
    zptr, flag = csrmm(xr, xc, x, yr, yc, y, zr, zc, z)


    # truncate
    print 'before truncate', zc.size, zptr
    zc.flush()
    N = zptr * zc.strides[0]
    fn = zc.filename
    _dtype = zc.dtype
    del zc
    f = open(fn, 'r+')
    f.truncate(N)
    f.close()
    zc = np.memmap(fn, mode='r+', dtype=_dtype)
    print 'after truncate', zc.size, zptr


    z.flush()
    N = zptr * z.strides[0]
    fn = z.filename
    _dtype = z.dtype
    del z
    f = open(fn, 'r+')
    f.truncate(N)
    f.close()
    z = np.memmap(fn, mode='r+', dtype=_dtype)


    #if type(z) != type(None):
    #    zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    #else:
    #    zmtx = sps.csr_matrix((a.shape[0], b.shape[1]), dtype=a.dtype)


    #zc.flush()
    #z.flush()


    shape = (a.shape[0], b.shape[1])
    #zmtx = sparse.csr_matrix(shape, dtype=z.dtype)

    #zmtx.indptr, zmtx.indices, zmtx.data = zr, zc, z

    #zmtx = sps.csr_matrix((z, zc, zr), shape=shape)
    #print 'Zr, Zc, Z is', zmtx.indptr, zmtx.indices, zmtx.data 
    #zmtx.eliminate_zeros()

    if disk:
        zmtx = sparse.csr_matrix(shape, dtype=z.dtype)
        zmtx.indptr, zmtx.indices, zmtx.data = zr, zc, z
    else:
        indptr = np.array(zr)
        indices = np.array(zc)
        data = np.array(z)
        zmtx = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=z.dtype)
        #zr._mmap.close()
        zc._mmap.close()
        z._mmap.close()
        del zr, zc, z

    gc.collect()

    return zmtx






def csrmm_ez_ms0(a, b, mm='msav', cpu=1, prefix=None, tmp_path=None, disk=False):
    np.nan_to_num(a.data, False)
    np.nan_to_num(b.data, False)

    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    nnz = chk
    nnz = min(max(int(1. * x.size * y.size / (D - 1)), chk * 33), chk * 50)

    if prefix == None:
        #tmpfn = tempfile.mktemp('tmp', dir='./tmp/')
        tmpfn = tempfile.mktemp('tmp', dir=tmp_path)

    else:
        #tmpfn = tempfile.mktemp(prefix, dir=tmp_path)
        tmpfn = prefix

    print 'the_tmpfn_fk', tmp_path, prefix




    zr = np.zeros(R, xr.dtype)

    #zc = np.asarray(np.memmap('zc_tmp.npy', mode='w+', shape=nnz,  dtype=xc.dtype))
    #zc = np.memmap(tmpfn + '_zc_ms.npy', mode='w+', shape=nnz,  dtype=xc.dtype)

    #z = np.asarray(np.memmap('z_tmp.npy', mode='w+', shape=nnz, dtype=x.dtype))
    #z = np.memmap(tmpfn + '_z_ms.npy', mode='w+', shape=nnz, dtype=x.dtype)

    with np.memmap(tmpfn + '_zc_ms.npy', mode='w+', shape=nnz,  dtype=xc.dtype) as zc, np.memmap(tmpfn + '_z_ms.npy', mode='w+', shape=nnz, dtype=x.dtype) as z:


        print 'a nnz', a.nnz, 'b nnz', b.nnz
        st = time()
        # if cpu > 1 and x.size > 5e8:
        #    csrmm = csrmm_sp
        # if cpu > 1 and x.size < 5e8:
        # if cpu > 1:
        csrmm = csrmm_ms

        nnzs = x.size + y.size
        #visit = np.zeros(yr.size, 'int8')
        #Zr, Zc, Z, flag = csrmm(xr, xc, x, yr, yc, y, zr, zc, z, visit)
        #zptr, flag = csrmm(xr, xc, x, yr, yc, y, zr, zc, z, visit)
        zptr, flag = csrmm(xr, xc, x, yr, yc, y, zr, zc, z)


        # truncate
        print 'before truncate', zc.size, zptr
        zc.flush()
        N = zptr * zc.strides[0]
        fn = zc.filename
        _dtype = zc.dtype
        del zc
        f = open(fn, 'r+')
        f.truncate(N)
        f.close()
        zc = np.memmap(fn, mode='r+', dtype=_dtype)
        print 'after truncate', zc.size, zptr


        z.flush()
        N = zptr * z.strides[0]
        fn = z.filename
        _dtype = z.dtype
        del z
        f = open(fn, 'r+')
        f.truncate(N)
        f.close()
        z = np.memmap(fn, mode='r+', dtype=_dtype)


        shape = (a.shape[0], b.shape[1])

        if disk:
            zmtx = sparse.csr_matrix(shape, dtype=z.dtype)
            zmtx.indptr, zmtx.indices, zmtx.data = zr, zc, z
        else:
            indptr = np.array(zr)
            indices = np.array(zc)
            data = np.array(z)
            zmtx = sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=z.dtype)
            zr._mmap.close()
            zc._mmap.close()
            z._mmap.close()
            del zr, zc, z
            gc.collect()


    return zmtx







# parallel version of csrmm
#@njit(fastmath=True, nogil=True, cache=True)
@njit(nogil=True, cache=True, fastmath=True)
def csrmm_sp(Xr, Xc, X, yr, yc, y, xrst, xred, cpu=1):

    xr = np.empty(xred + 1 - xrst, Xr.dtype)
    xr[:] = Xr[xrst:xred + 1]
    xr -= xr[0]
    xcst, xced = Xr[xrst], Xr[xred]
    xc = Xc[xcst: xced]
    x = X[xcst: xced]
    # print 'xrst %d xred %d xcst %d xced %d'%(xrst, xred, xcst, xced)

    R = xr.shape[0]
    D = yr.shape[0]
    chk = x.size + y.size
    nnz = chk
    # print 'nnz size %d %d %d %d'%(nnz, x.size, y.size, X.size)
    # zr, zc, z = np.zeros(R, 'int32'), np.empty(nnz*5, 'int32'), np.empty(nnz*5, dtype=x.dtype)
    zr, zc, z = np.zeros(R, xr.dtype), np.empty(
        nnz, xc.dtype), np.empty(nnz, dtype=x.dtype)
    data = np.zeros(D - 1, dtype=x.dtype)
    # print 'zr init', zr[:5]

    # hash table
    #visit = np.zeros(yr.size, 'int8')
    chk1 = yr.size // cpu
    nnz1 = chk1
    index = np.zeros(nnz1, yr.dtype)
    flag = 0
    zptr = 0
    for i in xrange(R - 1):

        # get ith row of a
        kst, ked = xr[i], xr[i + 1]
        if kst == ked:
            zr[i + 1] = zr[i]
            continue

        ks = 0
        nz = 0
        for k in xrange(kst, ked):
            x_col, x_val = xc[k], x[k]

            # get row of b
            jst, jed = yr[x_col], yr[x_col + 1]
            if jst == jed:
                continue

            nz += jed - jst
            flag += 2

            for j in xrange(jst, jed):
                # for j in prange(jst, jed):
                y_col, y_val = yc[j], y[j]
                y_col_val = data[y_col] + x_val * y_val
                if y_col_val != 0:
                    if ks >= nnz1:
                        nnz1 += chk1
                        index = resize(index, nnz1)

                    index[ks] = y_col
                    ks += 1
                    flag += 2

                data[y_col] = y_col_val
                flag += 3

        zend = zr[i] + nz
        if zend > nnz:
            nnz += chk
            #print('resize sparse matrix', n_size)
            zc = resize(zc, nnz)
            z = resize(z, nnz)
            flag += 2

        for pt in xrange(ks):
            # for pt in prange(ks):
            y_col = index[pt]
            #mx_col = max(mx_col, idx)
            y_col_val = data[y_col]
            if y_col_val != 0:
                zc[zptr], z[zptr] = y_col, y_col_val
                zptr += 1
                data[y_col] = 0
                flag += 3

            flag += 1

        zr[i + 1] = zptr

    #zr += xrst
    # print 'retunr value', zr
    return zr, zc[:zptr], z[:zptr], flag

#@jit(nogil=True)
# def csrmm_sp_wrapper(elem):
#    Xr, Xc, X, yr, yc, y, xrst, xred = elem
#    zr, zc, z, flag = csrmm_sp(Xr, Xc, X, yr, yc, y, xrst, xred)
#    print 'get_value'
#    #return zr, zc, z, flag
#    return sps.csr_matrix((z, zc, zr), dtype=z.dtype)


#csrmm_jit = jit(csrmm)

def csrmm_ez0(a, b, mm='msav', cpu=1):
    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data

    # print 'a shape', a.shape, 'b shape', b.shape, 'yc size', yc[:10],
    # yc.size, yc.max(), yc[-1], 'yr', yr.size, yr[:10]
    print 'a nnz', a.nnz, 'b nnz', b.nnz

    st = time()
    # if use_jit:
    #    zr, zc, z, flag = csrmm_jit(xr, xc, x, yr, yc, y)
    # else:
    #    zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)
    if cpu > 1:
        csrmm = csrmm_sp
    elif mm == 'msav':
        csrmm = csrmm_msav
    elif mm == 'ori':
        csrmm = csrmm_ori
    else:
        raise SystemExit()

    # if cpu <= 1:
    # close threads
    if 1:
        zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y)
        #zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    else:
        print 'using threads'
        N, D = a.shape
        step = N // (cpu * 4) + 1
        threads = []
        for i in xrange(0, N, step):
            start, end = i, min(i + step, N)
            t = worker(csrmm_sp, (xr, xc, x, yr, yc, y, start, end, cpu))
            t.start()
            threads.append(t)

        #res = []
        #offset = 0
        # for t in threads:
        tmpfn = tempfile.mkdtemp()
        #_ozr = open('./tmp_zr.npy', 'wb')
        #_ozc = open('./tmp_zc.npy', 'wb')
        #_oz = open('./tmp_z.npy', 'wb')
        _ozr = open(tmpfn + '_zr.npy', 'wb')
        _ozc = open(tmpfn + '_zc.npy', 'wb')
        _oz = open(tmpfn + '_z.npy', 'wb')

        flag = -1
        for i in xrange(0, N, step):
            start, end = i, min(i + step, N)
            t = threads[i // step]
            t.join()
            zr, zc, z, flag0 = t.get_result()
            if flag != -1:
                zr = zr[1:]
                zr += flag
            flag = zr[-1]
            _ozr.write(np.getbuffer(zr))
            _ozc.write(np.getbuffer(zc))
            _oz.write(np.getbuffer(z))

            #new_shape = (end-start, b.shape[1])
            # print 'new shape', new_shape, z
            #res.append(sps.csr_matrix((z, zc, zr), shape=new_shape, dtype=z.dtype))
            # print 'res', res
            #flag += flag0
            #flag += zr.size

        _ozr.close()
        _ozc.close()
        _oz.close()
        #zr = np.memmap('./tmp_zr.npy', dtype=xr.dtype)
        #zc = np.memmap('./tmp_zc.npy', dtype=xc.dtype)
        #z = np.memmap('./tmp_z.npy', dtype=x.dtype)
        try:
            zr = np.memmap(tmpfn + '_zr.npy', dtype=xr.dtype)
            zc = np.memmap(tmpfn + '_zc.npy', dtype=xc.dtype)
            z = np.memmap(tmpfn + '_z.npy', dtype=x.dtype)
            zr, zc, z = map(np.array, [zr, zc, z])
            #os.system('rm ./tmp_zr.npy ./tmp_zc.npy ./tmp_z.npy')
            os.system('rm %s_z*.npy' % tmpfn)
        except:
            zr = zc = z = None

        # print res
        #zmtx = sps.vstack(res)
        #paras = []
        # for i in xrange(0, N, step):
        #    start, end = i, min(i+step, N)
        #    paras.append([xr, xc, x, yr, yc, y, start, end])

        #pool = Pool(cpu)
        #results = pool.map(csrmm_sp_wrapper, paras)
        #results = map(csrmm_sp_wrapper, paras)

       # print 'threads is', threads
        #flag = sum([elem[-1] for elem in threads])

    print 'total operation', flag
    print 'csrmm cpu', time() - st
    # print 'zr min', zr.min(), 'zc max', zr.max(), 'zr size', zr.size
    # print 'zc min', zc.min(), 'zc max', zc.max(), 'zc size', zc.size
    if type(z) != type(None):
        zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    else:
        zmtx = sps.csr_matrix((a.shape[0], b.shape[1]), dtype=a.dtype)

    return zmtx





def csrmm_ez(a, b, mm='msav', cpu=1, prefix=None, tmp_path=None):
    np.nan_to_num(a.data, False)
    np.nan_to_num(b.data, False)

    xr, xc, x = a.indptr, a.indices, a.data
    yr, yc, y = b.indptr, b.indices, b.data
    print 'a nnz', a.nnz, 'b nnz', b.nnz
    st = time()
    # if cpu > 1 and x.size > 5e8:
    #    csrmm = csrmm_sp
    # if cpu > 1 and x.size < 5e8:
    # if cpu > 1:
    if mm == 'scipy':
        return a * b
    elif mm == 'msav':
        print 'using msav'
        csrmm = csrmm_msav
    elif mm == 'ori':
        csrmm = csrmm_ori
    elif mm == 'ms':
        csrmm = csrmm_ms
    else:
        raise SystemExit()

    nnzs = x.size + y.size
    if cpu <= 1 or nnzs <= 1e8:
        # shutdown threads
        # print 'try msav'
        # if 0:
        visit = np.zeros(yr.size, 'int8')
        zr, zc, z, flag = csrmm(xr, xc, x, yr, yc, y, visit)
    else:
        print 'using threads'
        N, D = a.shape
        step = N // (cpu * 4) + 1
        threads = []
        for i in xrange(0, N, step):
            start, end = i, min(i + step, N)
            t = worker(csrmm_sp, (xr, xc, x, yr, yc, y, start, end, cpu))
            t.start()
            threads.append(t)

        if prefix == None:
            #tmpfn = tempfile.mktemp('tmp', dir='./tmp/')
            tmpfn = tempfile.mktemp('tmp', dir=tmp_path)

        else:
            #tmpfn = tempfile.mktemp(prefix, dir=tmp_path)
            tmpfn = prefix

        print 'the_tmpfn_fk', tmp_path, prefix

        _ozr = open(tmpfn + '_zr.npy', 'wb')
        _ozc = open(tmpfn + '_zc.npy', 'wb')
        _oz = open(tmpfn + '_z.npy', 'wb')

        flag = -1
        for i in xrange(0, N, step):
            start, end = i, min(i + step, N)
            t = threads[i // step]
            t.join()
            zr, zc, z, flag0 = t.get_result()
            if flag != -1:
                zr = zr[1:]
                zr += flag
            flag = zr[-1]
            _ozr.write(np.getbuffer(zr))
            _ozc.write(np.getbuffer(zc))
            _oz.write(np.getbuffer(z))

        _ozr.close()
        _ozc.close()
        _oz.close()
        try:
            zr = np.memmap(tmpfn + '_zr.npy', dtype=xr.dtype)
            zc = np.memmap(tmpfn + '_zc.npy', dtype=xc.dtype)
            z = np.memmap(tmpfn + '_z.npy', dtype=x.dtype)
            zr, zc, z = map(np.array, [zr, zc, z])
            os.system('rm -f %s_z*.npy' % tmpfn)
        except:
            zr = zc = z = None

    print 'total operation', flag
    print 'csrmm cpu', time() - st
    if type(z) != type(None):
        zmtx = sps.csr_matrix((z, zc, zr), shape=(a.shape[0], b.shape[1]))
    else:
        zmtx = sps.csr_matrix((a.shape[0], b.shape[1]), dtype=a.dtype)

    return zmtx


# parallel matrix A * B
def mul_chk(m):
    x, start, end, y, shape = m
    X = sparse.csr_matrix(x, shape=shape)
    Y = sparse.csr_matrix(y, shape=shape)

    st = time()
    # print 'shape is', X[start: end].shape, Y.shape
    Z = X[start: end] * Y

    return Z


def Pmul(X, Y, chk=64, cache=10**8, cpu=6):
    N, d = X.shape
    D, M = Y.shape
    assert d == D

    if cpu == 1:
        Z = X * Y

    elif X.nnz < cache and Y.nnz < cache:
        Z = X * Y

    else:
        xi = sm.empty(X.data.shape, 'float32')
        xi[:] = X.data
        xj = sm.empty(X.indices.shape, 'int32')
        xj[:] = X.indices
        xk = sm.empty(X.indptr.shape, 'int32')
        xk[:] = X.indptr

        del X
        x = (xi, xj, xk)

        yi = sm.empty(Y.data.shape, 'float32')
        yi[:] = Y.data
        yj = sm.empty(Y.indices.shape, 'int32')
        yj[:] = Y.indices
        yk = sm.empty(Y.indptr.shape, 'int32')
        yk[:] = Y.indptr

        del Y
        y = (yi, yj, yk)

        step = N // chk
        Z = []
        loc = []
        for i in xrange(0, N, step):
            st, ed = i, min(i + step, N)
            loc.append([x, st, ed, y, (N, d)])
            if len(loc) >= 8:
                st = time()
                tmp = Parallel(n_jobs=cpu)(delayed(mul_chk)(elem)
                                           for elem in loc)
                Z.extend(tmp)
                loc = []
                del tmp
                gc.collect()

        if len(loc) > 0:
            st = time()
            tmp = Parallel(n_jobs=8)(delayed(mul_chk)(elem) for elem in loc)
            Z.extend(tmp)
            del tmp
            gc.collect()

        Z = sparse.vstack(Z)

    return Z


# reorder the matrix
def mat_reorder0(qry, q2n, shape=(10**7, 10**7), csr=False, tmp_path=None, step=4, chunk=5 * 10**7):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    N = shape[0]
    #tsize = os.path.getsize(qry)
    #tstep = tsize // (chunk*12)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(chunk, N // step + 1)
    #block = min(N//step+1, N//tstep+1)
    block = N // step + 1

    # print 'reorder block', block

    # reorder the matrix
    cs = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    # N = len(q2n)
    # shape = (N, N)
    for fn in fns:
        # print 'loading', fns
        g = load_matrix(fn, shape=shape, csr=csr)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    return q2n


# reorder the matrix, put the nodes into diag
def mat_reorder1(qry, q2n, shape=(10**7, 10**7), csr=False, tmp_path=None, step=4, chunk=5 * 10**7):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    N = shape[0]
    NNZ = 0
    # reorder the matrix
    cs = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    for fn in fns:
        g = load_matrix(fn, shape=shape, csr=csr)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)
        NNZ += g.nnz

    NNZ = max(1, NNZ)
    block = N * chunk // NNZ + 1

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    _os = {}
    for fn in fns:
        g = load_matrix(fn, shape=shape, csr=csr)
        xs, ys = g.nonzero()
        zs = g.data
        for i in xrange(xs.shape[0]):
            xo, yo, z = xs[i], ys[i], zs[i]
            x, y = idx[xo], idx[yo]
            out = pack('fff', *[x, y, z])
            xi, yi = x // block, y // block
            key = tmp_path + '/%d_%d.npz' % (xi, yi)
            try:
                _ox = _os[key]
            except:
                _o = open(key + '_reorder', 'wb')
                _os[key] = _o
                _ox = _os[key]

            _ox.write(out)

        # del the old block
        os.system('rm %s' % fn)

    # close the block file
    for _o in _os.values():
        _o.close()

    # convert the new block to csr and get row sum
    row_sum = None
    nnz = 0
    for fn in _os:
        g = load_matrix(fn + '_reorder', shape=shape, csr=False)
        nnz = g.nnz
        tmp = g.sum(0)
        try:
            row_sum += tmp
        except:
            row_sum = tmp

        sparse.save_npz(fn, g)

    return q2n, row_sum, fn, nnz


# reorder the matrix, put the nodes into diag
def mat_reorder3(qry, q2n, shape=(10**7, 10**7), csr=False, tmp_path=None, step=4, chunk=5 * 10**7):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    N = shape[0]
    NNZ = 0
    # reorder the matrix
    cs = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    for fn in fns:
        g = load_matrix(fn, shape=shape, csr=csr)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)
        NNZ += g.nnz

    NNZ = max(1, NNZ)
    block = N * chunk // NNZ + 1

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    _os = {}
    for fn in fns:
        g = load_matrix(fn, shape=shape, csr=csr)
        xs, ys = g.nonzero()
        zs = g.data
        for i in xrange(xs.shape[0]):
            xo, yo, z = xs[i], ys[i], zs[i]
            x, y = idx[xo], idx[yo]
            out = pack('fff', *[x, y, z])
            xi, yi = x // block, y // block
            key = tmp_path + '/%d_%d.npz' % (xi, yi)
            try:
                _ox = _os[key]
            except:
                _o = open(key + '_reorder', 'wb')
                _os[key] = _o
                _ox = _os[key]

            _ox.write(out)

        # del the old block
        os.system('rm %s' % fn)

    # close the block file
    for _o in _os.values():
        _o.close()

    # clean the old file
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if not elem.endswith('_reorder')]
    for fn in fns:
        os.system('rm %s' % fn)

    # convert the new block to csr and get row sum
    print 'after reorder', _os.keys()
    for fn in _os:
        g = load_matrix(fn + '_reorder', shape=shape, csr=False)
        sparse.save_npz(fn, g)

    fns = _os.keys()
    return q2n, fns


def mat_reorder4(qry, q2n, shape=(10**7, 10**7), csr=False, tmp_path=None, step=4, chunk=5 * 10**7, block=None, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    N = shape[0]
    NNZ = 0
    # reorder the matrix

    cs = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    for fn in fns:
        g = load_matrix(fn, shape=shape, csr=csr)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)
        NNZ += g.nnz

    NNZ = max(1, NNZ)
    if block == None:
        block = N * chunk // NNZ + 1

    block = int(block // sqrt(cpu)) + 1

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    flag = 0
    pairs = {}
    for fn in fns:
        g = load_matrix(fn, shape=shape, csr=csr)
        xs, ys = g.nonzero()
        zs = g.data
        for i in xrange(xs.shape[0]):
            flag += 1
            xo, yo, z = xs[i], ys[i], zs[i]
            x, y = idx[xo], idx[yo]
            out = pack('fff', *[x, y, z])
            xi, yi = x // block, y // block
            key = tmp_path + '/%d_%d.npz' % (xi, yi)
            try:
                pairs[key].append(out)
            except:
                pairs[key] = [out]

            if flag % 5000000 == 0:
                for key, vals in pairs.iteritems():
                    _o = open(key + '_reorder', 'a+b')
                    _o.writelines(vals)
                    # for val in vals:
                    #    _o.write(val)
                    _o.close()
                    pairs[key] = []

        # del the old block
        os.system('rm %s' % fn)

    for key, vals in pairs.iteritems():
        _o = open(key + '_reorder', 'a+b')
        _o.writelines(vals)
        # for val in vals:
        #    _o.write(val)

        _o.close()
        pairs[key] = []

    # clean the old file
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if not elem.endswith('_reorder')]
    for fn in fns:
        os.system('rm %s' % fn)

    # convert the new block to csr and get row sum
    print 'after reorder', pairs.keys(), [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('_reorder')]

    for fn in pairs:
        g = load_matrix(fn + '_reorder', shape=shape, csr=False)
        sparse.save_npz(fn, g)
        os.system('rm %s_reorder' % fn)

    fns = pairs.keys()
    return q2n, fns


def mat_reorder(qry, q2n, shape=(10**7, 10**7), csr=False, tmp_path=None, step=4, chunk=5 * 10**7, block=None, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    N = shape[0]
    NNZ = 0
    # reorder the matrix
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    d = max([max(map(int, elem)) for elem in Ns]) + 1
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + os.sep + str(i) + '_' + str(j) + '.npz'
            if os.path.isfile(fn):
                fns.append(fn)

    cs = None
    for fn in fns:
        try:
            g = load_matrix(fn, shape=shape, csr=csr)
        except:
            continue
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)
        NNZ += g.nnz

    NNZ = max(1, NNZ)
    if block == None:
        block = N * chunk // NNZ + 1

    block = int(block // sqrt(cpu)) + 1

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    flag = 0
    pairs = {}
    for fn in fns:
        try:
            g = load_matrix(fn, shape=shape, csr=csr)
        except:
            continue
        xs, ys = g.nonzero()
        zs = g.data
        for i in xrange(xs.shape[0]):
            flag += 1
            xo, yo, z = xs[i], ys[i], zs[i]
            x, y = idx[xo], idx[yo]
            out = pack('fff', *[x, y, z])
            xi, yi = x // block, y // block
            key = tmp_path + '/%d_%d.npz' % (xi, yi)
            try:
                pairs[key].append(out)
            except:
                pairs[key] = [out]

            if flag % 5000000 == 0:
                for key, vals in pairs.iteritems():
                    _o = open(key + '_reorder', 'a+b')
                    _o.writelines(vals)
                    # for val in vals:
                    #    _o.write(val)
                    _o.close()
                    pairs[key] = []

        # del the old block
        os.system('rm %s' % fn)

    for key, vals in pairs.iteritems():
        _o = open(key + '_reorder', 'a+b')
        _o.writelines(vals)
        # for val in vals:
        #    _o.write(val)

        _o.close()
        pairs[key] = []

    # clean the old file
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if not elem.endswith('_reorder')]
    for fn in fns:
        os.system('rm %s' % fn)

    # convert the new block to csr and get row sum
    print 'after reorder', pairs.keys(), [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('_reorder')]

    for fn in pairs:
        g = load_matrix(fn + '_reorder', shape=shape, csr=False)
        sparse.save_npz(fn, g)
        os.system('rm %s_reorder' % fn)

    fns = pairs.keys()
    return q2n, fns


# given a pairwise relationship, this function will convert the qid, sid into numbers
# and split these relationships into small file
def mat_split0(qry, shape=10**7, step=2 * 10**5, tmp_path=None):
    #_os0 = [open('row_%d.bin'%elem, 'wb') for elem in xrange(6*10**6//step)]
    #_os1 = [open('col_%d.bin'%elem, 'wb') for elem in xrange(6*10**6//step)]
    # build the tmp dir
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)

    flag = 0
    q2n = {}
    eye = [0] * shape
    f = open(qry, 'r')
    _oxs, _oys = [], []
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, score = j

        z = float(score)
        if qid not in q2n:
            q2n[qid] = flag
            flag += 1
        if sid not in q2n:
            q2n[sid] = flag
            flag += 1

        x, y = map(q2n.get, [qid, sid])

        out = pack('fff', *[x, y, z])

        xi, yi = x // step, y // step
        try:
            _ox = _oxs[xi]
            _ox.write(out)

        except:
            _o = open(tmp_path + '/%d_row.bin' % xi, 'wb')
            n, m = len(_oxs), xi + 1
            if n < m:
                _oxs.extend([None] * (m - n))

            _oxs[xi] = _o
            _ox = _oxs[xi]
            # print 'xi is', _oxs[xi], _o
            _ox.write(out)

        try:
            _oy = _oys[yi]
            _oy.write(out)
        except:
            _o = open(tmp_path + '/%d_col.bin' % yi, 'wb')
            n, m = len(_oys), yi + 1
            if n < m:
                _oys.extend([None] * (m - n))

            _oys[yi] = _o
            _oy = _oys[yi]
            _oy.write(out)

        # sym
        x, y = y, x
        out = pack('fff', *[x, y, z])
        xi, yi = x // step, y // step

        try:
            _ox = _oxs[xi]
            _ox.write(out)

        except:
            _o = open(tmp_path + '/%d_row.bin' % xi, 'wb')
            n, m = len(_oxs), xi + 1
            if n < m:
                _oxs.extend([None] * (m - n))

            _oxs[xi] = _o
            _ox = _oxs[xi]
            # print 'xi is', _oxs[xi], _o
            _ox.write(out)

        try:
            _oy = _oys[yi]
            _oy.write(out)
        except:
            _o = open(tmp_path + '/%d_col.bin' % yi, 'wb')
            n, m = len(_oys), yi + 1
            if n < m:
                _oys.extend([None] * (m - n))

            _oys[yi] = _o
            _oy = _oys[yi]
            _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(0, len(q2n), step):
        zs = q2n[i:i + step]
        xyzs = [[x, x, z] for x, z in zip(xrange(i, i + step), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f' * len(xyzs), *xyzs)
        j = i // step
        _ox = _oxs[j]
        _ox.write(out)
        _oy = _oys[j]
        _oy.write(out)

    for _o in _oxs + _oys:
        try:
            _o.close()
        except:
            continue

    return q2n


def mat_split1(qry, shape=10**8, step=2 * 10**5, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)

    flag = 0
    q2n = {}
    eye = [0] * shape
    f = open(qry, 'r')
    _os = {}
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, score = j

        z = float(score)
        if qid not in q2n:
            q2n[qid] = flag
            flag += 1
        if sid not in q2n:
            q2n[sid] = flag
            flag += 1

        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])

        xi, yi = x // step, y // step
        kxy = (xi, yi)

        try:
            _oxy = _os[kxy]

        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[kxy] = _o
            _oxy = _os[kxy]

        _oxy.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        kyx = (yi, xi)
        try:
            _oyx = _os[kyx]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[kyx] = _o
            _oyx = _os[kyx]

        _oyx.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(0, len(q2n), step):
        zs = eye[i:i + step]
        xyzs = [[x, x, z] for x, z in zip(xrange(i, i + step), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f' * len(xyzs), *xyzs)
        j = i // step
        _o = _os[(j, j)]
        _o.write(out)

    xy = set()
    for k, _o in _os.items():
        _o.close()
        xy = xy.union(k)

    xy = sorted(xy)
    return q2n, xy


def mat_split2(qry, step=16, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            # q2n[qid] = None
            qid_set.add(qid)
        if sid not in qid_set:
            # q2n[sid] = None
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()

    # print qid_set[:10]
    # print 'get all gene id'
    shape = len(qid_set)
    block = shape // step

    # eye = range(shape)
    # for i, j in zip(q2n, eye):
    for i in xrange(shape):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * shape
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        # xi, yi = x % block, y % block
        xi, yi = x // block, y // block

        try:
            _ox = _os[xi]
        except:
            _o = open(tmp_path + '/%d.npz' % xi, 'wb')
            _os[xi] = _o
            _ox = _os[xi]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[yi]
        except:
            _o = open(tmp_path + '/%d.npz' % yi, 'wb')
            _os[yi] = _o
            _oy = _os[yi]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(shape):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        _o = _os[j]
        _o.write(out)

    '''
    # set eye of matrix:
    chk = 100000
    for i in xrange(0, shape, chk):
        zs = eye[i:i+chk]
        xyzs = [[x,x,z] for x, z in zip(xrange(i, i+chk), zs)]
        xyzs = sum(xyzs, [])
        out = pack('f'*len(xyzs), *xyzs)
        j = i // block
        _o = _os[j]
        _o.write(out)
    '''

    # print 'finish', shape
    # return q2n, xy
    return q2n


def mat_split3(qry, step=4, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            # q2n[qid] = None
            qid_set.add(qid)
        if sid not in qid_set:
            # q2n[sid] = None
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()

    # print qid_set[:10]
    # print 'get all gene id'
    shape = len(qid_set)
    block = shape // step + 1

    # eye = range(shape)
    # for i, j in zip(q2n, eye):
    for i in xrange(shape):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * shape
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        # xi, yi = x % block, y % block
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(shape):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # print 'finish', shape
    # return q2n, xy
    return q2n


def mat_split4(qry, step=4, chunk=5 * 10**7, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            qid_set.add(qid)

        if sid not in qid_set:
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()
    N = len(qid_set)
    shape = (N, N)
    block = min(N // step + 1, chunk)

    for i in xrange(N):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # reorder the matrix
    cs = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    # N = len(q2n)
    # shape = (N, N)
    for fn in fns:
        # print 'loading', fns
        g = load_matrix(fn, shape=shape, csr=False)
        ci = csgraph.connected_components(g)
        if cs == None:
            cs = ci
        else:
            cs = merge_connected(cs, ci)

    idx = cs[1].argsort()
    idx_r = np.empty(N, 'int')
    idx_r[idx] = np.arange(N, dtype='int')
    idx = idx_r
    for i in q2n:
        j = q2n[i]
        q2n[i] = idx[j]

    # write reorder matrix
    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    return q2n


def mat_split5(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    lines = 0
    f = open(qry, 'r')
    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            qid_set.add(qid)

        if sid not in qid_set:
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    # qid_set.sort()
    np.random.shuffle(qid_set)
    N = len(qid_set)
    shape = (N, N)

    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )
    if lines * 3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = max(sqrt(lines * 3 // chunk) * sqrt(cpu), cpu)
        print 'tstep is', tstep
        tstep = min(max(tstep, 1), 30)
        print 'tstep 2 is', tstep
        # print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        #block = min(int(N/tstep)+ 1, int(N/cpu)+1)
        #block = N // step + 1
        block = N // tstep
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines * 3, chunk
        cpu = 1

    for i in xrange(N):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * N
    _os = {}

    f = open(qry, 'r')
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # reorder the matrix
    print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n


def mat_split6(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    lines = 0
    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            qid_set.add(qid)

        if sid not in qid_set:
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    # qid_set.sort()
    np.random.shuffle(qid_set)
    N = len(qid_set)
    shape = (N, N)

    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )
    if lines * 3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = max(sqrt(lines * 3 // chunk) * sqrt(cpu), cpu)
        print 'tstep is', tstep
        tstep = min(max(tstep, 1), 30)
        print 'tstep 2 is', tstep
        # print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        #block = min(int(N/tstep)+ 1, int(N/cpu)+1)
        #block = N // step + 1
        block = N // tstep
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines * 3, chunk
        cpu = 1

    for i in xrange(N):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * N
    _os = {}

    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            _ox = _os[(xi, yi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (xi, yi), 'wb')
            _os[(xi, yi)] = _o
            _ox = _os[(xi, yi)]

        _ox.write(out)

        # sym
        out = pack('fff', *[y, x, z])
        try:
            _oy = _os[(yi, xi)]
        except:
            _o = open(tmp_path + '/%d_%d.npz' % (yi, xi), 'wb')
            _os[(yi, xi)] = _o
            _oy = _os[(yi, xi)]

        if sym == False:
            _oy.write(out)

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            _o = _os[(j, j)]
        except:
            _os[(j, j)] = open(tmp_path + '/%d_%d.npz' % (j, j), 'wb')
            _o = _os[(j, j)]

        _o.write(out)

    # close the file
    for _o in _os.values():
        _o.close()

    # reorder the matrix
    # print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n


def mat_split7(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    qid_set = set()
    lines = 0
    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in qid_set:
            qid_set.add(qid)

        if sid not in qid_set:
            qid_set.add(sid)

    f.close()

    qid_set = list(qid_set)
    qid_set.sort()
    # np.random.seed(42)
    # np.random.shuffle(qid_set)
    N = len(qid_set)
    shape = (N, N)

    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )
    if lines * 3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = max(sqrt(lines * 3 // chunk) * sqrt(cpu), cpu)
        print 'tstep is', tstep
        #tstep = min(max(tstep, 1), 30)
        tstep = max(tstep, 1)
        print 'tstep 2 is', tstep
        # print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        #block = min(int(N/tstep)+ 1, int(N/cpu)+1)
        #block = N // step + 1
        block = N // tstep
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines * 3, chunk
        cpu = 1

    for i in xrange(N):
        qid = qid_set[i]
        q2n[qid] = i

    del qid_set
    gc.collect()

    eye = [0] * N
    _os = {}

    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    pairs = {}
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = float(score)
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            pairs[(xi, yi)].append(out)
        except:
            pairs[(xi, yi)] = [out]

        if sym == False:
            # sym
            out = pack('fff', *[y, x, z])
            try:
                pairs[(yi, xi)].append(out)
            except:
                pairs[(yi, xi)] = [out]
            flag += 1

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

        flag += 1
        # write batch to disk
        if flag % 5000000 == 0:
            for key, val in pairs.iteritems():
                if len(val) > 0:
                    a, b = key
                    _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
                    _o.writelines(val)
                    _o.close()
                    pairs[key] = []
                else:
                    continue

    f.close()

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            pairs[(j, j)].append(out)
        except:
            pairs[(j, j)] = [out]

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # reorder the matrix
    # print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n, block


# remove 1k file open limitation
def mat_split8(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4, sym=False, dtype='float32'):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    lines = 0
    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in q2n:
            q2n[qid] = None
        if sid not in q2n:
            q2n[sid] = None

    f.close()

    # np.random.seed(42)
    # np.random.shuffle(qid_set)
    N = len(q2n)
    shape = (N, N)

    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )
    if lines * 3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = max(sqrt(lines * 3 // chunk) * sqrt(cpu), cpu)
        print 'tstep is', tstep
        #tstep = min(max(tstep, 1), 30)
        tstep = max(tstep, 1)
        print 'tstep 2 is', tstep
        # print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        #block = min(int(N/tstep)+ 1, int(N/cpu)+1)
        #block = N // step + 1
        block = N // tstep
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines * 3, chunk
        cpu = 1

    #qn = q2n.keys()
    # qn.sort()
    # np.random.seed(42)
    # np.random.shuffle(qn)
    #flag = N - 1
    flag = 0
    for i in q2n:
        # for i in qn:
        q2n[i] = flag
        flag += 1
        #flag -= 1
    # for qid, i in izip(q2n, idxs):
    #    q2n[qid] = i
    #del qn
    gc.collect()

    eye = [0] * N
    _os = {}

    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    pairs = {}
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = abs(float(score))
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            pairs[(xi, yi)].append(out)
        except:
            pairs[(xi, yi)] = [out]

        if sym == False:
            # sym
            out = pack('fff', *[y, x, z])
            try:
                pairs[(yi, xi)].append(out)
            except:
                pairs[(yi, xi)] = [out]
            flag += 1

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

        flag += 1
        # write batch to disk
        if flag % 5000000 == 0:
            for key, val in pairs.iteritems():
                if len(val) > 0:
                    a, b = key
                    _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
                    _o.writelines(val)
                    _o.close()
                    pairs[key] = []
                else:
                    continue

    f.close()

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            pairs[(j, j)].append(out)
        except:
            pairs[(j, j)] = [out]

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # reorder the matrix
    # print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n, block


# breaks the input network into smaller ones
def mat_split9(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4, sym=False, dtype='float32', mem=4, prune=4000):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    lines = 0
    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        elif len(j) > 3:
            qid, sid, score = j[1:4]
        else:
            continue

        if qid not in q2n:
            q2n[qid] = None
        if sid not in q2n:
            q2n[sid] = None

    f.close()

    # np.random.seed(42)
    # np.random.shuffle(qid_set)
    N = len(q2n)

    # update chunk
    print 'memory limit', mem
    blk0 = N * prune * 12 * 50 / mem / 1e9
    blk1 = (N * prune * cpu * 6e2 / mem / 1e9) ** .5
    chunk = N * prune / blk1
    block0 = int(N / blk0) + 1
    block1 = int(N / blk1) + 1
    block = (block0 + block1) // 2

    print 'the new chunck size', N, cpu, mem, blk0, blk1, block
    shape = (N, N)

    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )

    '''
    if lines*3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = max(sqrt(lines*3//chunk) * sqrt(cpu), cpu)
        print 'tstep is', tstep
        #tstep = min(max(tstep, 1), 30)
        tstep = max(tstep, 1)
        print 'tstep 2 is', tstep
        #print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        #block = min(int(N/tstep)+ 1, int(N/cpu)+1)
        #block = N // step + 1
        block = N // tstep
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines*3, chunk
        cpu = 1
    '''

    #qn = q2n.keys()
    # qn.sort()
    # np.random.seed(42)
    # np.random.shuffle(qn)
    #flag = N - 1
    flag = 0
    for i in q2n:
        # for i in qn:
        q2n[i] = flag
        flag += 1
        #flag -= 1
    # for qid, i in izip(q2n, idxs):
    #    q2n[qid] = i
    #del qn
    gc.collect()

    eye = [0] * N
    _os = {}

    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    pairs = {}
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        elif len(j) > 3:
            qid, sid, score = j[1:4]
        else:
            continue

        z = abs(float(score))
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            pairs[(xi, yi)].append(out)
        except:
            pairs[(xi, yi)] = [out]

        if sym == False:
            # sym
            out = pack('fff', *[y, x, z])
            try:
                pairs[(yi, xi)].append(out)
            except:
                pairs[(yi, xi)] = [out]
            flag += 1

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

        flag += 1
        # write batch to disk
        if flag % 10000000 == 0:
            for key, val in pairs.iteritems():
                if len(val) > 0:
                    a, b = key
                    _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
                    _o.writelines(val)
                    _o.close()
                    pairs[key] = []
                else:
                    continue

    f.close()

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # set eye of matrix:
    for i in xrange(N):
        break
        z = eye[i]
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            pairs[(j, j)].append(out)
        except:
            pairs[(j, j)] = [out]

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # reorder the matrix
    # print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n, block


# split adj matrix
def mat_split(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4, sym=False, dtype='float32', mem=4, prune=4000, scale=True):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    lines = 0
    min_score = 0
    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        elif len(j) > 3:
            qid, sid, score = j[1:4]
        else:
            continue

        min_score = min(min_score, float(score))

        if qid not in q2n:
            q2n[qid] = None
        if sid not in q2n:
            q2n[sid] = None

    f.close()

    # np.random.seed(42)
    # np.random.shuffle(qid_set)
    N = len(q2n)
    #factor = scale and 1e9 / N / max_score or 1
    #factor = scale and 1e2 / max_score or 1
    factor = 1

    # update chunk
    print 'memory limit', mem
    blk0 = N * prune * 12 * 50 / mem / 1e9
    blk1 = (N * prune * cpu * 6e2 / mem / 1e9) ** .5
    chunk = N * prune / blk1
    block0 = int(N / blk0) + 1
    block1 = int(N / blk1) + 1
    block = (block0 + block1) // 2

    print 'the new chunck size', N, cpu, mem, blk0, blk1, block
    shape = (N, N)

   #flag = N - 1
    flag = 0
    for i in q2n:
        # for i in qn:
        q2n[i] = flag
        flag += 1
        #flag -= 1
    #del qn
    gc.collect()

    eye = [0] * N
    _os = {}

    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    pairs = {}
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        elif len(j) > 3:
            qid, sid, score = j[1:4]
        else:
            continue

        z = abs(float(score))
        if scale:
            z *= factor

        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x, y, z])
        xi, yi = x // block, y // block

        try:
            pairs[(xi, yi)].append(out)
        except:
            pairs[(xi, yi)] = [out]

        if sym == False:
            # sym
            out = pack('fff', *[y, x, z])
            try:
                pairs[(yi, xi)].append(out)
            except:
                pairs[(yi, xi)] = [out]
            flag += 1

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

        flag += 1
        # write batch to disk
        if flag % 10000000 == 0:
            for key, val in pairs.iteritems():
                if len(val) > 0:
                    a, b = key
                    if a == b:
                        continue
                    _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
                    _o.writelines(val)
                    _o.close()
                    pairs[key] = []
                else:
                    continue

    f.close()

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            if a == b:
                continue
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # set eye of matrix:
    for i in xrange(N):
        # break
        #z = eye[i] + 1
        z = 0 < eye[i] and eye[i] or 1
        #z = 1
        out = pack('fff', *[i, i, z])
        j = i // block
        try:
            pairs[(j, j)].append(out)
        except:
            pairs[(j, j)] = [out]

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # reorder the matrix
    # print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n, block


# add split method for gpu
def mat_split_gpu(qry, step=4, chunk=5 * 10**7, tmp_path=None, cpu=4, sym=False, dtype='float32', mem=4):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('mkdir -p %s' % tmp_path)
    q2n = {}
    lines = 0
    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    for i in f:
        lines += 1
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        if qid not in q2n:
            q2n[qid] = None
        if sid not in q2n:
            q2n[sid] = None

    f.close()

    # np.random.seed(42)
    # np.random.shuffle(qid_set)
    N = len(q2n)
    shape = (N, N)

    # update chunk
    print 'memory limit', mem
    blk0 = N * 1e3 * 12 * 50 / mem / 1e9
    blk1 = (N * 1e3 * cpu * 6e2 / mem / 1e9) ** .5
    chunk = N * 1e3 / blk1
    block0 = int(N / blk0) + 1
    block1 = int(N / blk1) + 1
    block = (block0 + block1) // 2

    '''
    # get the size of input file
    #tsize = os.path.getsize(qry)
    #tstep = max(tsize // (chunk*12), 1)
    #block = min(N//step+1, N//tstep+1 )
    if lines*3 > chunk:
        #tstep = max(lines*3//chunk*sqrt(cpu), cpu)
        tstep = max(sqrt(lines*3//chunk) * sqrt(cpu), cpu)
        print 'tstep is', tstep
        #tstep = min(max(tstep, 1), 30)
        tstep = max(tstep, 1)
        print 'tstep 2 is', tstep
        #print 'break point', step, tstep, lines * 3, chunk
        #block = min(N//step+1, int(N//tstep)+1)
        #block = min(int(N/tstep)+ 1, int(N/cpu)+1)
        #block = N // step + 1
        block = N // tstep
        print 'split block cpu=N', block, N // block
    else:
        block = N
        print 'split block cpu=1', block, N // block, lines*3, chunk
        cpu = 1
    '''

    block = int(block)
    #qn = q2n.keys()
    # qn.sort()
    # np.random.seed(42)
    # np.random.shuffle(qn)
    #flag = N - 1
    flag = 0
    for i in q2n:
        # for i in qn:
        q2n[i] = flag
        flag += 1
        #flag -= 1
    # for qid, i in izip(q2n, idxs):
    #    q2n[qid] = i
    #del qn
    gc.collect()

    eye = [0] * N
    _os = {}

    if mimetypes.guess_type(qry)[1] == 'gzip':
        f = gzip.open(qry, 'r')
    elif mimetypes.guess_type(qry)[1] == 'bzip2':
        f = bz2.BZ2File(qry, 'r')
    else:
        f = open(qry, 'r')

    pairs = {}
    flag = 0
    for i in f:
        j = i[:-1].split('\t')
        if len(j) == 3:
            qid, sid, score = j[:3]
        else:
            qid, sid, score = j[1:4]

        z = abs(float(score))
        x, y = map(q2n.get, [qid, sid])
        out = pack('fff', *[x % block, y % block, z])
        xi, yi = x // block, y // block

        try:
            pairs[(xi, yi)].append(out)
        except:
            pairs[(xi, yi)] = [out]

        if sym == False:
            # sym
            out = pack('fff', *[y % block, x % block, z])
            try:
                pairs[(yi, xi)].append(out)
            except:
                pairs[(yi, xi)] = [out]
            flag += 1

        if eye[x] < z:
            eye[x] = z
        if eye[y] < z:
            eye[y] = z

        flag += 1
        # write batch to disk
        if flag % 5000000 == 0:
            for key, val in pairs.iteritems():
                if len(val) > 0:
                    a, b = key
                    _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
                    _o.writelines(val)
                    _o.close()
                    pairs[key] = []
                else:
                    continue

    f.close()

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # set eye of matrix:
    for i in xrange(N):
        z = eye[i]
        out = pack('fff', *[i % block, i % block, z])
        j = i // block
        try:
            pairs[(j, j)].append(out)
        except:
            pairs[(j, j)] = [out]

    for key, val in pairs.iteritems():
        if len(val) > 0:
            a, b = key
            _o = open(tmp_path + '/%d_%d.npz' % (a, b), 'ab')
            _o.writelines(val)
            _o.close()
            pairs[key] = []
        else:
            continue

    # reorder the matrix
    # print 'reorder the matrix'
    #q2n = mat_reorder(qry, q2n, shape, False, tmp_path)

    return q2n, block


# save sparse csr on disk in my csr format
def save_npz_disk(csr, fn):
    #pass
    data = csr.data
    indices = csr.indices
    indptr = csr.indptr
    #r_size, c_size, d_size = indptr.size, indices.size, data.size
    #r_stride, c_stride, d_stride = indptr.strides, indices.strides, data.strides
    #N = r_size * r_stride + c_size * c_stride + d_size * d_stride
    a, b = indptr.size, indices.size
    print 'a', a, 'b', b, 'data', len(data)

    N = 4 + a + b + b
    fp = np.memmap(fn, mode='w+', shape=N, dtype='int32')
    R, C = csr.shape
    fp[:4] = [R, C, a, b]
    start = 4
    end = start + a
    #indptr.dtype = 'int8'
    fp[start: end] = indptr


    start = end
    end = b + start
    #indices.dtype = 'int8'
    fp[start:end] = indices

    start = end
    end = b + start
    data = np.asarray(data, 'float32')
    data.dtype = 'int32'
    fp[start:end] = data
    fp._mmap.close()
    del fp


# load my csr format on disk
def load_npz_disk(fn):
    fp = np.memmap(fn, dtype='int32')
    shape = (fp[0], fp[1])
    a, b = fp[2:4]
    indptr = fp[4:4+a]
    indices = fp[4+a:4+a+b]
    data = fp[4+a+b:4+a+b+b]
    data.dtype = 'float32'
    csr = sparse.csr_matrix(shape)
    csr.data, csr.indices, csr.indptr = data, indices, indptr
    return csr
      




# close my csr format:
def csr_close(csr):
    try:
       csr.indptr._mmap.close()
    except:
        pass


# load sparse matrix from disk
def load_matrix(qry, shape=(10**8, 10**8), csr=False):
    if csr == False:
        n = np.memmap(qry, mode='r+', dtype='float32').shape[0] / 3
        dat = np.memmap(qry, mode='r+', dtype='float32', shape=(n, 3))
        x, y, z = dat[:, 0], dat[:, 1], dat[:, 2]
        x = sparse.csr_matrix((z, (x, y)), shape=shape, dtype='float32')
        # print 'loading shape is', shape, qry, x.shape

    else:
        x = sparse.load_npz(qry)
        # print 'loading shape is', shape, qry, x.shape

    return x


# load sparse matrix from disk
def load_matrix_gpu(qry, shape=(10**8, 10**8), csr=False):
    if csr == False:
        return None
    else:
        x = sparse.load_npz(qry)
        block = x.shape[0]
        ij = qry.split(os.sep)[-1].split('.npz')[0].split('_')[:2]
        i, j = map(int, ij)
        a, b = x.nonzero()
        a += i * block
        b += j * block
        c = x.data
        x = sparse.csr_matrix((c, (a, b)), shape=shape)
        # print 'loading shape is', shape, qry, x.shape

    return x


# prune function
def prune_proto(x, p=1 / 4e4, S=500, R=600):
    R = min(S, R)
    R, C = x.shape
    dif = np.diff(x.indptr)
    dif_s = np.where(dif > R)
    for i in dif_s:
        st = x.indptr[st]
        ed = x.indptr[st + 1]
        dat = x.data[st: ed]
        n = (dat > p).sum()
        if n < R:
            dat[dat.argsort()[:-R]] = 0
        elif n > S:
            dat[dat.argsort()[:-S]] = 0
        else:
            dat[dat < p] = 0

    x.eliminate_zeros()
    return x

# csr sort by value


@njit(cache=True)
def csrsort_jit(a, b, c):
    #a, b, c = x.indices, x.indptr, x.data
    n = b.size
    flag = 0
    for i in xrange(n - 1):
        st, ed = b[i:i + 2]
        m = ed - st
        if m <= 1:
            # print st, ed
            continue
        # elif m == 2:
        #    j = st+1
        #    if c[st] < c[j]:
        #        c[st], c[j] = c[j], c[st]
        #        a[st], a[j] = a[j], a[st]
        else:
            idx = c[st:ed].argsort()
            idx = idx[::-1]
            a[st:ed] = a[st:ed][idx]
            c[st:ed] = c[st:ed][idx]
            flag += 1

    #print('sorting', flag, 'times')
    return flag


# sort the col by value
def csrsort0(x):
    a, b, c = x.indices, x.indptr, x.data
    flag = csrsort_jit(a, b, c)
    print 'sorting', flag, 'times'
    #x_s = sparse.csr_matrix((c, a, b), shape=x.shape, dtype=x.dtype)
    # return x_s



# sort each row of csr matrix
def csrsort(x, reverse=True):

    if reverse:
        idx = np.lexsort((-x.data, x.nonzero()[0]))
    else:
        idx = np.lexsort((x.data, x.nonzero()[0]))

    x.data = x.data[idx]
    x.indices = x.indices[idx]

    return 1







@njit(cache=True)
def csrmg_jit0(a0, b0, c0, a1, b1, c1, S=1):
    #a0, b0, c0 = x0.indices, x0.indptr, x0.data
    #a1, b1, c1 = x1.indices, x1.indptr, x1.data
    #row_min = np.empty(b0.size, c0.dtype)
    assert b0.size == b1.size
    print 'csrmg_jit_size_fk', len(a0), len(b0), len(c0), len(a1), len(b1), len(c1), S
    n = b0.size
    #nnz = min(a0.size + a1.size, b0.size*S)
    nnz = a0.size + a1.size
    a2, b2, c2 = np.empty(nnz, a0.dtype), np.empty(
        n, b0.dtype), np.empty(nnz, c0.dtype)
    b2[0] = 0
    ptr = 0
    for i in xrange(n - 1):
        st0, ed0 = b0[i:i + 2]
        st1, ed1 = b1[i:i + 2]
        p0, p1 = st0, st1
        flag = 0
        if st1 < ed1:
            print 'csrmg_jit_st_ed_fk', p0, st0, ed0, p1, st1, ed1, flag, S, '|', len(b1), b1.max(), c1[:20]
        while p0 < ed0 and p1 < ed1 and flag < S:
            if c0[p0] >= c1[p1]:
                c2[ptr] = c0[p0]
                a2[ptr] = a0[p0]
                p0 += 1
            else:
                c2[ptr] = c1[p1]
                a2[ptr] = a1[p1]
                p1 += 1

            ptr += 1
            flag += 1

        if flag > 0:
            print 'csrmg_jit_fk', flag
        #row_min[i] = a2[ptr-1]
        b2[i + 1] = ptr

    # print 'csrmge_jit_ptr_fk', ptr
    a2 = a2[:ptr]
    c2 = c2[:ptr]
    #z = sparse.csr_matrix((c2, a2, b2), shape=x0.shape, dtype=x0.dtype)
    # return z
    return a2, b2, c2


@njit(cache=True)
def csrmg_jit1(a0, b0, c0, a1, b1, c1, S=1):
    #a0, b0, c0 = x0.indices, x0.indptr, x0.data
    #a1, b1, c1 = x1.indices, x1.indptr, x1.data
    #row_min = np.empty(b0.size, c0.dtype)
    assert b0.size == b1.size
    print 'csrmg_jit_size_fk', len(a0), len(b0), len(c0), len(a1), len(b1), len(c1), S
    n = b0.size
    #nnz = min(a0.size + a1.size, b0.size*S)
    nnz = a0.size + a1.size
    a2, b2, c2 = np.empty(nnz, a0.dtype), np.empty(
        n, b0.dtype), np.empty(nnz, c0.dtype)
    b2[0] = 0
    ptr = 0
    for i in xrange(n - 1):
        st0, ed0 = b0[i:i + 2]
        st1, ed1 = b1[i:i + 2]
        p0, p1 = st0, st1
        flag = 0
        if st0 < ed0 and st1 == ed1:
            c0_rg = c0[st0: ed0][:S]
            a0_rg = a0[st0: ed0][:S]
            flag = c0_rg.size
            c2[ptr:ptr + flag] = c0_rg
            a2[ptr:ptr + flag] = a0_rg
            ptr += flag

        elif st0 == ed0 and st1 < ed1:
            c1_rg = c1[st0: ed0][:S]
            a1_rg = a1[st0: ed0][:S]
            flag = c1_rg.size
            c2[ptr:ptr + flag] = c1_rg
            a2[ptr:ptr + flag] = a1_rg
            ptr += flag

        else:
            while p0 < ed0 and p1 < ed1 and flag < S:
                if c0[p0] >= c1[p1]:
                    c2[ptr] = c0[p0]
                    a2[ptr] = a0[p0]
                    p0 += 1
                else:
                    c2[ptr] = c1[p1]
                    a2[ptr] = a1[p1]
                    p1 += 1

                ptr += 1
                flag += 1

        # if flag > 0:
        #    print 'csrmg_jit_fk', flag
        #row_min[i] = a2[ptr-1]
        b2[i + 1] = ptr

    # print 'csrmge_jit_ptr_fk', ptr
    a2 = a2[:ptr]
    c2 = c2[:ptr]
    #z = sparse.csr_matrix((c2, a2, b2), shape=x0.shape, dtype=x0.dtype)
    # return z
    return a2, b2, c2


@njit(cache=True)
def csrmg_jit2(a0, b0, c0, a1, b1, c1, S=1):
    #a0, b0, c0 = x0.indices, x0.indptr, x0.data
    #a1, b1, c1 = x1.indices, x1.indptr, x1.data
    #row_min = np.empty(b0.size, c0.dtype)
    assert b0.size == b1.size
    print 'csrmg_jit_size_fk', len(a0), len(b0), len(c0), len(a1), len(b1), len(c1), S
    n = b0.size
    #nnz = min(a0.size + a1.size, b0.size*S)
    nnz = a0.size + a1.size
    a2, b2, c2 = np.empty(nnz, a0.dtype), np.empty(
        n, b0.dtype), np.empty(nnz, c0.dtype)
    b2[0] = 0
    ptr = 0
    for i in xrange(n - 1):
        st0, ed0 = b0[i:i + 2]
        st1, ed1 = b1[i:i + 2]
        p0, p1 = st0, st1
        # if flag > 0:
        #    print 'csrmg_jit_fk', flag
        #row_min[i] = a2[ptr-1]
        ptr_mg = 0
        ln_mg = ed0 - st0 + ed1 - st1
        c_mg = np.empty(ln_mg, c0.dtype)
        a_mg = np.empty(ln_mg, a0.dtype)
        while p0 < ed0 and p1 < ed1:
            if c0[p0] >= c1[p1]:
                c_mg[ptr_mg] = c0[p0]
                a_mg[ptr_mg] = a0[p0]
                p0 += 1
            else:
                c_mg[ptr_mg] = c1[p1]
                a_mg[ptr_mg] = a1[p1]
                p1 += 1
            ptr_mg += 1

        # print 'csrmg_jit_while_fk', ln_mg, ptr_mg, p0, ed0, p1, ed1

        if p0 < ed0 and p1 >= ed1:
            c_mg[ptr_mg:] = c0[p0: ed0]
            a_mg[ptr_mg:] = a0[p0: ed0]

        elif p0 >= ed0 and p1 < ed1:
            c_mg[ptr_mg:] = c1[p1: ed1]
            a_mg[ptr_mg:] = a1[p1: ed1]
        else:
            pass

        # print 'csrmg_jit_while_fk_end', ln_mg, ptr_mg, p0, ed0, p1, ed1

        c_mg_S = c_mg[:S]
        a_mg_S = a_mg[:S]
        flag = c_mg_S.size
        c2[ptr:ptr + flag] = c_mg_S
        a2[ptr:ptr + flag] = a_mg_S
        ptr += flag

        b2[i + 1] = ptr

    # print 'csrmge_jit_ptr_fk', ptr
    a2 = a2[:ptr]
    c2 = c2[:ptr]
    #z = sparse.csr_matrix((c2, a2, b2), shape=x0.shape, dtype=x0.dtype)
    # return z
    return a2, b2, c2


@njit(cache=True)
def csrmg_jit(a0, b0, c0, a1, b1, c1, S=1000000):
    #a0, b0, c0 = x0.indices, x0.indptr, x0.data
    #a1, b1, c1 = x1.indices, x1.indptr, x1.data
    #row_min = np.empty(b0.size, c0.dtype)
    assert b0.size == b1.size
    print 'csrmg_jit_size_fk', len(a0), len(b0), len(c0), len(a1), len(b1), len(c1), S
    n = b0.size
    #nnz = min(a0.size + a1.size, b0.size*S)
    nnz = c0.size + c1.size
    a2, b2, c2 = np.empty(nnz, a0.dtype), np.empty(
        n, b0.dtype), np.empty(nnz, c0.dtype)
    b2[0] = 0
    ptr = 0
    c_mg = np.empty(S, c0.dtype)
    a_mg = np.empty(S, a0.dtype)

    for i in xrange(n - 1):
        st0, ed0 = b0[i:i + 2]
        st1, ed1 = b1[i:i + 2]
        p0, p1 = st0, st1
        # if flag > 0:
        #    print 'csrmg_jit_fk', flag
        #row_min[i] = a2[ptr-1]
        ptr_mg = 0
        #c_mg[:] = 0
        #a_mg[:] = 0
        while p0 < ed0 and p1 < ed1 and ptr_mg < S:
            if c0[p0] >= c1[p1]:
                c_mg[ptr_mg] = c0[p0]
                a_mg[ptr_mg] = a0[p0]
                p0 += 1
            else:
                c_mg[ptr_mg] = c1[p1]
                a_mg[ptr_mg] = a1[p1]
                p1 += 1
            ptr_mg += 1

        # print 'csrmg_jit_while_fk', ln_mg, ptr_mg, p0, ed0, p1, ed1
        if ptr_mg < S and p0 >= ed0 and p1 >= ed1:
            c_mg[ptr_mg:] = 0
            a_mg[ptr_mg:] = 0
            end = ptr_mg

        elif ptr_mg < S and p0 < ed0 and p1 >= ed1:
            M = min(ed0 - p0, S - ptr_mg)
            c_mg[ptr_mg: ptr_mg + M] = c0[p0: p0 + M]
            a_mg[ptr_mg: ptr_mg + M] = a0[p0: p0 + M]
            end = ptr_mg + M

        elif ptr_mg < S and p0 >= ed0 and p1 < ed1:
            M = min(ed1 - p1, S - ptr_mg)
            #c_mg[ptr_mg:] = c1[p1: ed1]
            #a_mg[ptr_mg:] = a1[p1: ed1]
            c_mg[ptr_mg: ptr_mg + M] = c1[p1: p1 + M]
            a_mg[ptr_mg: ptr_mg + M] = a1[p1: p1 + M]
            end = ptr_mg + M

        else:
            end = ptr_mg

        # print 'csrmg_jit_while_fk_end', ln_mg, ptr_mg, p0, ed0, p1, ed1

        #c_mg_S = c_mg[:S]
        #a_mg_S = a_mg[:S]
        #flag = c_mg_S.size

        # print 'S_fk_flag', S, flag
        c2[ptr:ptr + end] = c_mg[:end]
        a2[ptr:ptr + end] = a_mg[:end]
        ptr += end

        b2[i + 1] = ptr

    # print 'csrmge_jit_ptr_fk', ptr
    a2 = a2[:ptr]
    c2 = c2[:ptr]
    #z = sparse.csr_matrix((c2, a2, b2), shape=x0.shape, dtype=x0.dtype)
    # return z
    return a2, b2, c2


# select
@njit(cache=True)
def select_jit(a, b, c, S=1000000):
    #a, b, c = x.indices, x.indptr, x.data
    n = b.size
    flag = 0
    for i in xrange(n - 1):
        st, ed = b[i:i + 2]
        m = ed - st
        if m <= S:
            # print st, ed
            continue
        else:
            rdata = c[st:ed]
            idx = rdata.argsort()
            #p = idx[-S:]
            #rdata[rdata<p] = 0
            rdata[idx[:-S]] = 0
            c[st:ed] = rdata
            flag += 1

    #print('sorting', flag, 'times')
    print 'select_S', S, flag
    return flag


# def csrmerge(x0, x1, S=1400):
#@njit(cache=True)
def csrmerge(x0, x1, prune=1 / 4e3, S=1100, R=1400):
    thr = max(int(1. / prune) + 1, S, R)
    print 'before_csr_merge', x0.sum(0).max(), x1.sum(0).max(),  x0.sum(1).max(), x1.sum(1).max()

    a0, b0, c0 = x0.indices, x0.indptr, x0.data
    a1, b1, c1 = x1.indices, x1.indptr, x1.data
    #a2, b2, c2 = csrmg_jit(a0, b0, c0, a1, b1, c1, S)
    a2, b2, c2 = csrmg_jit(a0, b0, c0, a1, b1, c1, thr)
    z = sparse.csr_matrix((c2, a2, b2), shape=x0.shape, dtype=x0.dtype)
    # print 'after_csr_merge', len(a0), len(b0), len(c0),  np.diff(b0).max(),
    # '|', len(a1), len(b1), len(c1), np.diff(b0).max(), '|', len(a2),
    # len(b2), len(c2), '|', z.nnz, 1./prune, thr, S, R,
    # np.diff(z.indptr).max()
    print 'after_csr_merge', z.sum(0).max(), z.sum(1).max()
    return z


# find the lower bound of each row
@njit(cache=True)
def find_lower0(indptr, data, prune=1e-4, R=300):
    n = indptr.size
    ps = np.empty(n, data.dtype)
    for i in xrange(n - 1):
        st, ed = indptr[i:i + 2]
        row = data[st:ed]
        idx = row > prune
        j = idx.sum()
        if j > R:
            ps[i] = prune
        else:
            ps[i] = row[:R][-1]

    return ps


@njit(cache=True)
def find_lower1(indptr, data, prune=1 / 4e3, S=1100, R=1400):
    n = indptr.size
    ps = np.empty(n, data.dtype)
    #S = max(1./prune, R, S)
    for i in xrange(n - 1):
        st, ed = indptr[i:i + 2]
        m = ed - st
        if m <= R:
            row = data[st:ed]
            ps[i] = 0
            continue
            # print'ps_less', ps[i]
        else:
            row = data[st:ed]
            idx = row > prune
            j = idx.sum()
            if j <= R:
                idx_s = row.argsort()
                idx_m = idx_s[-R]
                ps[i] = row[idx_m]
                # print'ps_less_2', ps[i]

            elif j > S > R:
                idx_s = row.argsort()
                idx_m = idx_s[-S]
                ps[i] = row[idx_m]
                # print'ps_more', ps[i]

            else:
                ps[i] = prune
                # print'ps_good', ps[i]

    return ps


@njit(cache=True)
def find_lower2(indptr, data, prune=1 / 4e3, S=1100, R=1400):
    n = indptr.size
    ps = np.empty(n, data.dtype)
    #S = max(1./prune, R, S)
    for i in xrange(n - 1):
        st, ed = indptr[i:i + 2]
        m = ed - st
        if m <= R:
            row = data[st:ed]
            ps[i] = 0
            continue
            # print'ps_less', ps[i]
        else:
            row = data[st:ed]
            idx = row > prune
            j = idx.sum()
            if j <= R:
                idx_s = row.argsort()
                #idx_m = idx_s[m-R]
                idx_m = idx_s[-R]
                ps[i] = row[idx_m]
                # print'ps_less_2', ps[i]

            elif j > S > R and S <= m:
                idx_s = row.argsort()
                #idx_m = idx_s[m-S]
                idx_m = idx_s[-S]
                ps[i] = row[idx_m]
                # print'ps_more', ps[i]

            else:
                ps[i] = prune
                # print'ps_good', ps[i]

    return ps


@njit(cache=True)
def find_lower3(indptr, data, prune=1 / 4e3, S=1100, R=1400, order=True):
    n = indptr.size
    ps = np.empty(n, data.dtype)
    #ps[:] = prune
    #ps = np.zeros(n, data.dtype)
    print 'find_lower_P_fk', prune
    flag = 0
    for i in xrange(n - 1):
        st, ed = indptr[i:i + 2]
        rdata = data[st:ed]
        #m = ed - st
        m = rdata.size
        if m <= R:
            #row = data[st:ed]
            ps[i] = rdata[-1]
            # continue
            # print'ps_less', ps[i], i, m, R, rdata[0], rdata[-1], rdata[:10]
        else:
            idx = rdata > prune
            j = idx.sum()
            pct = rdata[idx].sum()
            if j < R < m and pct < .85:
                #idx_s = row.argsort()
                #idx_m = idx_s[m-R]
                #idx_m = idx_s[-R]
                #ps[i] = row[idx_m]
                # print'ps_less_2', ps[i]
                ps[i] = rdata[R]
                flag += m - R

            elif j > S < m:
                #idx_s = row.argsort()
                #idx_m = idx_s[m-S]
                #idx_m = idx_s[-S]
                #ps[i] = row[idx_m]
                # print'ps_more', ps[i]
                if S < R and pct < .85:
                    ps[i] = rdata[R]
                else:
                    ps[i] = rdata[S]

                flag += j < S and m - R or m - S

            else:
                ps[i] = prune
                # print'ps_good', ps[i]
                # continue
    print 'find_lower_rm_fk', flag
    return ps


@njit(cache=True)
def find_lower(indptr, data, prune=1 / 4e3, S=1100, R=1400, order=True, Pct=.9):
    n = indptr.size
    ps = np.empty(n, data.dtype)
    #ps[:] = prune
    #ps = np.zeros(n, data.dtype)
    print 'find_lower_P_fk', prune
    flag = 0
    pct_max = 0
    pct_min = 2**30
    for i in xrange(n - 1):
        st, ed = indptr[i:i + 2]
        if st == ed:
            ps[i] = prune
            continue
        rdata = data[st:ed]
        m = rdata.size
        idx = rdata > prune
        j = idx.sum()

        pct = rdata[idx].sum()

        #pct_max = max(pct, pct_max)
        #pct_min = min(pct, pct_min)

        #pct_max = max(pct, rdata.sum())
        #pct_min = min(pct, rdata.sum())

        if j < R < m and pct < Pct:
            ps[i] = rdata[R]
            flag += m - R

        elif j > S < m:
            if S < R and pct < Pct:
                ps[i] = rdata[R]
            else:
                ps[i] = rdata[S]

            flag += j < S and m - R or m - S

        else:
            ps[i] = prune

    print 'find_lower_rm_fk', flag
    return ps


# remove element by give threshold
@njit(cache=True)
def rm_elem0(indptr, data, prune):

    N = (data > 0).sum()
    # print 'before_prune_rm'
    n = indptr.size
    for i in xrange(n - 1):
        st, ed = indptr[i:i + 2]
        row = data[st:ed]
        p = prune[i]
        row[row < p] = 0

        #print (row<p).sum(), row.size

    Nw = (data > 0).sum()
    print 'after_prune_rm', N, Nw
    # print 'after_prune_rm', (data>0).sum(), (prune>0).sum()


# find the threshold of prune by row
def find_cutoff_row_mg(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    for elem in elems:
        a, b, tmp_path, p, S, R = elem
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
        except:
            continue

        # sort x1
        csrsort(x1)
        print 'csrsorting'
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            x0 = csrmerge(x0, x1, p, S, R)

    print 'max_diff', np.diff(x0.indptr).max(), x0.nnz, x0.indptr
    ps = find_lower(x0.indptr, x0.data, prune=p, S=S, R=R)

    # prune
    for elem in elems:
        a, b, tmp_path, p, S, R = elem
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()
        sparse.save_npz(fn, x1)




@njit(cache=True)
def rm_elem(indptr, data, prune, p1=-1):
    if p1 > 0:
        data[data<p1] = 0
    else:
        N = (data > 0).sum()
        # print 'before_prune_rm'
        n = indptr.size
        for i in xrange(n - 1):
            st, ed = indptr[i:i + 2]
            row = data[st:ed]
            p = prune[i]
            row[row < p] = 0

            #print (row<p).sum(), row.size

        Nw = (data > 0).sum()
        print 'after_prune_rm', N, Nw
        # print 'after_prune_rm', (data>0).sum(), (prune>0).sum()


# find the threshold of prune by row
def find_cutoff_row_mg(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    for elem in elems:
        a, b, tmp_path, p, S, R = elem
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
        except:
            continue

        # sort x1
        csrsort(x1)
        print 'csrsorting'
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            x0 = csrmerge(x0, x1, p, S, R)

    print 'max_diff', np.diff(x0.indptr).max(), x0.nnz, x0.indptr
    ps = find_lower(x0.indptr, x0.data, prune=p, S=S, R=R)

    # prune
    for elem in elems:
        a, b, tmp_path, p, S, R = elem
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()
        sparse.save_npz(fn, x1)


def find_cutoff_row(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    for elem in elems:
        a, b, tmp_path, p, S, R = elem
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
        except:
            continue

        # sort x1
        # csrsort(x1)
        # print 'csrsorting'
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            #x0 = csrmerge(x0, x1, p, S, R)
            x0 += x1

    csrsort(x0)

    # print 'max_diff', np.diff(x0.indptr).max(), x0.nnz, x0.indptr
    ps = find_lower(x0.indptr, x0.data, prune=p, S=S, R=R)

    # prune
    for elem in elems:
        a, b, tmp_path, p, S, R = elem
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()
        sparse.save_npz(fn, x1)


# find threshold of prune by col
def find_cutoff_col0(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        b, a = a, b
        print 'cutoff_ab_fk', a, b, elems
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn).T
        except:
            print 'max_fn', fn

            continue

        # sort x1
        # csrsort(x1)
        # print 'csrsorting', x1.nnz
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            # print 'csrmerge_fk', 1./P, S, R
            #x0 = csrmerge(x0, x1, P, S, R)
            x0 += x1

    csrsort(x0)
    x0.eliminate_zeros()
    print 'max_diff_fk', np.diff(x0.indptr).max(), x0.nnz, x0.indptr[:100]
    ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)

    # prune
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        b, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn).T
        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz, 'before_max_row', np.diff(x1.indptr).max(), 1. / P, S, R
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()

        tmp = np.diff(x1.indptr)
        tmp_index = np.where(tmp == tmp.max())[0][0]

        print 'after_prune_fk', tmp.max(), (ps > 0).sum(), x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index, ps[tmp_index]

        sparse.save_npz(fn, x1.T)


def find_cutoff_col1(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #b, a = a, b
        print 'cutoff_ab_fk', a, b, elems
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn).T
        except:
            print 'max_fn', fn

            continue

        # sort x1
        # csrsort(x1)
        # print 'csrsorting', x1.nnz
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            print 'csrmerge_fk', 1. / P, S, R
            #x0 = csrmerge(x0, x1, P, S, R)
            x0 += x1

    x0 = csrsort(x1)
    x0.eliminate_zeros()
    print 'max_diff_fk', np.diff(x0.indptr).max(), x0.nnz, x0.indptr[:100]
    ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)

    # prune
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #a, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn).T
        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz, 'before_max_row', np.diff(x1.indptr).max(), 1. / P, S, R
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()

        tmp = np.diff(x1.indptr)
        tmp_index = np.where(tmp == tmp.max())[0][0]

        # print 'after_prune_fk', tmp.max(), (ps > 0).sum(),
        # x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index,
        # ps[tmp_index]

        sparse.save_npz(fn, x1.T)


def find_cutoff_col(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    rowsum = None
    colsum = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #b, a = a, b
        # print 'cutoff_ab_fk', a, b, elems
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x1 = sparse.load_npz(fn)
            x1 = x1.T.tocsr()
            #xtmp = sparse.load_npz(fn)
            # try:
            #    rowsum += xtmp.sum(0)
            # except:
            #    rowsum = xtmp.sum(0)
            #x1 = xtmp.T

        except:
            print 'max_fn', fn
            continue

        # sort x1
        # csrsort(x1)
        # try:
        #    colsum += x1.sum(0)
        # except:
        #    colsum = x1.sum(0)

        print 'csrsorting', x1.nnz
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            # print 'csrmerge_fk', 1./P, S, R
            #x0 = csrmerge(x0, x1, P, S, R)
            x0 += x1

    csrsort(x0)

    #a, b, c = x.indices, x.indptr, x.data

    thr = max(1. / P, S, R)
    select_jit(x0.indices, x0.indptr, x0.data, thr)

    x0.eliminate_zeros()
    # print 'max_diff_fk', np.diff(x0.indptr).max(), x0.nnz, x0.indptr[:100]
    # print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max(), rowsum.max(),
    # colsum.max()
    print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max()

    #x0t = x0.T
    #ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)
    ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)

    # prune
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #a, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            #x1 = sparse.load_npz(fn).T
            x1 = sparse.load_npz(fn).T.tocsr()

        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz, 'before_max_row', np.diff(x1.indptr).max(), 1. / P, S, R
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()

        tmp = np.diff(x1.indptr)
        tmp_index = np.where(tmp == tmp.max())[0][0]

        print 'after_prune_fk', tmp.max(), (ps > 0).sum(), x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index, ps[tmp_index]

        sparse.save_npz(fn, x1.T.tocsr())


def find_cutoff_col_mg0(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    rowsum = None
    colsum = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #b, a = a, b
        print 'cutoff_ab_fk', a, b, elems
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            xtmp = sparse.load_npz(fn)
            # try:
            #    rowsum += xtmp.sum(0)
            # except:
            #    rowsum = xtmp.sum(0)

            x1 = xtmp.T.tocsr()

        except:
            print 'max_fn', fn
            continue

        # sort x1
        csrsort(x1)
        # try:
        #    colsum += x1.sum(0)
        # except:
        #    colsum = x1.sum(0)

        print 'csrsorting', x1.nnz
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            print 'csrmerge_fk', 1. / P, S, R
            x0 = csrmerge(x0, x1, P, S, R)

    if type(x0) == type(None):
        return []

    x0.eliminate_zeros()
    # print 'max_diff_fk', np.diff(x0.indptr).max(), x0.nnz, x0.indptr[:100]
    print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max()

    # print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max(), rowsum.max(), colsum.max()
    #x0t = x0.T
    #ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)
    ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R, order=True, Pct=.9)

    # prune
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #a, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            xtmp = sparse.load_npz(fn).T
            x1 = xtmp.tocsr()
        except:
            continue
        # remove small element
        print 'before_before_prune', x1.nnz, 'before_max_row', np.diff(x1.indptr).max(), 1. / P, S, R
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()

        tmp = np.diff(x1.indptr)
        tmp_index = np.where(tmp == tmp.max())[0][0]

        print 'after_prune_fk', tmp.max(), (ps > 0).sum(), x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index, ps[tmp_index]

        sparse.save_npz(fn, x1.T.tocsr())



def find_cutoff_col_mg1(elems):
    if len(elems) <= 0:
        return []
    x0 = None
    rowsum = None
    colsum = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #b, a = a, b
        print 'cutoff_ab_fk', a, b, elems
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            xtmp = sparse.load_npz(fn)
            x1 = xtmp.T.tocsr()

        except:
            print 'max_fn', fn
            continue

        # sort x1
        csrsort(x1)
        # try:
        #    colsum += x1.sum(0)
        # except:
        #    colsum = x1.sum(0)

        print 'csrsorting', x1.nnz
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            print 'csrmerge_fk', 1. / P, S, R
            x0 = csrmerge(x0, x1, P, S, R)

    if type(x0) == type(None):
        return []

    x0.eliminate_zeros()
    # print 'max_diff_fk', np.diff(x0.indptr).max(), x0.nnz, x0.indptr[:100]
    print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max()

    # print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max(), rowsum.max(), colsum.max()
    #x0t = x0.T
    #ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)
    ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R, order=True, Pct=.9)

    sq = None
    mx_c = None
    # prune
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #a, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            #xtmp = sparse.load_npz(fn).T
            #x1 = xtmp.tocsr()
            x1 = sparse.load_npz(fn).T.tocsr()

        except:
            continue
        # remove small element
        # print 'before_before_prune', x1.nnz, 'before_max_row',
        # np.diff(x1.indptr).max(), 1./P, S, R
        rm_elem(x1.indptr, x1.data, ps)

        x1.eliminate_zeros()

        tmp = np.diff(x1.indptr)
        tmp_index = np.where(tmp == tmp.max())[0][0]

        # print 'after_prune_fk', tmp.max(), (ps > 0).sum(),
        # x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index,
        # ps[tmp_index]

        x2 = x1.T.tocsr()
        #sparse.save_npz(fn, x1.T.tocsr())
        sparse.save_npz(fn, x2)
        if type(sq) == type(mx_c) == type(None):
            #sq = np.asarray(x2.power(2).sum(0))
            sq = x2.power(2).sum(0)
            #mx_c = np.asarray(x2.max(0).todense())[0]
            mx_c = x2.max(0).todense()

        else:
            #sq += np.asarray(x2.power(2).sum(0))
            sq += x2.power(2).sum(0)
            #mx_i = np.asarray(x2.max(0).todense())[0]
            mx_i = x2.max(0).todense()

            mx_c = np.max([mx_c, mx_i], 0)
            #idx = mx_c < mx_i
            #mx_c[idx] = mx_i[idx]

    #chaos = np.nan_to_num(mx_c/sq).max()
    chaos = (mx_c - sq).max()

    return chaos



def find_cutoff_col_mg(elems):
    if len(elems) <= 0:
        return 1
    x0 = None
    x1 = None
    rowsum = None
    colsum = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #b, a = a, b
        print 'cutoff_ab_fk', a, b, elems
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            xtmp = sparse.load_npz(fn)
            xtmp = xtmp.T.tocsr()
        except:
            print 'max_fn', fn
            continue

        if type(x1) != type(None):
            x1 += xtmp
        else:
            x1 = xtmp

        if x1.nnz >= 1e8:
            # sort x1
            csrsort(x1)
            print 'csrsorting', x1.nnz
            # merge with x0
            if type(x0) == type(None):
                x0 = x1
            else:
                #x0 = csrmerge(x0, x1, S)
                print 'csrmerge_fk', 1. / P, S, R
                x0 = csrmerge(x0, x1, P, S, R)

            x1 = None

    if type(x1) != type(None):
        # sort x1
        csrsort(x1)
        print 'csrsorting', x1.nnz
        # merge with x0
        if type(x0) == type(None):
            x0 = x1
        else:
            #x0 = csrmerge(x0, x1, S)
            print 'csrmerge_fk', 1. / P, S, R
            x0 = csrmerge(x0, x1, P, S, R)

    if type(x0) == type(None):
        return 1

    x0.eliminate_zeros()
    # print 'max_diff_fk', np.diff(x0.indptr).max(), x0.nnz, x0.indptr[:100]
    print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max()

    # print 'max_x_mg', x0.sum(0).max(), x0.sum(1).max(), rowsum.max(), colsum.max()
    #x0t = x0.T
    #ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R)
    ps = find_lower(x0.indptr, x0.data, prune=P, S=S, R=R, order=True, Pct=.9)

    sq = None
    mx_c = None
    # prune
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #a, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            #xtmp = sparse.load_npz(fn).T
            #x1 = xtmp.tocsr()
            x1 = sparse.load_npz(fn).T.tocsr()

        except:
            continue
        # remove small element
        # print 'before_before_prune', x1.nnz, 'before_max_row',
        # np.diff(x1.indptr).max(), 1./P, S, R
        rm_elem(x1.indptr, x1.data, ps, -1)

        x1.eliminate_zeros()

        #tmp = np.diff(x1.indptr)
        #tmp_index = np.where(tmp == tmp.max())[0][0]

        # print 'after_prune_fk', tmp.max(), (ps > 0).sum(),
        # x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index,
        # ps[tmp_index]

        x2 = x1.T.tocsr()
        #sparse.save_npz(fn, x1.T.tocsr())
        sparse.save_npz(fn, x2)
        if type(sq) == type(mx_c) == type(None):
            #sq = np.asarray(x2.power(2).sum(0))
            sq = x2.power(2).sum(0)
            #mx_c = np.asarray(x2.max(0).todense())[0]
            mx_c = x2.max(0).todense()

        else:
            #sq += np.asarray(x2.power(2).sum(0))
            sq += x2.power(2).sum(0)
            #mx_i = np.asarray(x2.max(0).todense())[0]
            mx_i = x2.max(0).todense()

            mx_c = np.max([mx_c, mx_i], 0)
            #idx = mx_c < mx_i
            #mx_c[idx] = mx_i[idx]

    #chaos = np.nan_to_num(mx_c/sq).max()
    chaos = (mx_c - sq).max()

    return chaos



def find_cutoff_col_mg_fast(elems):
    if len(elems) <= 0:
        return 1
    # prune
    PS = None
    sq = None
    mx_c = None
    #tmp = None
    for elem in elems:
        a, b, tmp_path, P, S, R = elem
        #print 'fast_a_b', a, b
        #a, a = a, b
        fn = tmp_path + '/%d_%d.npz' % (a, b)
        try:
            x2 = sparse.load_npz(fn)

        except:
            continue

        #try:
        #    tmp += x2
        #except:
        #    tmp = x2

        # remove small element
        #if type(PS) == type(None):
        #    PS = np.empty(x2.size)

        #rm_elem(x2.indptr, x2.data, PS, P)

        x2.data[x2.data<P] = 0
        x2.eliminate_zeros()

        #tmp = np.diff(x1.indptr)
        #tmp_index = np.where(tmp == tmp.max())[0][0]

        # print 'after_prune_fk', tmp.max(), (ps > 0).sum(),
        # x1[tmp_index].data,  len(x1[tmp_index].data), ps.shape, tmp_index,
        # ps[tmp_index]

        #x2 = x1
        #sparse.save_npz(fn, x1.T.tocsr())
        sparse.save_npz(fn, x2)
        if type(sq) == type(mx_c) == type(None):
            #sq = np.asarray(x2.power(2).sum(0))
            sq = x2.power(2).sum(0)
            #mx_c = np.asarray(x2.max(0).todense())[0]
            mx_c = x2.max(0).todense()

        else:
            #sq += np.asarray(x2.power(2).sum(0))
            sq += x2.power(2).sum(0)
            #mx_i = np.asarray(x2.max(0).todense())[0]
            mx_i = x2.max(0).todense()

            mx_c = np.max([mx_c, mx_i], 0)
            #idx = mx_c < mx_i
            #mx_c[idx] = mx_i[idx]

    #chaos = np.nan_to_num(mx_c/sq).max()
    chaos = (mx_c - sq).max()
    #tmp = mx_c - sq
    #idx = np.where(tmp == chaos)
    #print 'rm_elem_P_chaos', P, chaos, mx_c[idx], sq[idx], idx, mx_c, b
    #print 'rm_elem_P_chaos', P, chaos, idx, mx_c, a, b

    #tmp1 = tmp.power(2).sum(0)
    #tmp2 = tmp.max(0)
    #print 'rm_elem_max', tmp1[tmp1>0].min()
    #print 'rm_elem_max', (tmp2-tmp1).max(), (tmp1-sq).max(), (tmp2-mx_c).max()

    return chaos









#find_cutoff = find_cutoff_col
find_cutoff = find_cutoff_col_mg
#find_cutoff = find_cutoff_row_mg


# prune
def pruning0(qry, tmp_path=None, prune=1 / 4e3, S=1100, R=1400, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1

    # find the threshold
    #xys = [[[a, b, tmp_path, prune, S, R] for b in xrange(N)] for a in xrange(N)]

    xys = [[[b, a, tmp_path, prune, S, R]
            for b in xrange(N)] for a in xrange(N)]

    if cpu <= 1 or len(xys) <= 1:
        cutoff = map(find_cutoff, xys)
    else:
        cutoff = Parallel(n_jobs=cpu)(delayed(find_cutoff)(elem)
                                      for elem in xys)

        #pool = mp.Pool(cpu)
        #cutoff = pool.map(find_cutoff, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    return max(cutoff)



def pruning(qry, tmp_path=None, prune=1 / 4e3, S=1100, R=1400, cpu=1, fast=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1

    # find the threshold
    #xys = [[[a, b, tmp_path, prune, S, R] for b in xrange(N)] for a in xrange(N)]

    if fast:
        find_cutoff = find_cutoff_col_mg_fast
    else:
        find_cutoff = find_cutoff_col_mg

    xys = [[[b, a, tmp_path, prune, S, R]
            for b in xrange(N)] for a in xrange(N)]

    if cpu <= 1 or len(xys) <= 1:
        cutoff = map(find_cutoff, xys)
    else:
        cutoff = Parallel(n_jobs=cpu)(delayed(find_cutoff)(elem)
                                      for elem in xys)

    return max(cutoff)








# get threshold of large k
def topk(x, k):
    assert len(x) >= k
    lo = hi = 0
    for i in x:
        lo = lo > i and i or lo
        hi = hi < i and i or hi

    ct = N = len(x)
    p = lo
    while lo < hi:
        ct = 0
        p = (lo + hi) / 2.
        for i in x:
            ct += i >= p and 1 or 0
        if ct == k:
            break
        elif ct < k:
            hi = p
        else:
            lo = p

	return p, count


@njit(fastmath=True, cache=True)
def topks(indptr, indices, data, k):
    R = indptr.size
    nnz = indices.size
    lo, hi = np.zeros(R, dtype=np.float32), np.zeros(R, dtype=np.float32)
    end = 0
    for i in xrange(nnz):
        col = indices[i]
        val = data[i]
        if lo[col] > val:
            lo[col] = val
        if hi[col] < val:
            hi[col] = val

        end = col < end and end or col

    end += 1

    lo = lo[: end]
    hi = hi[: end]
    ct = np.zeros(end, dtype=np.int32)
    visit = np.ones(end, dtype=np.int8)
    mi = lo.copy()

    flag = np.any(visit)
    itr = 0
    while flag:

        print 'iteration', itr, visit.sum()
        itr += 1

        #ct[:] = 0
        for i in xrange(end):
            if visit[i] == 0:
                continue

            ct[i] = 0
            mi[i] = (hi[i] + lo[i]) / 2.

        for i in xrange(nnz):
            col = indices[i]
            if visit[col] == 0:
                #print 'yes', col
                continue

            val = data[i]
            mid = mi[col]
            # get top k
            if val > mid:
                ct[col] += 1

        for i in xrange(end):
            if visit[i] == 0:
                continue

            if ct[i] < k:
                hi[i] = mi[i]
            elif ct[i] > k:
                lo[i] = mi[i]
            else:
                visit[i] = 0

            if lo[i] >= hi[i]:
                visit[i] = 0

        flag = np.any(visit)

    return mi, ct






# split row block and col block into row_col block
def preprocess(qry, tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    rows = [elem for elem in fns if elem.endswith('_row.bin')]
    cols = [elem for elem in fns if elem.endswith('_col.bin')]

    for i in rows:
        xn = tmp_path + '/' + i
        x = load_matrix(xn)
        # x += x.transpose()
        for j in cols:
            yn = tmp_path + '/' + j
            y = load_matrix(yn)
            # y += y.transpose()
            z = x * y
            xi = i.split(os.sep)[-1].split('_row')[0]
            yj = j.split(os.sep)[-1].split('_col')[0]
            ij = xi + '_' + yj
            sparse.save_npz(tmp_path + '/' + ij, z)
            del y, z
            gc.collect()

        # remove x
        del x
        gc.collect()
        os.system('rm %s' % xn)


# matrix mul on small blocks
def mul0(qry, shape=(10**7, 10**7), tmp_path=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    rc = [elem for elem in fns if elem.endswith('.npz')]
    rows, cols = [], []
    for i in rc:
        j = i.split(os.sep)[-1].split('.npz')[0]
        x, y = j.split('_')
        rows.append(x)
        cols.append(y)
    for i in rows:
        for j in cols:
            z = sparse.csr_matrix(shape, dtype='float32')
            for k in rows:
                xn = tmp_path + i + '_' + k + '.npz'
                yn = tmp_path + k + '_' + j + '.npz'
                try:
                    x = load_matrix(xn, load=False)
                    y = load_matrix(yn, load=False)
                except:
                    continue
                z += x * y

            zn = tmp_path + i + '_' + j + '_new'
            sparse.save_npz(zn, z)

    # rename
    for i in xy:
        os.system('mv %s_new %s' % (i, i))


def mul1(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
              for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)
    for i in xy:
        for j in xy:
            z = sparse.csr_matrix(shape, dtype='float32')
            for k in xy:
                xn = tmp_path + '/' + i + '_' + k + '.npz'
                x = load_matrix(xn, load=load)
                if i != j:
                    yn = tmp_path + '/' + k + '_' + j + '.npz'
                    y = load_matrix(yn, load=load)
                else:
                    y = x
                print 'current', (i, k), (k, j), x.shape, y.shape, z.shape
                z += x * y

            zn = tmp_path + '/' + i + '_' + j + '_new'
            sparse.save_npz(zn, z)

    # rename
    for i in xy:
        for j in xy:
            k = i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))


def mul2(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
              for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)
    for i in xy:
        for k in xy:
            xn = tmp_path + '/' + i + '_' + k + '.npz'
            x = load_matrix(xn, load=load)
            for j in xy:
                if i != j:
                    yn = tmp_path + '/' + k + '_' + j + '.npz'
                    y = load_matrix(yn, load=load)
                else:
                    y = x

                zn = tmp_path + '/' + i + '_' + j + '_new'
                try:
                    z = load_matrix(zn + '.npz', load=True)
                except:
                    z = sparse.csr_matrix(shape, dtype='float32')

                print 'current', (i, k), (k, j), x.shape, y.shape, z.shape
                z += x * y
                sparse.save_npz(zn, z)

    # rename
    for i in xy:
        for j in xy:
            k = i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))


def mul3(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
              for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)

    for i in xy:
        # get row
        xs = []
        for idx in xy:
            xn = tmp_path + '/' + i + '_' + idx + '.npz'
            try:
                x = load_matrix(xn, load=load)
            except:
                x = None

            print 'loading x', x.shape
            # raise SystemExit()
            xs.append(x)

        for j in xy:
            # get col
            ys = []
            for idx in xy:
                if idx == i:
                    y = xs[int(j)]
                else:
                    yn = tmp_path + '/' + idx + '_' + j + '.npz'
                    try:
                        y = load_matrix(yn, load=load)
                    except:
                        y = None

                print 'loading y', y.shape
                ys.append(y)

            Z = sparse.csr_matrix(shape, dtype='float32')

            for X, Y in zip(xs, ys):
                try:
                    # Z += X * Y

                    start = time()
                    tmp = X * Y
                    print 'time usage', i, j, time() - start

                    Z += tmp
                    del tmp
                    gc.collect()

                except:
                    continue

            zn = tmp_path + '/' + i + '_' + j + '_new'
            sparse.save_npz(zn, Z)

    # rename
    for i in xy:
        for j in xy:
            k = tmp_path + '/' + i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))


def mul4(qry, shape=(10**8, 10**8), tmp_path=None, xy=[], load=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
              for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in xy:
        # get row
        xs = []
        for idx in xy:
            xn = tmp_path + '/' + i + '_' + idx + '.npz'
            try:
                x = load_matrix(xn, load=load)
            except:
                x = None

            print 'loading x', x.shape
            # raise SystemExit()
            xs.append(x)

        for j in xy:
            # get col
            ys = []
            for idx in xy:
                if idx == i:
                    y = xs[int(j)]
                else:
                    yn = tmp_path + '/' + idx + '_' + j + '.npz'
                    try:
                        y = load_matrix(yn, load=load)
                    except:
                        y = None

                print 'loading y', y.shape
                ys.append(y)

            Z = sparse.csr_matrix(shape, dtype='float32')

            for X, Y in zip(xs, ys):
                try:
                    # Z += X * Y

                    start = time()
                    tmp = X * Y
                    print 'time usage', i, j, time() - start

                    Z += tmp
                    del tmp
                    gc.collect()

                except:
                    continue

            zn = tmp_path + '/' + i + '_' + j + '_new'
            sparse.save_npz(zn, Z)
            row_sum += Z.sum(0)

    # rename
    for i in xy:
        for j in xy:
            k = tmp_path + '/' + i + '_' + j
            os.system('mv %s_new.npz %s.npz' % (k, k))

    return row_sum


def mul(qry, shape=(10**8, 10**8), tmp_path=None, csr=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        Z = sparse.csr_matrix(shape, dtype='float32')
        x = load_matrix(i, shape=shape, csr=csr)
        for j in fns:
            # get col
            start = time()
            if i != j:
                y = load_matrix(j, shape=shape, csr=csr)
            else:
                y = x

            print 'loading time', i.split('/')[-1], j.split('/')[-1], time() - start

            start = time()
            tmp = x * y
            Z += tmp
            print 'multiple time', time() - start

            del tmp
            gc.collect()

        sparse.save_npz(i + '_new', Z)
        print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum


def expand0(qry, shape=(10**8, 10**8), tmp_path=None, csr=False, I=1.5, prune=1e-5, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = float('+inf')
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        Z = sparse.csr_matrix(shape, dtype='float32')
        x = load_matrix(i, shape=shape, csr=csr)
        for j in fns:
            # get col
            start = time()
            if i != j:
                y = load_matrix(j, shape=shape, csr=csr)
            else:
                y = x

            print 'loading time', i.split('/')[-1], j.split('/')[-1], time() - start
            start = time()
            # tmp = x * y
            # tmp = np.dot(x, y)
            tmp = Pmul(x, y)
            Z += tmp
            print 'multiple time', time() - start

            del tmp
            gc.collect()

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        if check:
            err = min((abs(Z - x) - rtol * abs(x)).max(), err)

        sparse.save_npz(i + '_new', Z)
        print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    if check:
        print 'current error', err
    cvg = err < atol
    return row_sum, fns, cvg


def expand2(qry, shape=(10**8, 10**8), tmp_path=None, csr=False, I=1.5, prune=1e-5, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
               for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        # Z = sparse.csr_matrix(shape, dtype='float32')
        Z_old = Z = None
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        print 'current cell', a, b, num_set
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can\'t load x', xn
                continue
            if xn != yn:
                try:
                    y = load_matrix(yn, shape=shape, csr=csr)
                except:
                    print 'can\'t load y', yn
                    continue
            else:
                y = x

            # print 'loading time', xn.split('/')[-1], yn.split('/')[-1],
            # time() - start
            start = time()
            # tmp = x * y
            # tmp = np.dot(x, y)
            # get old z
            if xn == i:
                Z_old = x
            elif yn == i:
                Z_old = y
            else:
                pass

            tmp = Pmul(x, y)
            try:
                Z += tmp
            except:
                Z = tmp
            print 'multiple time', time() - start, a, j, j, b

            del tmp
            gc.collect()

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        # if check:
        #    err = max((abs(Z-Z_old)-rtol * abs(Z_old)).max(), err)

        sparse.save_npz(i + '_new', Z)
        print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old.npz' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    if check:
        print 'current error', err

    if err != None:
        cvg = err < atol
    else:
        cvg = False

    return row_sum, fns, cvg


def expand3(qry, shape=(10**8, 10**8), tmp_path=None, csr=False, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
               for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    # print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        # Z = sparse.csr_matrix(shape, dtype='float32')
        Z = None
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        # print 'current cell', a, b, num_set
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                # print 'can\'t load x', xn
                continue
            if xn != yn:
                try:
                    y = load_matrix(yn, shape=shape, csr=csr)
                except:
                    # print 'can\'t load y', yn
                    continue
            else:
                y = x

            # print 'loading time', xn.split('/')[-1], yn.split('/')[-1],
            # time() - start
            start = time()
            # tmp = x * y
            # tmp = np.dot(x, y)
            # get old z

            tmp = Pmul(x, y, cpu=cpu)
            try:
                Z += tmp
            except:
                Z = tmp
            # print 'multiple time', time() - start, a, j, j, b
            del tmp
            gc.collect()

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()

        sparse.save_npz(i + '_new', Z)
        # print 'saved', i
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns

# merge submatrix


def merge_submat0(fns, shape=(10**7, 10**7), csr=False):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    print 'merged names', names
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            if len(rows) == len(cols) == 1:
                r, c = rows[0], cols[0]
                R, C = map(str, [r, c])
                rc = tmp_path + os.sep + R + '_' + C + '.npz'
                if os.path.isfile(rc):
                    print 'single block', r, c, rows, i, names[i:i + 2]
                    print 'single block new', rc, out
                    os.system('mv %s %s' % (rc, out))
                    os.system('mv %s_old %s_old' % (rc, out))

                continue
            z = z_old = None
            for r in rows:
                for c in cols:
                    R, C = map(str, [r, c])
                    rc = tmp_path + os.sep + R + '_' + C + '.npz'
                    try:
                        tmp = load_matrix(rc, shape, csr=csr)
                        print 'rm old file', rc
                        os.system('rm %s' % rc)
                        print 'rmed old file', rc

                        tmp_old = load_matrix(rc + '_old', shape, csr=csr)
                        print 'rm prev old file', rc + '_old'
                        os.system('rm %s_old' % rc)
                        print 'rmed prev old file', rc + '_old'

                    except:
                        continue
                    try:
                        z += tmp
                        z_old += tmp_old
                    except:
                        z = tmp
                        z_old = tmp_old

            if type(z) != type(None):
                sparse.save_npz(out, z)
                sparse.save_npz(out + '_old', z_old)
                os.system('mv %s_old.npz %s_old' % (out, out))
                fns_new.append(out)
                nnz = max(nnz, z.nnz)
                merged = True
    print 'before merged', fns
    print 'after merged', fns_new
    return row_sum, fns_new, nnz, merged


def merge_submat1(fns, shape=(10**7, 10**7), csr=False):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    print 'merged names', names
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            if len(rows) == len(cols) == 1:
                r, c = rows[0], cols[0]
                R, C = map(str, [r, c])
                rc = tmp_path + os.sep + R + '_' + C + '.npz'
                if os.path.isfile(rc):
                    print 'single block', r, c, rows, i, names[i:i + 2]
                    print 'single block new', rc, out
                    os.system('mv %s %s' % (rc, out))
                    os.system('mv %s_old %s_old' % (rc, out))

                continue
            z = None
            for r in rows:
                for c in cols:
                    R, C = map(str, [r, c])
                    rc = tmp_path + os.sep + R + '_' + C + '.npz'
                    try:
                        tmp = load_matrix(rc, shape, csr=csr)
                        print 'rm old file', rc
                        os.system('rm %s' % rc)
                        print 'rmed old file', rc

                    except:
                        continue
                    try:
                        z += tmp
                    except:
                        z = tmp

            if type(z) != type(None):
                sparse.save_npz(out, z)
                fns_new.append(out)
                nnz = max(nnz, z.nnz)
                merged = True
                del z
                gc.collect()

            z_old = None
            for r in rows:
                for c in cols:
                    R, C = map(str, [r, c])
                    rc = tmp_path + os.sep + R + '_' + C + '.npz'
                    try:
                        tmp_old = load_matrix(rc + '_old', shape, csr=csr)
                        print 'rm prev old file', rc + '_old'
                        os.system('rm %s_old' % rc)
                        print 'rmed prev old file', rc + '_old'

                    except:
                        continue
                    try:
                        z_old += tmp_old
                    except:
                        z_old = tmp_old

            if type(z_old) != type(None):
                sparse.save_npz(out + '_old', z_old)
                os.system('mv %s_old.npz %s_old' % (out, out))

    print 'before merged', fns
    print 'after merged', fns_new
    return row_sum, fns_new, nnz, merged


# sub merge function
def submerge0(xys):
    i, j, rows, cols, shape, tmp_path, csr = xys
    I = str(i // 2)
    J = str(j // 2)
    out = tmp_path + os.sep + I + '_' + J + '.npz'
    fns_new = None
    row_sum_n = None
    nnz = 0

    merged = False
    z = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            #tmp = load_matrix(rc, shape, csr=csr)
            try:
                tmp = load_matrix(rc, shape, csr=csr)
                print 'rm old file', rc
                os.system('rm %s' % rc)
                print 'rmed old file', rc

            except:
                continue
            try:
                z += tmp
            except:
                z = tmp

    if type(z) != type(None):
        sparse.save_npz(out, z)
        fns_new = out
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        #row_sum_n = out + '_rowsum.npz'
        #np.savez_compressed(row_sum_n, row_sum)

        merged = True
        nnz = max(nnz, z.nnz)
        del z
        gc.collect()

    z_old = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            try:
                tmp_old = load_matrix(rc + '_old', shape, csr=csr)
                print 'rm prev old file', rc + '_old'
                os.system('rm %s_old' % rc)
                print 'rmed prev old file', rc + '_old'

            except:
                continue
            try:
                z_old += tmp_old
            except:
                z_old = tmp_old

    if type(z_old) != type(None):
        sparse.save_npz(out + '_old', z_old)
        os.system('mv %s_old.npz %s_old' % (out, out))
        del z_old
        gc.collect()

    return row_sum_n, fns_new, nnz, merged


# try to fix multiple cpu support
def submerge(xys):
    i, j, rows, cols, shape, tmp_path, csr = xys
    I = str(i // 2)
    J = str(j // 2)
    out = tmp_path + os.sep + I + '_' + J + '.npz'
    fns_new = None
    row_sum_n = None
    nnz = 0

    merged = False
    z = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            #tmp = load_matrix(rc, shape, csr=csr)
            try:
                tmp = load_matrix(rc, shape, csr=csr)
                print 'rm old file', rc
                os.system('rm %s' % rc)
                print 'rmed old file', rc

            except:
                continue
            try:
                z += tmp
            except:
                z = tmp

    if type(z) != type(None):
        sparse.save_npz(out + '_merge', z)
        fns_new = out
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        #row_sum_n = out + '_rowsum.npz'
        #np.savez_compressed(row_sum_n, row_sum)

        merged = True
        nnz = max(nnz, z.nnz)
        del z
        gc.collect()

    z_old = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            try:
                tmp_old = load_matrix(rc + '_old', shape, csr=csr)
                print 'rm prev old file', rc + '_old'
                os.system('rm %s_old' % rc)
                print 'rmed prev old file', rc + '_old'

            except:
                continue
            try:
                z_old += tmp_old
            except:
                z_old = tmp_old

    if type(z_old) != type(None):
        sparse.save_npz(out + '_old_merge', z_old)
        #os.system('mv %s_old.npz %s_old'%(out, out))
        del z_old
        gc.collect()

    return row_sum_n, fns_new, nnz, merged


def rsubmerge(xys):
    i, j, rows, cols, shape, tmp_path, csr = xys
    I = str(i // 2)
    J = str(j // 2)
    out = tmp_path + os.sep + I + '_' + J + '.npz'
    fns_new = None
    row_sum_n = None
    nnz = 0

    merged = False
    z = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            #tmp = load_matrix(rc, shape, csr=csr)
            try:
                tmp = load_matrix(rc, shape, csr=csr)
                print 'rm_old_file', rc
                os.system('rm %s' % rc)
                print 'rmed_old_file', rc

            except:
                continue
            try:
                z += tmp
            except:
                z = tmp

    if type(z) != type(None):
        sparse.save_npz(out + '_merge', z)
        fns_new = out
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        #row_sum_n = out + '_rowsum.npz'
        #np.savez_compressed(row_sum_n, row_sum)

        merged = True
        nnz = max(nnz, z.nnz)
        del z
        gc.collect()

    z_old = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            try:
                tmp_old = load_matrix(rc + '_old', shape, csr=csr)
                print 'rm_prev_old_z_file', rc + '_old'
                os.system('rm %s_old' % rc)
                print 'rmed_prev_old_z_file', rc + '_old'

            except:
                continue
            try:
                z_old += tmp_old
            except:
                z_old = tmp_old

    if type(z_old) != type(None):
        sparse.save_npz(out + '_old_merge', z_old)
        #os.system('mv %s_old.npz %s_old'%(out, out))
        del z_old
        gc.collect()

    z_mg = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'

            try:
                tmp_old = load_matrix(rc + '_Mg.npz', shape, csr=csr)
                print 'rm_prev_Mg_file', rc + '_Mg.npz'
                os.system('rm %s_Mg.npz' % rc)
                print 'rmed_prev_Mg_file', rc + '_Mg.npz'

            except:
                print 'cannot_found_Mg_file', rc
                continue
            try:
                z_mg += tmp_old
            except:
                z_mg = tmp_old

    if type(z_mg) != type(None):
        sparse.save_npz(out + '_Mg.npz_merge', z_mg)
        #os.system('mv %s_old.npz %s_old'%(out, out))
        del z_mg
        gc.collect()

    return row_sum_n, fns_new, nnz, merged


# submerge on batch data
def submerge_wrapper(elem):
    out = []
    for xys in elem:
        tmp = submerge(xys)
        out.append(tmp)

    return out


def rsubmerge_wrapper(elem):
    out = []
    for xys in elem:
        tmp = rsubmerge(xys)
        out.append(tmp)

    return out


# parallel merge_submat
def merge_submat0(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = []
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xys.append([i, j, rows, cols, shape, tmp_path, csr])

    if cpu <= 1 or len(xys) <= 1:
        zns = map(submerge, xys)
    else:
        zns = Parallel(n_jobs=cpu)(delayed(submerge)(elem) for elem in xys)

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for i in zns:
        row_sum_s, fns_s, nnz_s, merged_s = i
        if fns_s == None:
            continue
        # try:
        #    tmp = np.load(row_sum_s)
        #    tmp = tmp.items()[0][1]
        #    tmp = np.asarray(tmp, 'float32')
        #    os.system('rm %s'%row_sum_s)
        # except:
        #    continue
        # try:
        #    row_sum += tmp
        # except:
        #    row_sum = tmp

        fns_new.append(fns_s)
        nnz = max(nnz, nnz_s)
        if merged_s:
            merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


# parallel merge_submat
def merge_submat1(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xy = [i, j, rows, cols, shape, tmp_path, csr]
            xys[flag % cpu].append(xy)
            flag += 1

    if cpu <= 1 or len(xys) <= 1:
        zns = map(submerge_wrapper, xys)
    else:
        zns = Parallel(n_jobs=cpu)(delayed(submerge_wrapper)(elem)
                                   for elem in xys)

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for elem in zns:
        for i in elem:
            row_sum_s, fns_s, nnz_s, merged_s = i
            if fns_s == None:
                continue

            fns_new.append(fns_s)
            nnz = max(nnz, nnz_s)
            if merged_s:
                merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


def merge_submat2(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xy = [i, j, rows, cols, shape, tmp_path, csr]
            xys[flag % cpu].append(xy)
            flag += 1

    if cpu <= 1 or len(xys) <= 1:
        zns = map(submerge_wrapper, xys)
    else:
        zns = Parallel(n_jobs=cpu)(delayed(submerge_wrapper)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(submerge_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for elem in zns:
        for i in elem:
            row_sum_s, fns_s, nnz_s, merged_s = i
            if fns_s == None:
                continue

            fns_new.append(fns_s)
            nnz = max(nnz, nnz_s)
            if merged_s:
                merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


def merge_submat3(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz') or elem.endswith('.npz_old')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xy = [i, j, rows, cols, shape, tmp_path, csr]
            xys[flag % cpu].append(xy)
            flag += 1

    if cpu <= 1 or len(xys) <= 1:
        # if 1:
        zns = map(submerge_wrapper, xys)
    else:
        zns = Parallel(n_jobs=cpu)(delayed(submerge_wrapper)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(submerge_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        gc.collect()

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for elem in zns:
        for i in elem:
            row_sum_s, fns_s, nnz_s, merged_s = i
            if fns_s == None:
                continue

            fns_new.append(fns_s)
            nnz = max(nnz, nnz_s)
            if merged_s:
                merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


def merge_submat(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz') or elem.endswith('.npz_old')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xy = [i, j, rows, cols, shape, tmp_path, csr]
            xys[flag % cpu].append(xy)
            flag += 1

    # if cpu <= 1:
    if cpu <= 1 or len(xys) <= 1:
        # if 1:
        zns = map(submerge_wrapper, xys)
    else:
        print 'parallel_merge_submat'
        zns = Parallel(n_jobs=cpu)(delayed(submerge_wrapper)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(submerge_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        gc.collect()
    old_fns = [tmp_path + '/' +
               elem for elem in os.listdir(tmp_path) if not elem.endswith('_merge.npz')]
    for i in old_fns:
        os.system('rm %s' % i)
    new_fns = [tmp_path + '/' +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_merge.npz')]
    for i in new_fns:
        j = i.split('_merge.npz')[0]
        os.system('mv %s %s' % (i, j))
        # print 'old_fns_new_fns', i, j

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for elem in zns:
        for i in elem:
            row_sum_s, fns_s, nnz_s, merged_s = i
            if fns_s == None:
                continue

            fns_new.append(fns_s)
            nnz = max(nnz, nnz_s)
            if merged_s:
                merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


def rmerge_submat(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz') or elem.endswith('.npz_old')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xy = [i, j, rows, cols, shape, tmp_path, csr]
            xys[flag % cpu].append(xy)
            flag += 1

    # if cpu <= 1:
    if cpu <= 1 or len(xys) <= 1:
        # if 1:
        zns = map(rsubmerge_wrapper, xys)
    else:
        print 'parallel_merge_submat'
        zns = Parallel(n_jobs=cpu)(delayed(rsubmerge_wrapper)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(submerge_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        gc.collect()
    old_fns = [tmp_path + '/' +
               elem for elem in os.listdir(tmp_path) if not elem.endswith('_merge.npz')]
    for i in old_fns:
        os.system('rm %s' % i)
    new_fns = [tmp_path + '/' +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_merge.npz')]
    for i in new_fns:
        j = i.split('_merge.npz')[0]
        os.system('mv %s %s' % (i, j))
        # print 'old_fns_new_fns', i, j

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for elem in zns:
        for i in elem:
            row_sum_s, fns_s, nnz_s, merged_s = i
            if fns_s == None:
                continue

            fns_new.append(fns_s)
            nnz = max(nnz, nnz_s)
            if merged_s:
                merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


# submerge of gpu
def submerge_gpu(xys):
    i, j, rows, cols, shape, tmp_path, csr = xys
    I = str(i // 2)
    J = str(j // 2)
    out = tmp_path + os.sep + I + '_' + J + '.npz'
    fns_new = None
    row_sum_n = None
    nnz = 0

    merged = False
    #nrow, ncol = shape
    #z = sparse.csr_matrix(nrow*2, ncol*2)
    if len(rows) <= 1:
        Rows = [rows[0], rows[0] + 1]
    else:
        Rows = rows

    if len(cols) <= 1:
        Cols = [cols[0], cols[0] + 1]
    else:
        Cols = cols

    z_vs = []
    for r in Rows:
        z_hs = []
        for c in Cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            try:
                tmp = load_matrix(rc, shape, csr=csr)
                print 'rm old file', rc
                os.system('rm %s' % rc)
                print 'rmed old file', rc
                print 'before_mreged_z is', tmp.shape

            except:
                tmp = sparse.csr_matrix(shape, dtype='float32')

            z_hs.append(tmp)

        z_h = sparse.hstack(z_hs, format='csr')
        z_vs.append(z_h)

    z = sparse.vstack(z_vs, format='csr')
    print 'after_mreged_z is', z.shape

    if type(z) != type(None):
        sparse.save_npz(out + '_merge', z)
        fns_new = out
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        #row_sum_n = out + '_rowsum.npz'
        #np.savez_compressed(row_sum_n, row_sum)

        merged = True
        nnz = max(nnz, z.nnz)
        del z
        gc.collect()

    z_old = None
    for r in rows:
        for c in cols:
            R, C = map(str, [r, c])
            rc = tmp_path + os.sep + R + '_' + C + '.npz'
            try:
                tmp_old = load_matrix(rc + '_old', shape, csr=csr)
                print 'rm prev old file', rc + '_old'
                os.system('rm %s_old' % rc)
                print 'rmed prev old file', rc + '_old'

            except:
                continue
            try:
                z_old += tmp_old
            except:
                z_old = tmp_old

    if type(z_old) != type(None):
        sparse.save_npz(out + '_old_merge', z_old)
        #os.system('mv %s_old.npz %s_old'%(out, out))
        del z_old
        gc.collect()

    return row_sum_n, fns_new, nnz, merged


# submerge on batch data
def submerge_wrapper_gpu(elem):
    out = []
    for xys in elem:
        tmp = submerge_gpu(xys)
        out.append(tmp)

    return out


def merge_submat_gpu(fns, shape=(10**7, 10**7), csr=False, cpu=1):
    #fns = [tmp_path+'/'+elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    tmp_path = os.sep.join(fns[0].split(os.sep)[:-1])
    names = [elem.split(os.sep)[-1].split('.npz')[0].split('_')
             for elem in fns if elem.endswith('.npz') or elem.endswith('.npz_old')]
    names = map(int, sum(names, []))
    N = max(names) + 1
    names = range(N)
    print 'merged names', names
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for i in xrange(0, N, 2):
        for j in xrange(0, N, 2):
            I = str(i // 2)
            J = str(j // 2)
            out = tmp_path + os.sep + I + '_' + J + '.npz'
            rows = names[i:i + 2]
            cols = names[j:j + 2]
            xy = [i, j, rows, cols, shape, tmp_path, csr]
            xys[flag % cpu].append(xy)
            flag += 1

    if cpu <= 1 or len(xys) <= 1:
        zns = map(submerge_wrapper_gpu, xys)
    else:
        print 'parallel_merge_submat'
        zns = Parallel(n_jobs=cpu)(delayed(submerge_wrapper_gpu)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(submerge_wrapper_gpu, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        gc.collect()
    old_fns = [tmp_path + '/' +
               elem for elem in os.listdir(tmp_path) if not elem.endswith('_merge.npz')]
    for i in old_fns:
        os.system('rm %s' % i)
    new_fns = [tmp_path + '/' +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_merge.npz')]
    for i in new_fns:
        j = i.split('_merge.npz')[0]
        os.system('mv %s %s' % (i, j))
        # print 'old_fns_new_fns', i, j

    nnz = 0
    row_sum = None
    merged = False
    fns_new = []
    for elem in zns:
        for i in elem:
            row_sum_s, fns_s, nnz_s, merged_s = i
            if fns_s == None:
                continue

            fns_new.append(fns_s)
            nnz = max(nnz, nnz_s)
            if merged_s:
                merged = True

    print 'before merged', fns, zns
    print 'after merged', fns_new, zns
    return row_sum, fns_new, nnz, merged


def sdot(x, nnz=25000000):
    xn, yn, shape, csr = x
    try:
        x = load_matrix(xn, shape=shape, csr=csr)
    except:
        return None
    if xn != yn:
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
        except:
            return None
    else:
        y = x

    z = x * y
    del x
    del y
    gc.collect()
    if z.nnz > nnz:
        name = xn + '_tmp.npz'
        sparse.save_npz(name, z)
        del z
        gc.collect()
        return name
    else:
        return z


# calculate the element of matrix
def element0(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi
        print 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
        except:
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
        except:
            continue
        tmp = x * y
        try:
            z += tmp
        except:
            z = tmp

    if type(z) == type(None):
        return None, None, None

    z.data **= I
    z.data[z.data < prune] = 0
    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    #row_sum += np.asarray(z.sum(0))[0]
    # return row_sum
    row_sum = z.sum(0)
    # print 'row_sum is', type(row_sum)
    return row_sum, xyn, nnz


def element1(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    x = y = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            xt = load_matrix(xn, shape=shape, csr=csr)
        except:
            xt = None
        if type(xt) != type(None):
            try:
                x += xt
            except:
                x = xt

        try:
            yt = load_matrix(yn, shape=shape, csr=csr)
        except:
            yt = None

        if type(yt) != type(None):
            try:
                y += yt
            except:
                y = yt
        del xt
        del yt
        gc.collect()

    try:
        z = x * y
    except:
        z = None

    if type(z) == type(None):
        return None, None, None

    z.data **= I
    z.data[z.data < prune] = 0
    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)

    # return row_sum
    #row_sum = z.sum(0)
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    # print 'row_sum is', type(row_sum)
    # return row_sum, xyn, nnz
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def element2(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            #x_i, x_j = x.nonzero()
            #x.data[x_i==x_j] = 1
            #x.data[x.data<prune] = 0
            # x.eliminate_zeros()
        except:
            print 'can not load x', xn
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
            #y_i, y_j = y.nonzero()
            #y.data[y_i==y_j] = 1
            #y.data[y.data<prune] = 0
            # y.eliminate_zeros()

        except:
            print 'can not load y', yn
            continue
        #tmp = x * y
        tmp = csrmm_ez(x, y)
        try:
            z += tmp
        except:
            z = tmp

        del x, y, tmp
        gc.collect()

    if type(z) == type(None):
        return None, None, None

    z.data **= I
    #z.data[z.data < prune] = 0
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    z.data[norm_dat < prune] = 0
    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)

    # return row_sum
    #row_sum = z.sum(0)
    #row_sum = np.asarray(z.sum(0), 'float32')[0]
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    # print 'row_sum is', type(row_sum)
    # return row_sum, xyn, nnz
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def element3(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            #x_i, x_j = x.nonzero()
            #x.data[x_i==x_j] = 1
            #x.data[x.data<prune] = 0
            # x.eliminate_zeros()
        except:
            print 'can not load x', xn
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
            #y_i, y_j = y.nonzero()
            #y.data[y_i==y_j] = 1
            #y.data[y.data<prune] = 0
            # y.eliminate_zeros()

        except:
            print 'can not load y', yn
            continue
        #tmp = x * y
        tmp = csrmm_ez(x, y, cpu=cpu)
        try:
            z += tmp
        except:
            z = tmp

        del x, y, tmp
        gc.collect()

    if type(z) == type(None):
        return None, None, None

    z.data **= I
    #z.data[z.data < prune] = 0
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    z.data[norm_dat < prune] = 0
    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)

    # return row_sum
    #row_sum = z.sum(0)
    #row_sum = np.asarray(z.sum(0), 'float32')[0]
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    # print 'row_sum is', type(row_sum)
    # return row_sum, xyn, nnz
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def element4(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            #x_i, x_j = x.nonzero()
            #x.data[x_i==x_j] = 1
            #x.data[x.data<prune] = 0
            # x.eliminate_zeros()
        except:
            print 'can not load x', xn
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
            #y_i, y_j = y.nonzero()
            #y.data[y_i==y_j] = 1
            #y.data[y.data<prune] = 0
            # y.eliminate_zeros()

        except:
            print 'can not load y', yn
            continue
        #tmp = x * y
        #xyn_tmp = tmp_path + '/' + str(xi) + '_' + str(yi) + '_tmp'
        xyn_tmp = tmp_path + '/' + \
            str(xi) + '_' + str(i) + '_' + str(yi) + '_tmp'
        tmp = csrmm_ez(x, y, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)
        try:
            z += tmp
        except:
            z = tmp

        del x, y, tmp
        gc.collect()

    if type(z) == type(None):
        return None, None, None

    z.data **= I
    #z.data[z.data < prune] = 0
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    z.data[norm_dat < prune] = 0
    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)

    # return row_sum
    #row_sum = z.sum(0)
    #row_sum = np.asarray(z.sum(0), 'float32')[0]
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    # print 'row_sum is', type(row_sum)
    # return row_sum, xyn, nnz
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def element_fast0(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            if type(xr) == type(None):
                xr = x
            else:
                xr += x
        except:
            print 'can not load x', xn
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
            if type(yc) == type(None):
                yc = y
            else:
                yc += y

        except:
            print 'can not load y', yn
            continue

        #xyn_tmp = tmp_path + '/' + str(xi) + '_' + str(i) + '_' + str(yi) + '_tmp'
        #tmp = csrmm_ez(x, y, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)
        # try:
        #    z += tmp
        # except:
        #    z = tmp

        #del x, y, tmp
        del x, y
        gc.collect()
    if type(xr) != type(None) and type(yc) != type(None):
        xyn_tmp = tmp_path + '/' + str(xi) + '_x_' + str(yi) + '_tmp'
        z = csrmm_ez(xr, yc, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)
    else:
        return None, None, None

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    #z.data[norm_dat < prune] = 0

    P = int(1. / prune) + 1
    # print 'element_fk_P', prune, P
    select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz





def element_fast(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            if type(xr) == type(None):
                xr = x
            else:
                xr += x
        except:
            print 'can not load x', xn
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
            if type(yc) == type(None):
                yc = y
            else:
                yc += y

        except:
            print 'can not load y', yn
            continue

        del x, y
        gc.collect()
    if type(xr) != type(None) and type(yc) != type(None):
        xyn_tmp = tmp_path + '/' + str(xi) + '_x_' + str(yi) + '_tmp'
        #z = csrmm_ez(xr, yc, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)
        z = csrmm_ez_ms(xr, yc, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)

    else:
        return None, None, None

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    #z.data[norm_dat < prune] = 0

    #P = int(1. / prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)

    tmpfn = tmp_path + '/' + str(xi) + '_x_' + str(yi) + '_tmp_*_ms.npy'
    print 'try_remove_tmpfn', tmpfn
    os.system('rm %s'%tmpfn)

    try:
        z.indptr._mmap.close()
    except:
        pass

    try:
        z.indices._mmap.close()
    except:
        pass

    try:
        z.data._mmap.close()
    except:
        pass


    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def relement_fast(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz_Mg.npz'
        print 'xi', xi, 'yi', yi, xn, yn
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            if type(xr) == type(None):
                xr = x
            else:
                xr += x
        except:
            print 'can not load x', xn
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
            if type(yc) == type(None):
                yc = y
            else:
                yc += y

        except:
            print 'can not load y', yn
            continue

        #xyn_tmp = tmp_path + '/' + str(xi) + '_' + str(i) + '_' + str(yi) + '_tmp'
        #tmp = csrmm_ez(x, y, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)
        # try:
        #    z += tmp
        # except:
        #    z = tmp

        #del x, y, tmp
        del x, y
        gc.collect()
    if type(xr) != type(None) and type(yc) != type(None):
        xyn_tmp = tmp_path + '/' + str(xi) + '_x_' + str(yi) + '_tmp'
        z = csrmm_ez(xr, yc, cpu=cpu, prefix=xyn_tmp, tmp_path=tmp_path)
    else:
        return None, None, None

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    #z.data[norm_dat < prune] = 0

    P = int(1. / prune) + 1
    # print 'element_fk_P', prune, P
    select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


# bmat
def bkmat0(xyns, cpu=1):
    print 'working on block mat', xyns
    z = None
    for xyn in xyns:
        xn, yn, shape, csr, tmp_path = xyn
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            if xn == yn:
                y = x
            else:
                y = load_matrix(yn, shape=shape, csr=csr)
            print 'bkmat loading', xn, yn
        except:
            print 'not get', xn, yn
            # return None
            continue

        z0 = csrmm_ez(x, y, tmp_path=tmp_path, cpu=1)
        if type(z) != type(None):
            z += z0
        else:
            z = z0
    # print 'get z', z
    return z


def bkmat1(xyns, cpu=1, ms=True):
    print 'working on block mat', xyns
    z = None
    X = None
    Y = None
    for xyn in xyns:
        xn, yn, shape, csr, tmp_path = xyn
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            if xn == yn:
                y = x
            else:
                y = load_matrix(yn, shape=shape, csr=csr)
            print 'bkmat loading', xn, yn
        except:
            print 'not get', xn, yn
            # return None
            continue
        if type(X) == type(None):
            X = x
        else:
            X += x

        if type(Y) == type(None):
            Y = y
        else:
            Y += y

    
    try:
        z = csrmm_ez_ms(X, Y, tmp_path=tmp_path, cpu=1)
    except:
        z = None
    # print 'get z', z
    return z




def bkmat(xyns, cpu=1, ms=True):
    print 'working on block mat', xyns
    z = None
    X = None
    Y = None
    prefix = None
    for xyn in xyns:
        #xn, yn, shape, csr, tmp_path, xi, yi = xyn
        xn, yn, shape, csr, tmp_path, xi, yi, label = xyn
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
            if xn == yn:
                y = x
            else:
                y = load_matrix(yn, shape=shape, csr=csr)
            print 'bkmat loading', xn, yn
        except:
            print 'not get', xn, yn
            # return None
            continue
        if type(X) == type(None):
            X = x
        else:
            X += x

        if type(Y) == type(None):
            Y = y
        else:
            Y += y

        prefix = tmp_path + '/' + str(xi) + '_' + str(yi) + '_' + str(label) + '_'
   
    try:
        prefix += tempfile.mkstemp()[1].split(os.sep)[-1]
    except:
        pass

    try:
        z = csrmm_ez_ms(X, Y, prefix=prefix, tmp_path=tmp_path, cpu=1)
        #z = csrmm_ez(X, Y, prefix=prefix, tmp_path=tmp_path, cpu=1)

    except:
        z = None
    # print 'get z', z
    return z



def badd0(xy):
    x, y = xy
    z = x + y
    del x, y
    gc.collect()
    return z


def badd(xy):
    #x, y = xy
    #z = x + y
    #del x, y
    # gc.collect()
    # return z
    z = None
    for i in xy:
        if type(z) == type(None):
            z = i
        else:
            z += i
        try:
            i._mmap.close()
        except:
            pass
        del i
        gc.collect()

    return z

# block merge


def bmerge0(zs, cpu=1):
    if len(zs) == 1:
        return zs

    while len(zs) > 1:
        print 'working on bmerge', len(zs)
        xys = []
        unpair = []
        while len(zs) > 0:
            z0 = zs.pop()
            if type(z0) == type(None):
                continue

            try:
                z1 = zs.pop()
            except:
                z1 = None
            if type(z1) != type(None):
                xys.append([z0, z1])
            else:
                try:
                    unpair.append(z0)
                except:
                    unpair = [z0]
        # if cpu <= 1:
        if cpu <= 1 or len(xys) <= 1:
            new_zs = map(badd, xys)
        else:
            new_zs = Parallel(n_jobs=cpu)(delayed(badd)(elem) for elem in xys)

        while len(new_zs) > 0:
            z = new_zs.pop()
            if type(z) != type(None):
                unpair.append(z)

        zs = unpair
    try:
        return zs[0]
    except:
        return None


# block merge
def bmerge(zs, cpu=1):
    if len(zs) == 1:
        return zs[0]

    z = None
    if cpu <= 1:
        return badd(zs)

    while len(zs) > 1:
        print 'working on bmerge', len(zs)
        xys = []
        unpair = []
        tmp = []
        while zs:
            tmp.append(zs.pop())
            if len(tmp) >= 4:
                xys.append(tmp)
                tmp = []
        if len(tmp) > 1:
            xys.append(tmp)
        else:
            unpair = tmp

        # if cpu <= 1:
        if cpu <= 1 or len(xys) <= 1:
            new_zs = map(badd, xys)
        else:
            new_zs = Parallel(n_jobs=cpu)(delayed(badd)(elem) for elem in xys)

        while len(new_zs) > 0:
            z = new_zs.pop()
            if type(z) != type(None):
                unpair.append(z)

        zs = unpair
    try:
        return zs[0]
    except:
        return None


# disk based matrix add function
def badd_disk(xyzs):
    #x, y = xy
    #z = x + y
    #del x, y
    # gc.collect()
    # return z
    z = None
    idx = None
    for i in xyzs:
        if type(z) == type(None):
            z = sparse.load_npz('tmp_mat_%d.npz' % i)
            idx = i
        else:
            z += sparse.load_npz('tmp_mat_%d.npz' % i)

        os.system('rm tmp_mat_%d.npz' % i)

    if type(z) != type(None) and idx != None:
        sparse.save_npz('tmp_mat_%d' % idx, z)
        del z
        gc.collect()

    return idx


# disk based merge function
def bmerge_disk(zs, cpu=1):
    # write z to disk
    N = len(zs)
    Nraw = N
    if N == 1:
        return zs[0]

    Ns = range(N)
    for i in Ns:
        sparse.save_npz('tmp_mat_%d.npz' % i, zs[i])

    del zs
    gc.collect()

    zs = Ns
    while len(zs) > 1:
        print 'working on bmerge', len(zs)
        xys = []
        unpair = []
        for idx in xrange(0, len(zs), 4):
            if len(zs[idx:idx + 4]) > 1:
                xys.append(zs[idx:idx + 4])
            else:
                unpair.append(zs[idx:idx + 4])

        # if cpu <= 1:
        if cpu <= 1 or len(xys) <= 1:
            new_zs = map(badd_disk, xys)
        else:
            new_zs = Parallel(n_jobs=cpu)(delayed(badd_disk)(elem)
                                          for elem in xys)

        # print 'unfinished_merge0', new_zs, xys, unpair, Nraw
        for un in unpair:
            new_zs.extend(un)

        # print 'unfinished_merge1', new_zs, xys, unpair, Nraw

        zs = [elem for elem in new_zs if elem != None]
        # print 'unfinished_merge_flt', zs, Nraw

    print 'finish_merge', zs
    try:
        # return zs[0]
        idx = zs[0]
        z = sparse.load_npz('tmp_mat_%d.npz' % idx)
        os.system('rm tmp_mat_%d.npz' % idx)
    except:
        z = None

    return z


# processing entry blocks one by one
def element5(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    xyn = [[] for elem in xrange(cpu)]
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        # print 'xi', xi, 'yi', yi
        if os.path.isfile(xn) and os.path.isfile(yn):
            #xyn.append([xn, yn, shape, csr])
            xyn[i % cpu].append([xn, yn, shape, csr, tmp_path])
            print 'in_bkt', i, cpu, i % cpu

    xyn = [elem for elem in xyn if elem]

    print 'compute_element_cpu', cpu
    print 'parallel_bmat', xyn[0], len(xyn)
    #zs = bmat(xyns, cpu)
    # if cpu <= 1:
    # if 1:
    #    zs = map(bkmat, xyn)
    # else:
    #    zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    if len(xyn) > 1 and cpu > 1:
        zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    else:
        zs = map(bkmat, xyn)

    z = bmerge(zs, cpu=cpu)
    #z = bmerge_disk(zs, cpu=cpu)
    # print 'breakpoint', zs, z
    #raise SystemExit()
    if type(z) == type(None):
        # print 'return_none_z'
        return None, None, None
    # else:
    #    z = zs_merge(zs)

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')

    #z.data[norm_dat < prune] = 0
    #P = int(1./prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz





def relement0(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    xyn = [[] for elem in xrange(cpu)]
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz_Mg.npz'
        # print 'xi', xi, 'yi', yi
        if os.path.isfile(xn) and os.path.isfile(yn):
            #xyn.append([xn, yn, shape, csr])
            xyn[i % cpu].append([xn, yn, shape, csr, tmp_path])
            print 'in_bkt', i, cpu, i % cpu

    xyn = [elem for elem in xyn if elem]

    print 'compute_element_cpu', cpu
    print 'parallel_bmat', xyn[0], len(xyn)
    #zs = bmat(xyns, cpu)
    # if cpu <= 1:
    # if 1:
    #    zs = map(bkmat, xyn)
    # else:
    #    zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    if len(xyn) > 1 and cpu > 1:
        zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    else:
        zs = map(bkmat, xyn)

    z = bmerge(zs, cpu=cpu)
    #z = bmerge_disk(zs, cpu=cpu)
    # print 'breakpoint', zs, z
    #raise SystemExit()
    if type(z) == type(None):
        # print 'return_none_z'
        return None, None, None
    # else:
    #    z = zs_merge(zs)

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    #z.data[norm_dat < prune] = 0

    #P = int(1./prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz



def element6(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    xyn = [[] for elem in xrange(cpu)]
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        # print 'xi', xi, 'yi', yi
        if os.path.isfile(xn) and os.path.isfile(yn):
            #xyn.append([xn, yn, shape, csr])
            xyn[i % cpu].append([xn, yn, shape, csr, tmp_path])
            print 'in_bkt', i, cpu, i % cpu

    xyn = [elem for elem in xyn if elem]

    print 'compute_element_cpu', cpu
    print 'parallel_bmat', xyn[0], len(xyn)
    #zs = bmat(xyns, cpu)
    # if cpu <= 1:
    # if 1:
    #    zs = map(bkmat, xyn)
    # else:
    #    zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    if len(xyn) > 1 and cpu > 1:
        zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    else:
        zs = map(bkmat, xyn)

    z = bmerge(zs, cpu=cpu)
    #z = bmerge_disk(zs, cpu=cpu)
    # print 'breakpoint', zs, z
    #raise SystemExit()
    if type(z) == type(None):
        # print 'return_none_z'
        return None, None, None
    # else:
    #    z = zs_merge(zs)

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')

    #z.data[norm_dat < prune] = 0
    #P = int(1./prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz





def element(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    xyn = [[] for elem in xrange(cpu)]
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        # print 'xi', xi, 'yi', yi
        if os.path.isfile(xn) and os.path.isfile(yn):
            #xyn.append([xn, yn, shape, csr])
            label = i % cpu
            xyn[label].append([xn, yn, shape, csr, tmp_path, xi, yi, label])
            print 'in_bkt', i, cpu, i % cpu

    xyn = [elem for elem in xyn if elem]

    print 'compute_element_cpu', cpu
    print 'parallel_bmat', xyn[0], len(xyn)
    #zs = bmat(xyns, cpu)
    # if cpu <= 1:
    # if 1:
    #    zs = map(bkmat, xyn)
    # else:
    #    zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)

    if len(xyn) > 1 and cpu > 1:
        zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    else:
        zs = map(bkmat, xyn)

    #zs = map(bkmat, xyn)

    z = bmerge(zs, cpu=cpu)
    #z = bmerge_disk(zs, cpu=cpu)
    # print 'breakpoint', zs, z
    #raise SystemExit()
    if type(z) == type(None):
        # print 'return_none_z'
        return None, None, None
    # else:
    #    z = zs_merge(zs)

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')

    #z.data[norm_dat < prune] = 0
    #P = int(1./prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z


    tmpfn = tmp_path + '/' + str(xi) + '_' + str(yi) + '_*_*ms.npy'
    os.system('rm %s'%tmpfn)

    gc.collect()

    return row_sum_n, xyn, nnz


def relement1(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    xyn = [[] for elem in xrange(cpu)]
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz_Mg.npz'
        # print 'xi', xi, 'yi', yi
        if os.path.isfile(xn) and os.path.isfile(yn):
            #xyn.append([xn, yn, shape, csr])
            xyn[i % cpu].append([xn, yn, shape, csr, tmp_path])
            print 'in_bkt', i, cpu, i % cpu

    xyn = [elem for elem in xyn if elem]

    print 'compute_element_cpu', cpu
    print 'parallel_bmat', xyn[0], len(xyn)
    #zs = bmat(xyns, cpu)
    # if cpu <= 1:
    # if 1:
    #    zs = map(bkmat, xyn)
    # else:
    #    zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    if len(xyn) > 1 and cpu > 1:
        zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    else:
        zs = map(bkmat, xyn)

    z = bmerge(zs, cpu=cpu)
    #z = bmerge_disk(zs, cpu=cpu)
    # print 'breakpoint', zs, z
    #raise SystemExit()
    if type(z) == type(None):
        # print 'return_none_z'
        return None, None, None
    # else:
    #    z = zs_merge(zs)

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    #z.data[norm_dat < prune] = 0

    #P = int(1./prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz




def relement(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xr = yc = z = None
    xyn = [[] for elem in xrange(cpu)]
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz_Mg.npz'
        # print 'xi', xi, 'yi', yi
        if os.path.isfile(xn) and os.path.isfile(yn):
            #xyn.append([xn, yn, shape, csr])
            xyn[i % cpu].append([xn, yn, shape, csr, tmp_path, xi, yi])
            print 'in_bkt', i, cpu, i % cpu

    xyn = [elem for elem in xyn if elem]

    print 'compute_element_cpu', cpu
    print 'parallel_bmat', xyn[0], len(xyn)
    #zs = bmat(xyns, cpu)
    # if cpu <= 1:
    # if 1:
    #    zs = map(bkmat, xyn)
    # else:
    #    zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    if len(xyn) > 1 and cpu > 1:
        zs = Parallel(n_jobs=cpu)(delayed(bkmat)(elem) for elem in xyn)
    else:
        zs = map(bkmat, xyn)

    z = bmerge(zs, cpu=cpu)
    #z = bmerge_disk(zs, cpu=cpu)
    # print 'breakpoint', zs, z
    #raise SystemExit()
    if type(z) == type(None):
        # print 'return_none_z'
        return None, None, None
    # else:
    #    z = zs_merge(zs)

    z.data **= I
    z.eliminate_zeros()

    # remove element < prune
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    #row_sum = np.asarray(z.max(0).todense(), 'float32')[0]

    #norm_dat = z.data / row_sum.take(z.indices, mode='clip')
    #z.data[norm_dat < prune] = 0

    #P = int(1./prune) + 1
    # print 'element_fk_P', prune, P
    #select_jit(z.indices, z.indptr, z.data, S=P)

    z.eliminate_zeros()

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    del z
    gc.collect()

    return row_sum_n, xyn, nnz









# use gpu to speed up
def element_gpu0(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    x = y = None
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            xt = load_matrix(xn, shape=shape, csr=csr)
        except:
            xt = None
        if type(xt) != type(None):
            try:
                x += xt
            except:
                x = xt

        try:
            yt = load_matrix(yn, shape=shape, csr=csr)
        except:
            yt = None

        if type(yt) != type(None):
            try:
                y += yt
            except:
                y = yt
        del xt, yt
        gc.collect()

    try:
        xg, yg = map(cp.sparse.csr_matrix, [x, y])
        zg = cp.cusparse.csrgemm(xg, yg)
        zg.data **= I
        zg.data[zg.data < prune] = 0
        z = zg.get()
        z.eliminate_zeros()

    except:
        z = None

    del x, y, xg, yg, zg
    gc.collect()

    if type(z) == type(None):
        return None, None, None

    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)

    # return row_sum
    #row_sum = z.sum(0)
    row_sum = np.asarray(z.sum(0), 'float32')[0]
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    # print 'row_sum is', type(row_sum)
    # return row_sum, xyn, nnz
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def element_gpu(xi, yi, d, qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xg = cp.sparse.csr_matrix(shape)
    yg = cp.sparse.csr_matrix(shape)
    #zg = cp.sparse.csr_matrix(shape)
    zg = None
    #tmp = cp.sparse.csr_matrix(shape)
    for i in xrange(d):
        xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
        yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
        print 'xi', xi, 'yi', yi
        try:
            x = load_matrix(xn, shape=shape, csr=csr)
        except:
            print 'can not load x', xn, csr
            continue
        try:
            y = load_matrix(yn, shape=shape, csr=csr)
        except:
            print 'can not load y', yn, csr
            continue

        a, b, c = map(cp.asarray, [x.indices, x.indptr, x.data])
        xg.indices, xg.indptr, xg.data = a, b, c

        a, b, c = map(cp.asarray, [y.indices, y.indptr, y.data])
        yg.indices, yg.indptr, yg.data = a, b, c

        tmp = cp.cusparse.csrgemm(xg, yg)
        try:
            zg += tmp
        except:
            zg = tmp

        del x, y, a, b, c
        gc.collect()

    if type(zg) == type(None):
        return None, None, None

    zg.data **= I
    #zg.data[zg.data < prune] = 0

    z = zg.get()
    row_sum = np.asarray(zg.sum(0).get(), 'float32')[0]

    del zg, tmp
    gc.collect()
    z.eliminate_zeros()
    nnz = z.nnz
    xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
    sparse.save_npz(xyn + '_new', z)

    # return row_sum
    #row_sum = z.sum(0)
    #row_sum = np.asarray(z.sum(0), 'float32')[0]
    row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
    np.savez_compressed(row_sum_n, row_sum)
    # print 'row_sum is', type(row_sum)
    # return row_sum, xyn, nnz
    del z
    gc.collect()

    return row_sum_n, xyn, nnz


def element_wrapper0(elem):
    x, y, d, qry, shape, tmp_path, csr, I, prune = elem
    return element(x, y, d, qry, shape, tmp_path, csr, I, prune)


def element_wrapper1(elems):
    outs = []
    for elem in elems:
        x, y, d, qry, shape, tmp_path, csr, I, prune = elem
        out = element(x, y, d, qry, shape, tmp_path, csr, I, prune)
        outs.append(out)
    return outs


def element_wrapper(elems):
    outs = []
    for elem in elems:
        x, y, d, qry, shape, tmp_path, csr, I, prune, cpu, fast = elem

        print 'tmp_path', tmp_path
        if fast:
            out = element_fast(x, y, d, qry, shape,
                               tmp_path, csr, I, prune, cpu)
        else:
            out = element(x, y, d, qry, shape, tmp_path, csr, I, prune, cpu)
        outs.append(out)
    return outs


def relement_wrapper(elems):
    outs = []
    for elem in elems:
        x, y, d, qry, shape, tmp_path, csr, I, prune, cpu, fast = elem

        print 'tmp_path', tmp_path
        if fast:
            out = relement_fast(x, y, d, qry, shape,
                                tmp_path, csr, I, prune, cpu)
        else:
            out = relement(x, y, d, qry, shape, tmp_path, csr, I, prune, cpu)
        outs.append(out)
    return outs


def element_wrapper_gpu0(elems):
    outs = []
    for elem in elems:
        x, y, d, qry, shape, tmp_path, csr, I, prune = elem
        out = element_gpu(x, y, d, qry, shape, tmp_path, csr, I, prune)
        outs.append(out)
    return outs


def element_wrapper_gpu1(elems):
    if len(elems) == 0:
        return []
    elem = elems[0]
    x, y, d, qry, shape, tmp_path, csr, I, prune = elem
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xg = cp.sparse.csr_matrix(shape, dtype='float32')
    yg = cp.sparse.csr_matrix(shape, dtype='float32')
    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x', xn
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y', yn
                continue

            #a, b, c = map(cp.asarray, [x.indices, x.indptr, x.data])
            a = cp.asarray(x.indices, dtype=cp.int32)
            b = cp.asarray(x.indptr, dtype=cp.int32)
            c = cp.asarray(x.data, dtype=cp.float32)
            xg.indices, xg.indptr, xg.data = a, b, c

            #a, b, c = map(cp.asarray, [y.indices, y.indptr, y.data])
            a = cp.asarray(y.indices, dtype=cp.int32)
            b = cp.asarray(y.indptr, dtype=cp.int32)
            c = cp.asarray(y.data, dtype=cp.float32)
            yg.indices, yg.indptr, yg.data = a, b, c

            tmp = cp.cusparse.csrgemm(xg, yg)
            try:
                zg += tmp
            except:
                zg = tmp

            del x, y, a, b, c, tmp
            gc.collect()
            cp.cuda.memory.gc.collect()

        if type(zg) == type(None):
            return None, None, None

        zg.data **= I
        zg.data[zg.data < prune] = 0

        z = zg.get()
        row_sum = np.asarray(zg.sum(0).get(), 'float32')[0]

        #del zg, tmp
        del zg
        gc.collect()
        cp.cuda.memory.gc.collect()

        z.eliminate_zeros()
        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    return outs


def element_wrapper_gpu2(elems):
    if len(elems) == 0:
        return []
    elem = elems[0]
    x, y, d, qry, shape, tmp_path, csr, I, prune = elem
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    xg = cp.sparse.csr_matrix(shape, dtype='float32')
    yg = cp.sparse.csr_matrix(shape, dtype='float32')
    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        z = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x', xn
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y', yn
                continue

            #a, b, c = map(cp.asarray, [x.indices, x.indptr, x.data])
            a = cp.asarray(x.indices, dtype=cp.int32)
            b = cp.asarray(x.indptr, dtype=cp.int32)
            c = cp.asarray(x.data, dtype=cp.float32)
            xg.indices, xg.indptr, xg.data = a, b, c

            #a, b, c = map(cp.asarray, [y.indices, y.indptr, y.data])
            a = cp.asarray(y.indices, dtype=cp.int32)
            b = cp.asarray(y.indptr, dtype=cp.int32)
            c = cp.asarray(y.data, dtype=cp.float32)
            yg.indices, yg.indptr, yg.data = a, b, c

            tmp = cp.cusparse.csrgemm(xg, yg)
            if type(zg) != type(None):
                zg += tmp
            else:
                zg = tmp

            if zg.nnz >= 2 * 10**8:
                if type(z) != type(None):
                    z += zg.get()
                else:
                    z = zg.get()
                zg = None

            del x, y, a, b, c, tmp
            gc.collect()
            cp.cuda.memory.gc.collect()

        if type(zg) != type(None):
            if type(z) != type(None):
                z += zg.get()
            else:
                z = zg.get()
            zg = None

        if type(z) == type(None):
            return None, None, None

        z.eliminate_zeros()
        z.data **= I
        z.data[z.data < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    return outs


# use pyculib instead of cupy
def element_wrapper_gpu3(elems):

    clf = pyculib.sparse.Sparse()

    if len(elems) == 0:
        return []
    elem = elems[0]
    x, y, d, qry, shape, tmp_path, csr, I, prune = elem
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        z = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x', xn
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y', yn
                continue

            print 'running on gpu'
            tmp = clf.csrgemm_ez(x, y)
            if type(zg) != type(None):
                #zg += tmp
                zg = csrgeam_ez(zg, tmp, clf=clf)
            else:
                zg = tmp

            if zg.nnz >= 2.5e7:
                # if zg.nnz >= 10**5:
                print 'copy to host', i, zg.nnz
                if type(z) != type(None):
                    #z += zg.get()
                    z += zg.copy_to_host()
                else:
                    #z = zg.get()
                    z = zg.copy_to_host()
                del zg
                zg = None
                gc.collect()

            del x, y, tmp
            gc.collect()

        if type(zg) != type(None):
            if type(z) != type(None):
                #z += zg.get()
                z += zg.copy_to_host()
            else:
                #z = zg.get()
                z = zg.copy_to_host()

            del zg
            zg = None
            gc.collect()

        if type(z) == type(None):
            return None, None, None

        z.eliminate_zeros()
        z.data **= I
        z.data[z.data < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    return outs


# correct out of video memory error when using gpu
def element_wrapper_gpu4(elems):

    clf = pyculib.sparse.Sparse()

    if len(elems) == 0:
        return []
    elem = elems[0]
    x, y, d, qry, shape, tmp_path, csr, I, prune = elem
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        z = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x', xn
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y', yn
                continue

            print 'running on gpu'
            try:
                tmp = clf.csrgemm_ez(x, y)
                gpu = 1
            except:
                gpu = 0

            if gpu == 1:
                if type(zg) != type(None):
                    #zg += tmp
                    try:
                        zg = csrgeam_ez(zg, tmp, clf=clf)
                    except:
                        gpu = 0
                else:
                    zg = tmp

                if gpu == 1:
                    if zg.nnz >= 2.5e7:
                        # if zg.nnz >= 10**5:
                        print 'copy to host', i, zg.nnz
                        if type(z) != type(None):
                            #z += zg.get()
                            z += zg.copy_to_host()
                        else:
                            #z = zg.get()
                            z = zg.copy_to_host()

                        del zg
                        zg = None
                        gc.collect()

                else:
                    if type(z) != type(None):
                        z += zg.copy_to_host()
                    else:
                        z = zg.copy_to_host()
                    z = tmp.copy_to_host()
                    del zg
                    zg = None
                    gc.collect()

                del x, y, tmp

            else:
                if type(z) != type(None):
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()

                    z += x * y
                else:
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()
                        z += x * y
                    else:
                        z = x * y

                del x, y

        gc.collect()
        if type(zg) != type(None):
            if type(z) != type(None):
                #z += zg.get()
                z += zg.copy_to_host()
            else:
                #z = zg.get()
                z = zg.copy_to_host()

            del zg
            zg = None
            gc.collect()

        if type(z) == type(None):
            return None, None, None

        z.eliminate_zeros()
        z.data **= I
        z.data[z.data < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    return outs


# add multiple gpu support
def element_wrapper_gpu5(elems):

    if len(elems) <= 1:
        return []

    # init gpu
    gid = elems[0] % len(pyculib.cuda.devices.gpus.lst)
    pyculib.cuda.close()
    pyculib.cuda.select_device(gid)
    clf = pyculib.sparse.Sparse()

    x, y, d, qry, shape, tmp_path, csr, I, prune = elems[1]
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems[1:]:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        z = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x', xn
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y', yn
                continue

            print 'running on gpu'
            try:
                #tmp = clf.csrgemm_ez(x, y)
                tmp = csrgemm_ez(x, y)
                gpu = 1
            except:
                gpu = 0

            if gpu == 1:
                if type(zg) != type(None):
                    #zg += tmp
                    try:
                        zg = csrgeam_ez(zg, tmp, clf=clf)
                    except:
                        gpu = 0
                else:
                    zg = tmp

                if gpu == 1:
                    if zg.nnz >= 2.5e7:
                        # if zg.nnz >= 10**5:
                        print 'copy to host', i, zg.nnz
                        if type(z) != type(None):
                            #z += zg.get()
                            z += zg.copy_to_host()
                        else:
                            #z = zg.get()
                            z = zg.copy_to_host()

                        del zg
                        zg = None
                        gc.collect()

                else:
                    if type(z) != type(None):
                        z += zg.copy_to_host()
                    else:
                        z = zg.copy_to_host()
                    z = tmp.copy_to_host()
                    del zg
                    zg = None
                    gc.collect()

                del x, y, tmp

            else:
                if type(z) != type(None):
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()

                    z += x * y
                else:
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()
                        z += x * y
                    else:
                        z = x * y

                del x, y

        gc.collect()
        if type(zg) != type(None):
            if type(z) != type(None):
                #z += zg.get()
                z += zg.copy_to_host()
            else:
                #z = zg.get()
                z = zg.copy_to_host()

            del zg
            zg = None
            gc.collect()

        if type(z) == type(None):
            # print 'z is none, wtf'
            # return None, None, None
            #outs.append([None, None, None])
            continue

        z.eliminate_zeros()
        z.data **= I
        z.data[z.data < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    pyculib.cuda.close()
    return outs


# adjust prune step
def element_wrapper_gpu6(elems):

    if len(elems) <= 1:
        return []

    # init gpu
    try:
        gid = elems[0] % len(pyculib.cuda.devices.gpus.lst)
        pyculib.cuda.close()
        pyculib.cuda.select_device(gid)
        csrgemm_ez = pyculib.sparse.Sparse().csrgemm_ez
        clf = pyculib.sparse.Sparse()
        flag_gpu = 1
    except:
        clf = None
        flag_gpu = 0
        csrgemm_ez = lambda x, y: x * y

        print 'gpu disable gid', elems[0] % len(pyculib.cuda.devices.gpus.lst), pyculib.sparse.Sparse()

    x, y, d, qry, shape, tmp_path, csr, I, prune = elems[1]
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems[1:]:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        z = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
                #x_i, x_j = x.nonzero()
                #x.data[x_i==x_j] = 1
                print 'before pruning x nnz', x.nnz
                #x.data[x.data<prune] = 0
                # x.eliminate_zeros()
                print 'after pruning x nnz', x.nnz

            except:
                print 'can not load x', xn
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
                #y_i, y_j = y.nonzero()
                #y.data[y_i==y_j] = 1
                print 'before pruning y nnz', y.nnz
                #y.data[y.data<prune] = 0
                # y.eliminate_zeros()
                print 'after pruning y nnz', y.nnz

            except:
                print 'can not load y', yn
                continue

            if flag_gpu == 1:
                # print 'running on gpu', csrgemm_ez, csrgemm_ez(x, y).shape
                print 'running on gpu', csrgemm_ez
                try:
                    #tmp = clf.csrgemm_ez(x, y)
                    tmp = csrgemm_ez(x, y)
                    gpu = 1
                except:
                    gpu = 0
            else:
                gpu = 0

            if gpu == 1:
                if type(zg) != type(None):
                    #zg += tmp
                    try:
                        zg = csrgeam_ez(zg, tmp, clf=clf)
                    except:
                        gpu = 0
                else:
                    zg = tmp

                if gpu == 1:
                    if zg.nnz >= 1.5e8:
                        # if zg.nnz >= 10**5:
                        print 'copy to host', i, zg.nnz
                        if type(z) != type(None):
                            #z += zg.get()
                            z += zg.copy_to_host()
                        else:
                            #z = zg.get()
                            z = zg.copy_to_host()

                        del zg
                        zg = None
                        gc.collect()

                else:
                    if type(z) != type(None):
                        z += zg.copy_to_host()
                    else:
                        z = zg.copy_to_host()
                    z = tmp.copy_to_host()
                    del zg
                    zg = None
                    gc.collect()

                del x, y, tmp

            else:
                if type(z) != type(None):
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()

                    z += x * y
                else:
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()
                        z += x * y
                    else:
                        z = x * y

                del x, y

        gc.collect()
        if type(zg) != type(None):
            print 'copy from device to host'
            if type(z) != type(None):
                #z += zg.get()
                z += zg.copy_to_host()
            else:
                #z = zg.get()
                z = zg.copy_to_host()

            del zg
            zg = None
            gc.collect()

        if type(z) == type(None):
            # return None, None, None
            # continue
            #outs.append([None, None, None])
            continue

        z.eliminate_zeros()
        z.data **= I
        #z.data[z.data < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    try:
        pyculib.cuda.close()
    except:
        pass
    return outs


# adjust prune
def element_wrapper_gpu7(elems):

    if len(elems) <= 1:
        return []

    # init gpu
    try:
        gid = elems[0] % len(pyculib.cuda.devices.gpus.lst)
        pyculib.cuda.close()
        pyculib.cuda.select_device(gid)
        csrgemm_ez = pyculib.sparse.Sparse().csrgemm_ez
        clf = pyculib.sparse.Sparse()
        flag_gpu = 1
    except:
        clf = None
        flag_gpu = 0
        csrgemm_ez = lambda x, y: x * y

        print 'gpu disable gid', elems[0] % len(pyculib.cuda.devices.gpus.lst), pyculib.sparse.Sparse()

    x, y, d, qry, shape, tmp_path, csr, I, prune = elems[1]
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems[1:]:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = None
        z = None
        #tmp = cp.sparse.csr_matrix(shape)
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
                #x_i, x_j = x.nonzero()
                #x.data[x_i==x_j] = 1
                # print 'before pruning x nnz', x.nnz
                #x.data[x.data<prune] = 0
                # x.eliminate_zeros()
                # print 'after pruning x nnz', x.nnz

            except:
                print 'can not load x elem_wrapper_gpu', xn, csr
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
                #y_i, y_j = y.nonzero()
                #y.data[y_i==y_j] = 1
                # print 'before pruning y nnz', y.nnz
                #y.data[y.data<prune] = 0
                # y.eliminate_zeros()
                # print 'after pruning y nnz', y.nnz

            except:
                print 'can not load y elem_wrapper_gpu', yn, csr
                continue

            if flag_gpu == 1:
                # print 'running on gpu', csrgemm_ez, csrgemm_ez(x, y).shape
                print 'running on gpu', csrgemm_ez
                try:
                    #tmp = clf.csrgemm_ez(x, y)
                    tmp = csrgemm_ez(x, y)
                    gpu = 1
                except:
                    gpu = 0
            else:
                gpu = 0

            if gpu == 1:
                if type(zg) != type(None):
                    #zg += tmp
                    try:
                        zg = csrgeam_ez(zg, tmp, clf=clf)
                    except:
                        gpu = 0
                else:
                    zg = tmp

                if gpu == 1:
                    if zg.nnz >= 1.5e8:
                        # if zg.nnz >= 10**5:
                        print 'copy to host', i, zg.nnz
                        if type(z) != type(None):
                            #z += zg.get()
                            z += zg.copy_to_host()
                        else:
                            #z = zg.get()
                            z = zg.copy_to_host()

                        del zg
                        zg = None
                        gc.collect()

                else:
                    if type(z) != type(None):
                        z += zg.copy_to_host()
                    else:
                        z = zg.copy_to_host()
                    z = tmp.copy_to_host()
                    del zg
                    zg = None
                    gc.collect()

                print 'x nnz', x.nnz, y.nnz, tmp.nnz
                del x, y, tmp

            else:
                if type(z) != type(None):
                    if type(zg) != type(None):
                        z += zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()

                    z += x * y
                else:
                    if type(zg) != type(None):
                        try:
                            z += zg.copy_to_host()
                        except:
                            z = zg.copy_to_host()
                        del zg
                        zg = None
                        gc.collect()
                        z += x * y
                    else:
                        z = x * y

                del x, y

        gc.collect()
        if type(zg) != type(None):
            print 'copy from device to host'
            if type(z) != type(None):
                #z += zg.get()
                z += zg.copy_to_host()
            else:
                #z = zg.get()
                z = zg.copy_to_host()

            del zg
            zg = None
            gc.collect()

        if type(z) == type(None):
            # return None, None, None
            # continue
            #outs.append([None, None, None])
            continue

        z.eliminate_zeros()
        z.data **= I
        #z.data[z.data < prune] = 0
        z.eliminate_zeros()

        # remove element < prune
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        norm_dat = z.data / row_sum.take(z.indices, mode='clip')
        z.data[norm_dat < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    try:
        pyculib.cuda.close()
    except:
        pass
    return outs


# use cupy instead of pyculib
def element_wrapper_gpu8(elems):

    if len(elems) <= 1:
        return []

    # init gpu
    device = len(cuda.gpus.lst)
    gid = elems[0] % device
    cp.cuda.Device(gid).use()
    #csrmm = lambda x,y: cp.cusparse.csrgemm(cp.sparse.csr_matrix(x), cp.sparse.csr_matrix(y))

    x, y, d, qry, shape, tmp_path, csr, I, prune = elems[1]
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    #zg = cp.sparse.csr_matrix(shape)
    for elem in elems[1:]:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = cp.sparse.csr_matrix(shape, dtype='float32')
        z = sparse.csr_matrix(shape, dtype='float32')
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x elem_wrapper_gpu', xn, csr
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)

            except:
                print 'can not load y elem_wrapper_gpu', yn, csr
                continue

            try:
                xc = cp.sparse.csr_matrix(x)
                yc = cp.sparse.csr_matrix(y)
                zg += cp.sparse.csrgemm(xc, yc)
                del x, y, xc, yc
                gc.collect()

                #zg += csrmm(x, y)
            except:
                print 'cp_gpu fail'
                z += zg.get()
                #z += x * y
                z += csrmm_ez(x, y)
                print 'z_nnz', z.nnz
                del zg, x, y
                gc.collect()
                zg = cp.sparse.csr_matrix(shape, dtype='float32')

            if zg.nnz > 5e7:
                z += zg.get()
                del zg
                gc.collect()
                zg = cp.sparse.csr_matrix(shape, dtype='float32')

        gc.collect()
        z += zg.get()
        del zg
        zg = None
        gc.collect()

        if z.nnz == 0:
            # return None, None, None
            # continue
            #outs.append([None, None, None])
            continue

        z.eliminate_zeros()
        z.data **= I
        #z.data[z.data < prune] = 0
        z.eliminate_zeros()

        # remove element < prune
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        norm_dat = z.data / row_sum.take(z.indices, mode='clip')
        z.data[norm_dat < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        # return row_sum
        #row_sum = z.sum(0)
        #row_sum = np.asarray(z.sum(0), 'float32')[0]
        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        # print 'row_sum is', type(row_sum)
        # return row_sum, xyn, nnz
        del z
        gc.collect()
        cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    # try:
    #    pyculib.cuda.close()
    # except:
    #    pass
    return outs


# use pyculib instead of cupy
def element_wrapper_gpu(elems):

    if len(elems) <= 1:
        return []

    # init gpu
    try:
        gid = elems[0] % len(pyculib.cuda.devices.gpus.lst)
        # pyculib.cuda.close()
        pyculib.cuda.select_device(gid)
        clf = pyculib.sparse.Sparse()
        #csrgemm_ez = pyculib.sparse.Sparse().csrgemm_ez
        csrgemm_ez = clf.csrgemm_ez
        has_gpu = 1
    except:
        clf = None
        has_gpu = 0
        #csrgemm_ez = lambda x, y: csrmm_ez(x, y)
        # print 'gpu disable gid', elems[0] %
        # len(pyculib.cuda.devices.gpus.lst), pyculib.sparse.Sparse()
        print 'gpu disable gid', elems[0] % len(pyculib.cuda.devices.gpus.lst)

    x, y, d, qry, shape, tmp_path, csr, I, prune = elems[1]
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    try:
        zg = pyculib.sparse.csr_matrix(shape, dtype='float32')
    except:
        has_gpu = 0

    #has_gpu = 0
    for elem in elems[1:]:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        z = sparse.csr_matrix(shape, dtype='float32')
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x elem_wrapper_gpu', xn, csr
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y elem_wrapper_gpu', yn, csr
                continue
            if has_gpu == 1:
                try:
                    xyg = csrgemm_ez(x, y)
                except:
                    xy = csrmm_ez(x, y)

                try:
                    zg = csrgeam_ez(zg, xyg, clf)
                    del xyg
                except:
                    try:
                        z += xyg.copy_to_host()
                        del xyg
                    except:
                        try:
                            z += xy
                            del xy
                        except:
                            # print 'xyg_exist', xyg.copy_to_host()
                            z += xyg.copy_to_host()
                            del xyg

                    z += zg.copy_to_host()
                    del zg
                    gc.collect()
                    zg = pyculib.sparse.csr_matrix(shape, dtype='float32')

            else:
                z += csrmm_ez(x, y)

            gc.collect()

        if has_gpu == 1:
            try:
                z += zg.copy_to_host()
            except:
                pass

        if z.nnz <= 0:
            continue

        z.eliminate_zeros()
        z.data **= I
        z.eliminate_zeros()

        # remove element < prune
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        norm_dat = z.data / row_sum.take(z.indices, mode='clip')
        #z.data[norm_dat < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        del z
        gc.collect()
        # cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    try:
        pyculib.cuda.close()
    except:
        pass
    return outs


def element_wrapper_gpu9(elems):

    if len(elems) <= 1:
        return []

    # init gpu
    gid = elems[0] % len(pyculib.cuda.devices.gpus.lst)
    pyculib.cuda.select_device(gid)
    clf = pyculib.sparse.Sparse()
    csrgemm_ez = clf.csrgemm_ez
    has_gpu = 1

    x, y, d, qry, shape, tmp_path, csr, I, prune = elems[1]
    outs = []
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    for elem in elems[1:]:
        xi, yi, d, qry, shape, tmp_path, csr, I, prune = elem
        zg = pyculib.sparse.csr_matrix(shape)
        z = sparse.csr_matrix(shape, dtype='float32')
        for i in xrange(d):
            xn = tmp_path + '/' + str(xi) + '_' + str(i) + '.npz'
            yn = tmp_path + '/' + str(i) + '_' + str(yi) + '.npz'
            print 'xi', xi, 'yi', yi
            try:
                x = load_matrix(xn, shape=shape, csr=csr)
            except:
                print 'can not load x elem_wrapper_gpu', xn, csr
                continue
            try:
                y = load_matrix(yn, shape=shape, csr=csr)
            except:
                print 'can not load y elem_wrapper_gpu', yn, csr
                continue
            try:
                xyg = csrgemm_ez(x, y)
            except:
                z += csrmm_ez(x, y)
                continue
            try:
                zg = csrgeam_ez(zg, xyg, clf)
            except:
                z += csrmm_ez(x, y)
            del xyg
            gc.collect()

        #z = sparse.csr_matrix(shape, dtype='float32')
        z += zg.copy_to_host()
        #z = zg.copy_to_host()
        del zg
        gc.collect()
        if z.nnz <= 0:
            continue

        z.eliminate_zeros()
        z.data **= I
        z.eliminate_zeros()

        # remove element < prune
        row_sum = np.asarray(z.sum(0), 'float32')[0]
        norm_dat = z.data / row_sum.take(z.indices, mode='clip')
        z.data[norm_dat < prune] = 0
        z.eliminate_zeros()

        nnz = z.nnz
        xyn = tmp_path + '/' + str(xi) + '_' + str(yi) + '.npz'
        sparse.save_npz(xyn + '_new', z)

        row_sum_n = tmp_path + '/' + str(xi) + '_' + str(yi) + '_rowsum.npz'
        np.savez_compressed(row_sum_n, row_sum)
        del z
        gc.collect()
        # cp.cuda.memory.gc.collect()
        outs.append([row_sum_n, xyn, nnz])

    try:
        pyculib.cuda.close()
    except:
        pass
    return outs


def expand4(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
               for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    # print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    for i in fns:
        # get row
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        xys = []
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            xys.append([xn, yn, shape, csr])

        if len(xys) > 1 and cpu > 1:
            zns = Parallel(n_jobs=cpu)(delayed(sdot)(elem) for elem in xys)
        elif len(xys) == 1:
            zns = [sdot(xys[0])]
        else:
            continue

        Z = None
        for zn in zns:
            if type(zn) == type(None):
                continue
            elif type(zn) == str:
                tmp = load_matrix(zn, shape, csr)
                os.system('rm %s' % zn)
            else:
                tmp = zn
            try:
                Z += tmp
            except:
                Z = tmp

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()
        sparse.save_npz(i + '_new', Z)
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns


def expand5(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    num_set = [elem.split('.')[0].split('_')
               for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    num_set = list(set(sum(num_set, [])))
    num_set.sort(key=lambda x: int(x))
    # print 'num set is', num_set

    row_sum = np.zeros(shape[0], dtype='float32')
    nnz = 0
    for i in fns:
        # get row
        a, b = i.split(os.sep)[-1].split('.')[0].split('_')[:2]
        xys = []
        for j in num_set:

            start = time()
            xn = tmp_path + '/' + a + '_' + j + '.npz'
            yn = tmp_path + '/' + j + '_' + b + '.npz'
            xys.append([xn, yn, shape, csr])

        if len(xys) > 1:
            zns = Parallel(n_jobs=cpu)(delayed(sdot)(elem) for elem in xys)
        elif len(xys) == 1:
            zns = [sdot(xys[0])]
        else:
            continue

        Z = None
        for zn in zns:
            if type(zn) == type(None):
                continue
            elif type(zn) == str:
                tmp = load_matrix(zn, shape, csr)
                os.system('rm %s' % zn)
            else:
                tmp = zn
            try:
                Z += tmp
            except:
                Z = tmp

        Z.data **= I
        Z.data[Z.data < prune] = 0
        Z.eliminate_zeros()
        nnz = max(nnz, Z.nnz)
        sparse.save_npz(i + '_new', Z)
        row_sum += np.asarray(Z.sum(0))[0]

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns, nnz


def expand6(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = []
    for x in xrange(N):
        for y in xrange(N):
            xys.append([x, y, d, qry, shape, tmp_path, csr, I, prune])

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem)
                                   for elem in xys)

    zs = [elem[0] for elem in zns if type(elem[0]) != type(None)]

    rows_sum = sum(zs)
    # print 'rows_sum 0', rows_sum
    rows_sum = np.asarray(rows_sum, 'float32')[0]
    # print 'rows_sum 1', rows_sum

    fns = [elem[1] for elem in zns if type(elem[1]) != type(None)]
    nnz = max([elem[2] for elem in zns])

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns, nnz


def expand7(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-5, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = []
    for x in xrange(N):
        for y in xrange(N):
            xys.append([x, y, d, qry, shape, tmp_path, csr, I, prune])

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem)
                                   for elem in xys)

    row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    rows_sum = None
    print 'row_sum_name', row_sum_ns
    for row_sum_n in row_sum_ns:
        try:
            tmp = np.load(row_sum_n)
            tmp = tmp.items()[0][1]
            tmp = np.asarray(tmp, 'float32')
            os.system('rm %s' % row_sum_n)

        except:
            continue

        try:
            rows_sum += tmp
        except:
            rows_sum = tmp

    fns = [elem[1] for elem in zns if type(elem[1]) != type(None)]
    nnz = max([elem[2] for elem in zns])

    # rename
    for i in fns:
        os.system('mv %s %s_old' % (i, i))
        os.system('mv %s_new.npz %s' % (i, i))

    return row_sum, fns, nnz

# parallelize row sum


def prsum(fns):
    print 'parallel row sum', fns
    row_sum = None
    for fn in fns:
        print 'parallel row sum fn', fn
        #tmp = np.load(fn)
        try:
            tmp = np.load(fn)
            tmp = tmp.items()[0][1]
            tmp = np.asarray(tmp, 'float32')
            os.system('rm %s' % fn)
            print 'del rowsum'
        except:
            continue

        try:
            row_sum += tmp
            #idx = row_sum < tmp
            #row_sum[idx] = tmp[idx]
            #row_sum = np.sum([row_sum, tmp], 0)
        except:
            row_sum = tmp

        del tmp
        gc.collect()

    return row_sum


def expand8(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = []
    for x in xrange(d):
        for y in xrange(d):
            xys.append([x, y, d, qry, shape, tmp_path, csr, I, prune])

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem)
                                   for elem in xys)

    row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        #xys = row_sum_ns[idx:idx+cpu*4]
        #rfn = row_sum_ns.pop()
        # xys[i%cpu].append(rfn)
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)

    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    nnz = max([elem[2] for elem in zns])

    #fns = [elem[1] for elem in zns if type(elem[1]) != type(None)]

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # print 'old_new', set([elem.replace('_old', '') for elem in old_fns]) - set(fns), set(fns) - set([elem.replace('_old', '') for elem in old_fns])
    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    return row_sum, fns, nnz
    # rename
    # for i in fns:
    #    os.system('mv %s %s_old' % (i, i))
    #    os.system('mv %s_new.npz %s' % (i, i))

    # return row_sum, fns, nnz


def expand9(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune]
            xys[flag % cpu].append(xy)
            flag += 1

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(element_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()
    #row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    #nnz = max([elem[2] for elem in zns])
    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # print 'old_new', set([elem.replace('_old', '') for elem in old_fns]) - set(fns), set(fns) - set([elem.replace('_old', '') for elem in old_fns])
    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    return row_sum, fns, nnz
    # rename
    # for i in fns:
    #    os.system('mv %s %s_old' % (i, i))
    #    os.system('mv %s_new.npz %s' % (i, i))

    # return row_sum, fns, nnz


def expand10(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune, cpu]
            xys[flag % cpu].append(xy)
            flag += 1

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(element_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()
    #row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    #nnz = max([elem[2] for elem in zns])
    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # print 'old_new', set([elem.replace('_old', '') for elem in old_fns]) - set(fns), set(fns) - set([elem.replace('_old', '') for elem in old_fns])
    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    return row_sum, fns, nnz


def expand(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1, fast=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N

    nnz = 0
    row_sum = None
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune, cpu, fast]
            xys[flag % cpu].append(xy)
            flag += 1

    # if cpu <= 1:
    #    print 'cpu < 1', cpu, len(xys)
    #    zns = map(element_wrapper, xys)
    # else:
    #    print 'cpu > 1', cpu, len(xys)
    #    zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem) for elem in xys)
    #    #pool = mp.Pool(cpu)
    #    #zns = pool.map(element_wrapper, xys)
    #    #pool.terminate()
    #    #pool.close()
    #    #del pool
    #    #gc.collect()
    if fast and cpu > 1 and len(xys) > 1:
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem)
                                   for elem in xys)
    else:
        zns = map(element_wrapper, xys)

    gc.collect()
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    # remove z_ms.npy zr_ms.npy
    os.system('rm %s/*_z_ms.npy %s/*_zc_ms.npy'%(tmp_path, tmp_path))

    return row_sum, fns, nnz


def regularize(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1 / 4e3, cpu=1, fast=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N

    nnz = 0
    row_sum = None
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune, cpu, fast]
            xys[flag % cpu].append(xy)
            flag += 1

    # if cpu <= 1:
    #    print 'cpu < 1', cpu, len(xys)
    #    zns = map(element_wrapper, xys)
    # else:
    #    print 'cpu > 1', cpu, len(xys)
    #    zns = Parallel(n_jobs=cpu)(delayed(element_wrapper)(elem) for elem in xys)
    #    #pool = mp.Pool(cpu)
    #    #zns = pool.map(element_wrapper, xys)
    #    #pool.terminate()
    #    #pool.close()
    #    #del pool
    #    #gc.collect()
    if fast and cpu > 1 and len(xys) > 1:
        zns = Parallel(n_jobs=cpu)(delayed(relement_wrapper)(elem)
                                   for elem in xys)
    else:
        zns = map(relement_wrapper, xys)

    gc.collect()
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)


    # remove z_ms.npy zr_ms.npy
    os.system('rm %s/*_z_ms.npy %s/*_zc_ms.npy'%(tmp_path, tmp_path))


    return row_sum, fns, nnz


def expand_gpu0(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune]
            xys[flag % cpu].append(xy)
            flag += 1

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper_gpu, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper_gpu)(elem)
                                   for elem in xys)

    gc.collect()
    #row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    #nnz = max([elem[2] for elem in zns])
    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # print 'old_new', set([elem.replace('_old', '') for elem in old_fns]) - set(fns), set(fns) - set([elem.replace('_old', '') for elem in old_fns])
    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    return row_sum, fns, nnz
    # rename
    # for i in fns:
    #    os.system('mv %s %s_old' % (i, i))
    #    os.system('mv %s_new.npz %s' % (i, i))

    # return row_sum, fns, nnz


def expand_gpu1(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune]
            xys[flag % cpu].append(xy)
            flag += 1

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper_gpu, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper_gpu)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(element_wrapper_gpu, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()
    #row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)
        #pool = mp.Pool(cpu)
        #row_sums = pool.map(prsum, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    #nnz = max([elem[2] for elem in zns])
    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # print 'old_new', set([elem.replace('_old', '') for elem in old_fns]) - set(fns), set(fns) - set([elem.replace('_old', '') for elem in old_fns])
    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    return row_sum, fns, nnz
    # rename
    # for i in fns:
    #    os.system('mv %s %s_old' % (i, i))
    #    os.system('mv %s_new.npz %s' % (i, i))

    # return row_sum, fns, nnz


# add multiple gpu support
def expand_gpu(qry, shape=(10**8, 10**8), tmp_path=None, csr=True, I=1.5, prune=1e-6, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    err = None
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    # print 'num set is', num_set

    nnz = 0
    row_sum = None
    xys = [[elem] for elem in xrange(cpu)]
    flag = 0
    for x in xrange(d):
        for y in xrange(d):
            xy = [x, y, d, qry, shape, tmp_path, csr, I, prune]
            xys[flag % cpu].append(xy)
            flag += 1

    #zns = map(element_wrapper, xys)
    if cpu <= 1 or len(xys) <= 1:
        print 'cpu < 1', cpu, len(xys)
        zns = map(element_wrapper_gpu, xys)
    else:
        print 'cpu > 1', cpu, len(xys)
        zns = Parallel(n_jobs=cpu)(delayed(element_wrapper_gpu)(elem)
                                   for elem in xys)
        #pool = mp.Pool(cpu)
        #zns = pool.map(element_wrapper_gpu, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        gc.collect()

    gc.collect()
    #row_sum_ns = [elem[0] for elem in zns if type(elem[0]) != type(None)]
    row_sum_ns = []
    for elem_zns in zns:
        for elem in elem_zns:
            if type(elem[0]) != type(None):
                row_sum_ns.append(elem[0])

    print 'row_sum_name', row_sum_ns
    xys = [[] for elem in xrange(cpu)]
    Nrs = len(row_sum_ns)
    for i in xrange(Nrs):
        xys[i % cpu].append(row_sum_ns[i])
    if cpu <= 1 or len(xys) <= 1:
        print 'row sum cpu < 1', cpu, len(xys)
        row_sums = map(prsum, xys)
    else:
        print 'row sum cpu > 1', cpu, len(xys)
        row_sums = Parallel(n_jobs=cpu)(delayed(prsum)(elem) for elem in xys)
        #pool = mp.Pool(cpu)
        #row_sums = pool.map(prsum, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()
    rows_sum = sum([elem for elem in row_sums if type(elem) != type(None)])

    #nnz = max([elem[2] for elem in zns])
    nnz = 0
    for elem_zns in zns:
        for elem in elem_zns:
            nnz = max(nnz, elem[2])

    # remove old file
    old_fns = [tmp_path + os.sep +
               elem for elem in os.listdir(tmp_path) if elem.endswith('_old')]
    for i in old_fns:
        os.system('rm %s' % i)

    # print 'old_new', set([elem.replace('_old', '') for elem in old_fns]) - set(fns), set(fns) - set([elem.replace('_old', '') for elem in old_fns])
    # rename
    fns = []
    for x in xrange(d):
        for y in xrange(d):
            fn = tmp_path + os.sep + str(x) + '_' + str(y) + '.npz'
            if os.path.isfile(fn):
                os.system('mv %s %s_old' % (fn, fn))
            if os.path.isfile(fn + '_new.npz'):
                os.system('mv %s_new.npz %s' % (fn, fn))
                fns.append(fn)

    return row_sum, fns, nnz
    # rename
    # for i in fns:
    #    os.system('mv %s %s_old' % (i, i))
    #    os.system('mv %s_new.npz %s' % (i, i))

    # return row_sum, fns, nnz


# normalizatin
def norm0(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = os.listdir(tmp_path)
    if not xy:
        xy = [elem.split('.npz')[0].split('_')
              for elem in fns if elem.endswith('.npz')]
        # xy = list(set(map(int, xy)))
        # print xy
        xy = sum(xy, [])
        xy = list(set(xy))
        xy.sort(key=lambda x: int(x))
    else:
        xy = map(str, xy)

    if row_sum == None:
        row_sum = np.zeros(shape[0], dtype='float32')

        for i in xy:
            for j in xy:
                xn = tmp_path + '/' + i + '_' + j + '.npz'
                try:
                    x = load_matrix(qry, load=True)
                    row_sum += x.sum(0)
                except:
                    continue

    for i in xy:
        for j in xy:
            xn = tmp_path + '/' + i + '_' + j + '.npz'
            try:
                x = load_matrix(qry, load=True)
                x.eliminate_zeros()
                x.data /= row_sum.take(x.indices, mode='clip')
                sparse.save_npz(xn, x)

            except:
                continue


def norm1(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue

    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
            x.data /= row_sum.take(x.indices, mode='clip')
            sparse.save_npz(i + '_new', x)
        except:
            continue

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    return fns


def norm2(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue

    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
            x.data /= row_sum.take(x.indices, mode='clip')
            sparse.save_npz(i + '_new', x)
        except:
            continue

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    return fns


def norm3(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue

    err = None
    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
        except:
            continue
        try:
            x.data /= row_sum.take(x.indices, mode='clip')
        except:
            # print 'start norm3', check, x.data, row_sum.max()
            break

        if check:
            # print 'start norm4'
            x_old = load_matrix(i + '_old', shape=shape, csr=csr)
            # print 'start norm4 x x_old', abs(x - x_old).shape

            gap = abs(x - x_old) - abs(rtol * x_old)
            err = max(err, gap.max())
            # print 'check err is', err, i, i+'_old'

        sparse.save_npz(i + '_new', x)

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    if err != None and err <= atol:
        cvg = True
    else:
        cvg = False

    return fns, cvg


def norm4(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    nnz = 0
    fns = [tmp_path + '/' +
           elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue
    print 'norm nnz is', nnz, i, fns

    err = None
    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
        except:
            continue
        try:
            x.data /= row_sum.take(x.indices, mode='clip')
        except:
            # print 'start norm3', check, x.data, row_sum.max()
            break

        if check:
            # print 'start norm4'
            x_old = load_matrix(i + '_old', shape=shape, csr=csr)
            # print 'start norm4 x x_old', abs(x - x_old).shape

            gap = abs(x - x_old) - abs(rtol * x_old)
            err = max(err, gap.max())
            # print 'check err is', err, i, i+'_old'

        sparse.save_npz(i + '_new', x)

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    if err != None and err <= atol:
        cvg = True
    else:
        cvg = False

    return fns, cvg, nnz


def norm5(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue
    print 'norm nnz is', nnz, i, fns

    # normalize
    for i in fns:
        try:
            x = load_matrix(i, shape=shape, csr=csr)
            x.data /= row_sum.take(x.indices, mode='clip')
        except:
            # print 'start norm3', check, x.data, row_sum.max()
            continue

        sparse.save_npz(i + '_new', x)

    for i in fns:
        os.system('mv %s_new.npz %s' % (i, i))

    err = None
    if check:
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
            except:
                x = None

            # print 'start norm4'
            try:
                x_old = load_matrix(i + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                continue

            print 'check err is', err, type(x), type(x_old)

    if err != None and err <= atol:
        cvg = True
    else:
        cvg = False

    return fns, cvg, nnz

# sub function of norm


def sdiv0(parameters):
    fn, shape, csr, check, rtol, tmp_path = parameters
    row_sum = np.asarray(
        np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
                #os.system('rm %s_old'%fn)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0
    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


def sdiv1(parameters):
    fn, shape, csr, check, rtol, tmp_path = parameters
    row_sum = np.asarray(
        np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    row_sum._mmap.close()
    del row_sum
    gc.collect()

    if check and err != None:
        return err
    else:
        return float('+inf')


def sdiv2(parameters, row_sum=None):
    fn, shape, csr, check, rtol, tmp_path = parameters
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


# add 16 bit float support
def sdiv3(parameters, row_sum=None, dtype='float32'):
    fn, shape, csr, check, rtol, tmp_path = parameters
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


# prune element < threshold
def sdiv4(parameters, row_sum=None, dtype='float32'):
    fn, shape, csr, check, rtol, tmp_path, prune = parameters
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        print 'norm before nnz', x.nnz, fn
        x.data[x.data < prune] = 0
        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')

# remove pruning operation


def sdiv5(parameters, row_sum=None, dtype='float32'):
    fn, shape, csr, check, rtol, tmp_path, prune = parameters
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        print 'norm before nnz', x.nnz, fn
        #x.data[x.data < prune] = 0
        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


def sdiv6(parameters, row_sum=None, dtype='float32', order='c'):
    fn, shape, csr, check, rtol, tmp_path, prune = parameters
    P = int(1. / prune) + 1
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        x.data /= row_sum.take(x.indices, mode='clip')
        print 'max_x_data_fk', x.data.max(), x.sum(0).max(), x.sum(1).max(), row_sum.max(), row_sum.min()
        #xt = load_matrix(fn, shape=shape, csr=csr).T
        #xt.data /= row_sum.take(xt.indices, mode='clip')
        #x = xt.T.tocsr()
        #del xt
        # gc.collect()

        # reduce the size of matrix
        # if order == 'c':
        #    xt = x.T.tocsr()
        # else:
        #    xt = x

        #a, b, c = xt.indices, xt.indptr, xt.data
        #select_jit(a, b, c, S=P)
        # print 'sdiv_fk_S', prune, P
        #select_jit(xt.indices, xt.indptr, xt.data, S=P)

        # if order == 'c':
        #    x = xt.T.tocsr()
        # else:
        #    x = xt

        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        print 'norm before nnz', x.nnz, fn
        #x.data[x.data < prune] = 0
        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


# correct sdiv
def sdiv(parameters, row_sum=None, dtype='float32', order='c'):
    fn, shape, csr, check, rtol, tmp_path, prune, diag = parameters
    P = int(1. / prune) + 1
    if type(row_sum) == type(None):
        #row_sum = np.asarray(
        #    np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

        fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32')
        row_sum = np.asarray(fp)

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        if diag == True:
            print 'yes_set_diag'
            R, C = x.nonzero()
            st = min(R.min(), C.min())
            ed = max(R.max(), C.max())
            idx = np.arange(st, ed + 1)
            dat = np.ones(idx.size)
            dia = sparse.csr_matrix((dat, (idx, idx)), shape=x.shape)
            x += dia

        x.data /= row_sum.take(x.indices, mode='clip')
        x.data = np.nan_to_num(x.data)
        #x.data[x.data < prune] = 0

        try:
            fp._mmap.close()
        except:
            pass
        print 'max_x_data_fk', x.data.max(), x.sum(0).max(), x.sum(1).max(), row_sum.max(), row_sum.min()
        #xt = load_matrix(fn, shape=shape, csr=csr).T
        #xt.data /= row_sum.take(xt.indices, mode='clip')
        #x = xt.T.tocsr()
        #del xt
        # gc.collect()
        # if diag == True:
        #    print 'yes_set_diag'
        #    R, C = x.nonzero()
        #    st = min(R.min(), C.min())
        #    ed = max(R.max(), C.max())
        #    idx = np.arange(st, ed+1)
        #    dat = np.ones(idx.size)
        #    dia = sparse.csr_matrix((dat, (idx, idx)), shape=x.shape)
        #    x += dia

        # reduce the size of matrix
        # if order == 'c':
        #    xt = x.T.tocsr()
        # else:
        #    xt = x

        #a, b, c = xt.indices, xt.indptr, xt.data
        #select_jit(a, b, c, S=P)
        # print 'sdiv_fk_S', prune, P
        #select_jit(xt.indices, xt.indptr, xt.data, S=P)

        # if order == 'c':
        #    x = xt.T.tocsr()
        # else:
        #    x = xt

        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)

        print 'norm before nnz', x.nnz, fn

        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


def rsdiv(parameters, row_sum=None, dtype='float32', order='c'):
    fn, shape, csr, check, rtol, tmp_path, prune, rgl = parameters
    P = int(1. / prune) + 1
    if type(row_sum) == type(None):
        #row_sum = np.asarray(
        #    np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

        fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32')
        row_sum = np.asarray(fp)

    err = None
    try:
        x = load_matrix(fn, shape=shape, csr=csr)
        # if check and rgl:
        #    x.data /= row_sum.take(x.indices, mode='clip')

        x.data /= row_sum.take(x.indices, mode='clip')

        try:
            fp._mmap.close()
        except:
            pass

        #xt = load_matrix(fn, shape=shape, csr=csr).T
        #xt.data /= row_sum.take(xt.indices, mode='clip')
        #x = xt.T
        #del xt
        # gc.collect()

        # reduce the size of matrix
        # if order == 'c':
        #    xt = x.T.tocsr()
        # else:
        #    xt = x

        #a, b, c = xt.indices, xt.indptr, xt.data
        #select_jit(a, b, c, S=P)
        # if order == 'c':
        #    x = xt.T.tocsr()
        # else:
        #    x = xt

        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        print 'norm before nnz', x.nnz, fn
        #x.data[x.data < prune] = 0
        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('cp %s_new.npz %s' % (fn, fn))
        if rgl:
            os.system('mv %s_new.npz %s_Mg.npz' % (fn, fn))

        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


# sdiv for batch input
def sdiv_wrapper0(elem):
    out = []
    for parameters in elem:
        tmp = sdiv(parameters)
        out.append(tmp)

    return out


def sdiv_wrapper(elem):
    if len(elem) > 0:
        tmp_path = elem[0][5]
    else:
        return []
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32')
    row_sum = np.asarray(fp, 'float32')
    out = []
    for parameters in elem:
        tmp = sdiv(parameters, row_sum)
        out.append(tmp)

    fp._mmap.close()
    del fp
    del row_sum
    gc.collect()

    return out


def rsdiv_wrapper(elem):
    if len(elem) > 0:
        tmp_path = elem[0][5]
    else:
        return []
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32')
    row_sum = np.asarray(fp, 'float32')
    out = []
    for parameters in elem:
        tmp = rsdiv(parameters, row_sum)
        out.append(tmp)

    fp._mmap.close()
    del fp
    del row_sum
    gc.collect()

    return out


def sdiv_gpu0(parameters, row_sum=None, dtype='float32'):
    fn, shape, csr, check, rtol, tmp_path, prune = parameters
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    block = shape[0]
    err = None
    try:
        print 'sdiv_gpu', fn, 'csr', csr
        x = load_matrix(fn, shape=shape, csr=csr)

        j = int(fn.split('_')[-1].split('.npz')[0])
        start = j * block
        end = start + block

        rs_part = row_sum[start: end]

        print 'sdiv_gpu load x', x.shape, 'row sum', rs_part.shape

        x.data /= rs_part.take(x.indices, mode='clip')

        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        print 'norm before nnz', x.nnz, fn
        x.data[x.data < prune] = 0
        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        #os.system('cp %s_new.npz %s' % (fn, fn))

        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


# remove prune operon in this function
def sdiv_gpu(parameters, row_sum=None, dtype='float32'):
    fn, shape, csr, check, rtol, tmp_path, prune = parameters
    if type(row_sum) == type(None):
        row_sum = np.asarray(
            np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32'))

    block = shape[0]
    err = None
    try:
        print 'sdiv_gpu', fn, 'csr', csr
        x = load_matrix(fn, shape=shape, csr=csr)

        j = int(fn.split('_')[-1].split('.npz')[0])
        start = j * block
        end = start + block

        rs_part = row_sum[start: end]

        print 'sdiv_gpu load x', x.shape, 'row sum', rs_part.shape

        x.data /= rs_part.take(x.indices, mode='clip')
        #xt = x.T
        #xt.data /= rs_part.take(x.indices, mode='clip')
        #x = xt.T

        # convert entries to 16 bit float
        #x.data = np.asarray(x.data, dtype=dtype)
        print 'norm before nnz', x.nnz, fn
        #x.data[x.data < prune] = 0
        x.eliminate_zeros()
        print 'norm after nnz', x.nnz, fn

        sparse.save_npz(fn + '_new', x)
        os.system('mv %s_new.npz %s' % (fn, fn))
        #os.system('cp %s_new.npz %s' % (fn, fn))

        if check:
            try:
                x_old = load_matrix(fn + '_old', shape=shape, csr=csr)
            except:
                x_old = None
            # print 'start norm4 x x_old', abs(x - x_old).shape

            if type(x) != type(None) and type(x_old) != type(None):
                gap = abs(x - x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            elif type(x) != type(None) and type(x_old) == type(None):
                gap = abs(x)
                err = max(err, gap.max())
            elif type(x) == type(None) and type(x_old) != type(None):
                gap = abs(x_old) - abs(rtol * x_old)
                err = max(err, gap.max())
            else:
                err = 0

            del x_old
            gc.collect()

        del x
        gc.collect()

    except:
        pass

    if check and err != None:
        return err
    else:
        return float('+inf')


def sdiv_wrapper_gpu(elem):
    if len(elem) > 0:
        tmp_path = elem[0][5]
    else:
        return []
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='r', dtype='float32')
    row_sum = np.asarray(fp, 'float32')
    out = []
    for parameters in elem:
        tmp = sdiv_gpu(parameters, row_sum)
        out.append(tmp)

    fp._mmap.close()
    del fp
    del row_sum
    gc.collect()

    return out


# parallel norm step
def norm6(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                x = np.asarray(x.sum(0))[0]
                row_sum += x
            except:
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    # normalize
    xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    if cpu <= 1 or len(xys) <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv)(elem) for elem in xys)

    if check:
        err = max(errs)
        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


def norm7(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                row_sum += y
                del x
                del y
                gc.collect()
            except:
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1 or len(xys) <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv_wrapper, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)

        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


# remove element < prune
def norm8(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1, prune=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if prune == None:
        prune = .05 / shape[0]

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                row_sum += y
                del x
                del y
                gc.collect()
            except:
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path, prune]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv_wrapper, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)

        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


def norm9(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1, prune=None, diag=True):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if prune == None:
        #prune = .05 / shape[0]
        prune = 1 / 4e3

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                #y = np.asarray(x.sum(1).T)[0]
                row_sum += y
                del x
                del y
                gc.collect()
            except:
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path, prune]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv_wrapper, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)

        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


# correct row sum
def norm(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1, prune=None, diag=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if prune == None:
        #prune = .05 / shape[0]
        prune = 1 / 4e3

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                row_sum += y

                #y = np.asarray(x.max(0))[0]
                #idx = row_sum < y
                #row_sum[idx] = y[idx]

                del x
                del y
                gc.collect()
            except:
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path, prune, diag]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv_wrapper, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)

        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


# regularized norm
def rnorm(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1, prune=None, rgl=True):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if prune == None:
        #prune = .05 / shape[0]
        prune = 1 / 4e3

    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]

    if isinstance(row_sum, type(None)):
        row_sum = np.zeros(shape[0], dtype='float32')
        for i in fns:
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                row_sum += y

                #y = np.asarray(x.max(0))[0]
                #idx = row_sum < y
                #row_sum[idx] = y[idx]

                del x
                del y
                gc.collect()
            except:
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path, prune, rgl]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(rsdiv_wrapper, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        errs = Parallel(n_jobs=cpu)(delayed(rsdiv_wrapper)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)
        print 'current_max_err', err
        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


# normal function for gpu
def norm_gpu0(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1, prune=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if prune == None:
        prune = 1 / 4e3

    block = shape[0]
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    if isinstance(row_sum, type(None)):
        #row_sum = np.zeros(shape[0], dtype='float32')
        row_sum = np.zeros(block * N, dtype='float32')
        for i in fns:
            j = int(i.split('_')[-1].split('.npz')[0])
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                start = j * block
                end = start + block
                row_sum[start: end] += y
                del x
                del y
                gc.collect()
                print 'get rowsum'
            except:
                print 'can\'t get rowsum'
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path, prune]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1 or len(xys) <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv_wrapper_gpu, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        #errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper)(elem) for elem in xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper_gpu)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper_gpu, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)

        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


# correct row sum
def norm_gpu(qry, shape=(10**8, 10**8), tmp_path=None, row_sum=None, csr=False, rtol=1e-5, atol=1e-8, check=False, cpu=1, prune=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if prune == None:
        prune = .05 / shape[0]

    block = shape[0]
    Ns = [elem.split('.')[0].split('_')
          for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    N = max([max(map(int, elem)) for elem in Ns]) + 1
    d = N
    fns = []
    for i in xrange(d):
        for j in xrange(d):
            fn = tmp_path + '/' + str(i) + '_' + str(j) + '.npz'
            fns.append(fn)

    nnz = 0
    #fns = [tmp_path + '/' + elem for elem in os.listdir(tmp_path) if elem.endswith('.npz')]
    if isinstance(row_sum, type(None)):
        #row_sum = np.zeros(shape[0], dtype='float32')
        row_sum = np.zeros(block * N, dtype='float32')
        for i in fns:
            j = int(i.split('_')[-1].split('.npz')[0])
            try:
                x = load_matrix(i, shape=shape, csr=csr)
                nnz = max(nnz, x.nnz)
                y = np.asarray(x.sum(0))[0]
                #y = np.asarray(x.sum(1).T)[0]

                start = j * block
                end = start + block
                row_sum[start: end] += y
                del x
                del y
                gc.collect()
                print 'get rowsum'
            except:
                print 'can\'t get rowsum'
                continue
    # print 'norm nnz is', nnz, i, fns
    # write row sum to disk
    fp = np.memmap(tmp_path + '/row_sum_total.npy', mode='w+',
                   dtype='float32', shape=row_sum.shape)
    fp[:] = row_sum
    fp.flush()
    fp._mmap.close()
    # normalize
    #xys = [[elem, shape, csr, check, rtol, tmp_path] for elem in fns]
    xys = [[] for elem in xrange(cpu)]
    flag = 0
    for elem in fns:
        #elem = fns[i]
        xy = [elem, shape, csr, check, rtol, tmp_path, prune]
        xys[flag % cpu].append(xy)
        flag += 1

    if cpu <= 1 or len(xys) <= 1:
        print 'norm cpu < 1', cpu, len(xys)
        errs = map(sdiv_wrapper_gpu, xys)
    else:
        print 'norm cpu > 1', cpu, len(xys)
        #errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper)(elem) for elem in xys)
        errs = Parallel(n_jobs=cpu)(delayed(sdiv_wrapper_gpu)(elem)
                                    for elem in xys)
        #pool = mp.Pool(cpu)
        #errs = pool.map(sdiv_wrapper_gpu, xys)
        # pool.terminate()
        # pool.close()
        #del pool
        # gc.collect()

    gc.collect()

    if check:
        #err = max(errs)
        err = 0
        for i in errs:
            for j in i:
                err = max(err, j)

        cvg = err < atol and True or False
    else:
        cvg = False

    return fns, cvg, nnz


# mcl algorithm
def mcl0(qry, tmp_path=None, xy=[], I=1.5, prune=1e-5, itr=100, rtol=1e-5, atol=1e-8, check=5):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    q2n = mat_split(qry)
    N = len(q2n)
    shape = (N, N)
    # norm
    print 'finish norm'
    # expension
    for i in xrange(itr):
        print 'iteration', i

        if i <= 0:
            print '1st row sum'
            fns = norm(qry, shape, tmp_path, csr=False)
        else:
            fns = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True)

        if i > 0 and i % check == 0:
            row_sum, fns, cvg = expend(qry, shape, tmp_path, True, check=True)
        else:
            row_sum, fns, cvg = expend(qry, shape, tmp_path, True)

        if cvg:
            print 'yes, convergency'
            break


# merge two connected components array
@jit
def merge_connected(c0, c1):
    n0, a0 = c0
    n1, a1 = c1
    l0 = len(a0)
    l1 = len(a1)
    assert l0 == l1

    # sort by a1
    ht = np.zeros(n1, dtype='int')
    for i in a1:
        ht[i] += 1

    for i in xrange(1, n1):
        ht[i] += ht[i - 1]

    htc = ht.copy()
    s1 = np.empty(l1, dtype='int')
    for i in xrange(l1):
        x = a1[i]
        y = ht[x] - 1
        s1[y] = i
        ht[x] = y

    ht = htc
    # relabel a0
    visit = -np.ones(n0, dtype='int')
    a0_n = np.empty(l0, dtype='int')
    flag = 0
    total = 0
    for i in xrange(n1):
        if i <= 0:
            st, ed = 0, ht[i]
        else:
            st, ed = ht[i - 1], ht[i]

        total += ed - st
        # check current components has been visited
        c = -1
        for j in xrange(st, ed):
            cj = a0[s1[j]]

            if visit[cj] > -1:
                c = visit[cj]
                idx = s1[st:ed]
                a0_n[idx] = c
                visit[a0[idx]] = c
                break

        if c == -1:
            idx = s1[st:ed]
            a0_n[idx] = flag
            visit[a0[idx]] = flag
            flag += 1

        else:
            continue

    return flag, a0_n


def mcl1(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n = mat_split(qry, chunk=chunk, cpu=cpu)
    N = len(q2n)
    prune = min(prune, 1e2 / N)
    shape = (N, N)
    # norm
    fns, cvg = norm(qry, shape, tmp_path, csr=False)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)

        if i > 0 and i % check == 0:
            fns, cvg = norm(qry, shape, tmp_path,
                            row_sum=row_sum, csr=True, check=True)
        else:
            fns, cvg = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True)

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)
        del g
        gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    for v in groups.itervalues():
        print '\t'.join(v)


def mcl2(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n = mat_split(qry, chunk=chunk, cpu=cpu)
    N = len(q2n)
    prune = min(prune, 1e2 / N)
    shape = (N, N)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        row_sum, fns, nnz = expend(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % check == 0:
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True)
        else:
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True)

        if nnz < chunk / 2:
            print 'we try to merge 4 block into one', chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl3(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    prune = min(prune, 1e2 / N)
    shape = (N, N)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % check == 0:
            #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
            q2n, fns = mat_reorder(qry, q2n, shape=shape,
                                   chunk=chunk, csr=True)

        row_sum, fns, nnz = expend(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True)
        else:
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True)

        if nnz < chunk / 2:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl4(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    prune = min(prune, 1e2 / N)
    shape = (N, N)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % (check * 2) == 0:
            #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
            q2n, fns = mat_reorder(qry, q2n, shape=shape,
                                   chunk=chunk, csr=True)

        row_sum, fns, nnz = expend(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True)

        else:
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True)

        if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl5(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    prune = min(prune, 1e2 / N)
    shape = (N, N)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        row_sum, fns, nnz = expend(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True)
            q2n, fns = mat_reorder(qry, q2n, shape=shape,
                                   chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True)

        if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl6(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False, cpu=cpu)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        row_sum, fns, nnz = expand(qry, shape, tmp_path, True, I, prune, cpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, cpu=cpu)

        if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# add pruning function
def mcl7(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)

    N = len(q2n)

    # save q2n to disk
    print 'saving q2n to disk'
    _o = open(tmp_path + '_dict.pkl', 'wb')
    cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
    _o.close()

    del q2n
    gc.collect()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False, cpu=cpu)
    pruning(qry, tmp_path, prune=prune, cpu=cpu)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        if i == 0:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu)

        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, cpu=cpu)

        pruning(qry, tmp_path, cpu=cpu)
        # if nnz < chunk / 4 and len(fns) > cpu * cpu:
        if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# add resume parameter
def mcl8(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, rsm=False):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    if rsm == False:
        q2n, block = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)

        N = len(q2n)

        # save q2n to disk
        print 'saving q2n to disk'
        _o = open(tmp_path + '_dict.pkl', 'wb')
        cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
        _o.close()

        del q2n
        gc.collect()
    else:
        f = open(tmp_path + '_dict.pkl', 'rb')
        q2n = cPickle.load(f)
        N = len(q2n)
        os.system('rm %s/*new* %s/*old' % (tmp_path, tmp_path))
        f.close()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False, cpu=cpu)
    pruning(qry, tmp_path, prune=prune, cpu=cpu)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        if i == 0:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu)

        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, cpu=cpu)

        pruning(qry, tmp_path, cpu=cpu)
        # if nnz < chunk / 4 and len(fns) > cpu * cpu:
        if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# add memory usage limit
def mcl9(qry, tmp_path=None, xy=[], I=1.5, prune=1 / 4e3, select=1100, recover=1400, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, rsm=False, mem=4):

    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if rsm == False:
        os.system('mkdir -p %s' % tmp_path)
        os.system('rm -rf %s/*' % tmp_path)

        q2n, block = mat_split(qry, tmp_path=tmp_path,
                               chunk=chunk, cpu=cpu, sym=sym, mem=mem)

        N = len(q2n)

        # save q2n to disk
        print 'saving q2n to disk'
        _o = open(tmp_path + '_dict.pkl', 'wb')
        cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
        _o.close()

        del q2n
        gc.collect()
    else:
        f = open(tmp_path + '_dict.pkl', 'rb')
        q2n = cPickle.load(f)
        N = len(q2n)
        #os.system('rm %s/*new* %s/*old'%(tmp_path, tmp_path))
        for tmp in os.listdir(tmp_path):
            if tmp.endswith('_old'):
                a_tmp = tmp_path + '/' + tmp
                b_tmp = tmp_path + '/' + tmp.split('_old')[0]
                os.system('mv %s %s' % (a_tmp, b_tmp))

        os.system('rm %s/*new*' % tmp_path)

        f.close()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False,
                         cpu=cpu, prune=prune, diag=False)

    #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
    pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)

    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        if i == 0:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu)

        if i > check and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, prune=prune)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu, prune=prune)

        #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
        pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)

        if nnz < chunk / 4 and len(fns) > cpu ** 2:
            # if nnz < chunk / 4 or nnz <= N:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    '''
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()
    '''

    g = load_matrix(fns[0], shape, True)
    #cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g += load_matrix(fn, shape, True)
        #ci = csgraph.connected_components(g)
        #cs = merge_connected(cs, ci)

    cs = csgraph.connected_components(g)
    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl0(qry, tmp_path=None, xy=[], I=1.5, prune=1/4e3, select=1100, recover=1400, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, rsm=False, mem=4):

    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if rsm == False:
        os.system('mkdir -p %s' % tmp_path)
        os.system('rm -rf %s/*' % tmp_path)

        q2n, block = mat_split(qry, tmp_path=tmp_path,
                               chunk=chunk, cpu=cpu, sym=sym, mem=mem)

        N = len(q2n)

        # save q2n to disk
        print 'saving q2n to disk'
        _o = open(tmp_path + '_dict.pkl', 'wb')
        cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
        _o.close()

        del q2n
        gc.collect()
    else:
        f = open(tmp_path + '_dict.pkl', 'rb')
        q2n = cPickle.load(f)
        N = len(q2n)
        #os.system('rm %s/*new* %s/*old'%(tmp_path, tmp_path))
        for tmp in os.listdir(tmp_path):
            if tmp.endswith('_old'):
                a_tmp = tmp_path + '/' + tmp
                b_tmp = tmp_path + '/' + tmp.split('_old')[0]
                os.system('mv %s %s' % (a_tmp, b_tmp))

        os.system('rm %s/*new*' % tmp_path)

        f.close()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False,
                         cpu=cpu, prune=prune, diag=False)
    #raise SystemExit()

    #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
    #chaos = pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)
    chaos = 0

    # print 'finish norm', cvg
    changed = 0
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        #if i == 0:
        if 1:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu)

        # if i > check and i % check == 0:
        #    print 'reorder the matrix'
        #    fns, cvg, nnz = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, prune=prune)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)
        # else:
        #    #os.system('rm %s/*.npz_old'%tmp_path)
        #    fns, cvg, nnz = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu, prune=prune)

        fns, cvg, nnz = norm(qry, shape, tmp_path,
                             row_sum=row_sum, csr=True, cpu=cpu, prune=prune)

        #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
        chao_old = chaos
        chaos = pruning(qry, tmp_path, prune=prune,
                        S=select, R=recover, cpu=cpu, fast=True)
        changed = abs(chaos - chao_old) < 1e-9 and changed + 1 or 0
        print 'current_chaos', i, chaos, chao_old

        #if chaos < 1e-3 or changed >= 5:
        if chaos < 1e-3:
            break

        if nnz < chunk / 4 and len(fns) > cpu ** 2:
            # if nnz < chunk / 4 or nnz <= N:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    '''
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()
    '''

    g = load_matrix(fns[0], shape, True)
    #cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g += load_matrix(fn, shape, True)
        #ci = csgraph.connected_components(g)
        #cs = merge_connected(cs, ci)

    cs = csgraph.connected_components(g)
    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def get_connect(fns):
    g = None
    #g = load_matrix(fns[0], shape, True)
    #cs = csgraph.connected_components(g)
    cs = None
    for fn in fns:
        print 'fn', fn
        try:
            g0 = load_matrix(fn, csr=True)
            print 'g0', g0.nnz
        except:
            g0 = None
            continue

        try:
            g += g0
        except:
            g = g0

        if g.nnz > 1e8:
            ci = csgraph.connected_components(g)
            try:
                cs = merge_connected(cs, ci)
            except:
                cs = ci
            g = None

    if type(g) != type(None):
        ci = csgraph.connected_components(g)
        try:
            cs = merge_connected(cs, ci)
        except:
            cs = ci

    return cs



def mcl(qry, tmp_path=None, xy=[], I=1.5, prune=1/4e3, select=1100, recover=1400, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, rsm=False, mem=4):

    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if rsm == False:
        os.system('mkdir -p %s' % tmp_path)
        os.system('rm -rf %s/*' % tmp_path)

        q2n, block = mat_split(qry, tmp_path=tmp_path,
                               chunk=chunk, cpu=cpu, sym=sym, mem=mem)

        N = len(q2n)

        # save q2n to disk
        print 'saving q2n to disk'
        _o = open(tmp_path + '_dict.pkl', 'wb')
        cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
        _o.close()

        del q2n
        gc.collect()
    else:
        f = open(tmp_path + '_dict.pkl', 'rb')
        q2n = cPickle.load(f)
        N = len(q2n)
        #os.system('rm %s/*new* %s/*old'%(tmp_path, tmp_path))
        for tmp in os.listdir(tmp_path):
            if tmp.endswith('_old'):
                a_tmp = tmp_path + '/' + tmp
                b_tmp = tmp_path + '/' + tmp.split('_old')[0]
                os.system('mv %s %s' % (a_tmp, b_tmp))

        os.system('rm %s/*new*' % tmp_path)

        f.close()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False,
                         cpu=cpu, prune=prune, diag=False)
    #raise SystemExit()

    #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
    #chaos = pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)
    chaos = 0

    # print 'finish norm', cvg
    changed = 0
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        #row_sum, fns, nnz = expand(qry, shape, tmp_path, True, I, prune, cpu, fast=False)
        if i == 0:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu)



        # if i > check and i % check == 0:
        #    print 'reorder the matrix'
        #    fns, cvg, nnz = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, prune=prune)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)
        # else:
        #    #os.system('rm %s/*.npz_old'%tmp_path)
        #    fns, cvg, nnz = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu, prune=prune)

        fns, cvg, nnz = norm(qry, shape, tmp_path,
                             row_sum=row_sum, csr=True, cpu=cpu, prune=prune)

        #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
        chao_old = chaos
        chaos = pruning(qry, tmp_path, prune=prune,
                        S=select, R=recover, cpu=cpu, fast=True)
        changed = abs(chaos - chao_old) < 1e-9 and changed + 1 or 0
        print 'current_chaos', i, chaos, chao_old

        #if chaos < 1e-3 or changed >= 5:
        if chaos < 1e-3:
            break

        if nnz < chunk / 4 and len(fns) > cpu ** 2:
            # if nnz < chunk / 4 or nnz <= N:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    '''
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()
    '''

    g = load_matrix(fns[0], shape, True)
    #cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g += load_matrix(fn, shape, True)
        #ci = csgraph.connected_components(g)
        #cs = merge_connected(cs, ci)

    cs = csgraph.connected_components(g)
    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# regularized MCL
def rmcl0(qry, tmp_path=None, xy=[], I=1.5, prune=1 / 4e3, select=1100, recover=1400, itr=65, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, rsm=False, mem=4, rgl=True):

    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if rsm == False:
        os.system('mkdir -p %s' % tmp_path)
        os.system('rm -rf %s/*' % tmp_path)

        q2n, block = mat_split(qry, tmp_path=tmp_path,
                               chunk=chunk, cpu=cpu, sym=sym, mem=mem)

        N = len(q2n)

        # save q2n to disk
        print 'saving q2n to disk'
        _o = open(tmp_path + '_dict.pkl', 'wb')
        cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
        _o.close()

        del q2n
        gc.collect()
    else:
        f = open(tmp_path + '_dict.pkl', 'rb')
        q2n = cPickle.load(f)
        N = len(q2n)
        #os.system('rm %s/*new* %s/*old'%(tmp_path, tmp_path))
        for tmp in os.listdir(tmp_path):
            if tmp.endswith('_old'):
                a_tmp = tmp_path + '/' + tmp
                b_tmp = tmp_path + '/' + tmp.split('_old')[0]
                os.system('mv %s %s' % (a_tmp, b_tmp))

        os.system('rm %s/*new*' % tmp_path)
        f.close()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    #fns, cvg, nnz = rnorm(qry, shape, tmp_path, csr=False, cpu=cpu, check=True, rgl=True, prune=prune)
    #fns, cvg, nnz = rnorm(qry, shape, tmp_path, csr=False, cpu=cpu, check=False, rgl=False, prune=prune)
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False,
                         cpu=cpu, prune=prune, diag=False)

    #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
    pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)

    # get the Mg
    for i in os.listdir(tmp_path):
        if i.endswith('.npz') and 'new' not in i:
            j = tmp_path + '/' + i
            os.system('cp %s %s_Mg.npz' % (j, j))

    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        if i == 0:
            row_sum, fns, nnz = regularize(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = regularize(
                qry, shape, tmp_path, True, I, prune, cpu)

        if i > check and i % check == 0:
            print 'reorder the matrix'

            fns, cvg, nnz = norm(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, prune=prune)
            #fns, cvg, nnz = rnorm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, rgl=False, prune=prune)

            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)

            fns, cvg, nnz = norm(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu, prune=prune)
            #fns, cvg, nnz = rnorm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=False, cpu=cpu, rgl=False, prune=prune)

        #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
        pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)

        if nnz < chunk / 4 and len(fns) > cpu ** 2:
            # if nnz < chunk / 4 or nnz <= N:
            # if 0:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = rmerge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    '''
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()
    '''

    g = load_matrix(fns[0], shape, True)
    #cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        if fn.endswith('_Mg.npz'):
            continue
        g += load_matrix(fn, shape, True)
        #ci = csgraph.connected_components(g)
        #cs = merge_connected(cs, ci)

    cs = csgraph.connected_components(g)
    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# regularized mcl
def rmcl(qry, tmp_path=None, xy=[], I=1.5, prune=1 / 4e3, select=1100, recover=1400, itr=65, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, rsm=False, mem=4, rgl=True):

    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    if rsm == False:
        os.system('mkdir -p %s' % tmp_path)
        os.system('rm -rf %s/*' % tmp_path)

        q2n, block = mat_split(qry, tmp_path=tmp_path,
                               chunk=chunk, cpu=cpu, sym=sym, mem=mem)

        N = len(q2n)

        # save q2n to disk
        print 'saving q2n to disk'
        _o = open(tmp_path + '_dict.pkl', 'wb')
        cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
        _o.close()

        del q2n
        gc.collect()
    else:
        f = open(tmp_path + '_dict.pkl', 'rb')
        q2n = cPickle.load(f)
        N = len(q2n)
        #os.system('rm %s/*new* %s/*old'%(tmp_path, tmp_path))
        for tmp in os.listdir(tmp_path):
            if tmp.endswith('_old'):
                a_tmp = tmp_path + '/' + tmp
                b_tmp = tmp_path + '/' + tmp.split('_old')[0]
                os.system('mv %s %s' % (a_tmp, b_tmp))

        os.system('rm %s/*new*' % tmp_path)
        f.close()

    #prune = min(prune, 100. / N)
    shape = (N, N)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    #fns, cvg, nnz = rnorm(qry, shape, tmp_path, csr=False, cpu=cpu, check=True, rgl=True, prune=prune)
    #fns, cvg, nnz = rnorm(qry, shape, tmp_path, csr=False, cpu=cpu, check=False, rgl=False, prune=prune)
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False,
                         cpu=cpu, prune=prune, diag=False)

    #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
    chaos = pruning(qry, tmp_path, prune=prune, S=select, R=recover, cpu=cpu)

    # get the Mg
    for i in os.listdir(tmp_path):
        if i.endswith('.npz') and 'new' not in i:
            j = tmp_path + '/' + i
            os.system('cp %s %s_Mg.npz' % (j, j))

    # print 'finish norm', cvg
    changed = 0

    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        if i == 0:
            row_sum, fns, nnz = regularize(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = regularize(
                qry, shape, tmp_path, True, I, prune, cpu)

        # if i > check and i % check == 0:
        #    print 'reorder the matrix'
        #    fns, cvg, nnz = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, prune=prune)
        #    #fns, cvg, nnz = rnorm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu, rgl=False, prune=prune)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)
        # else:
        #    #os.system('rm %s/*.npz_old'%tmp_path)
        #    fns, cvg, nnz = norm(qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu, prune=prune)
        #    #fns, cvg, nnz = rnorm(qry, shape, tmp_path, row_sum=row_sum, csr=True, check=False, cpu=cpu, rgl=False, prune=prune)

        fns, cvg, nnz = norm(qry, shape, tmp_path,
                             row_sum=row_sum, csr=True, cpu=cpu, prune=prune)
        #pruning(qry, tmp_path, prune=1/50., S=50, R=50, cpu=cpu)
        chao_old = chaos
        chaos = pruning(qry, tmp_path, prune=prune,
                        S=select, R=recover, cpu=cpu)
        changed = abs(chaos - chao_old) < 1e-9 and changed + 1 or 0
        print 'current_chaos', i, chaos, chao_old

        #if chaos < 1e-3 or changed >= 5:
        if chaos < 1e-3:
            break

        if nnz < chunk / 4 and len(fns) > cpu ** 2:
            # if nnz < chunk / 4 or nnz <= N:
            # if 0:
            print 'we try to merge 4 block into one', nnz, chunk / 4
            row_sum_new, fns_new, nnz_new, merged = rmerge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    '''
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        g = load_matrix(fn, shape, True)
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()
    '''

    g = load_matrix(fns[0], shape, True)
    #cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        if fn.endswith('_Mg.npz'):
            continue
        g += load_matrix(fn, shape, True)
        #ci = csgraph.connected_components(g)
        #cs = merge_connected(cs, ci)

    cs = csgraph.connected_components(g)
    del g
    gc.collect()

    # print 'find components', cs
    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl_gpu0(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, gpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    prune = min(prune, 100. / N)
    #shape = (N, N)
    shape = (block, block)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm(qry, shape, tmp_path, csr=False, cpu=cpu)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        row_sum, fns, nnz = expand_gpu(
            qry, shape, tmp_path, True, I, prune, gpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm(qry, shape, tmp_path,
                                 row_sum=row_sum, csr=True, cpu=cpu)

        if nnz < chunk / 4 and len(fns) / 4 > cpu:
            # if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4, len(fns)
            row_sum_new, fns_new, nnz_new, merged = merge_submat(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    g = load_matrix(fns[0], shape, True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        try:
            g = load_matrix(fn, shape, True)
        except:
            continue
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


def mcl_gpu1(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, gpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split_gpu(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    #prune = min(prune, 100. / N)

    #shape = (N, N)
    shape = (block, block)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm_gpu(qry, shape, tmp_path, csr=False, cpu=cpu)
    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i, os.listdir(tmp_path)
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        row_sum, fns, nnz = expand_gpu(
            qry, shape, tmp_path, True, I, prune, gpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm_gpu(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm_gpu(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu)

        # if 0:
        if nnz < chunk / 4 and len(fns) / 4 > cpu:
            # if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4, len(fns)
            row_sum_new, fns_new, nnz_new, merged = merge_submat_gpu(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
                shape = (shape[0] * 2, shape[1] * 2)
                print 'merge_shape is', shape
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    g = load_matrix_gpu(fns[0], (N, N), True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        try:
            g = load_matrix_gpu(fn, (N, N), True)
        except:
            continue
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# add pruning function after normalization
def mcl_gpu(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, gpu=1, mem=4):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split_gpu(qry, chunk=chunk, cpu=cpu, sym=sym, mem=mem)
    N = len(q2n)
    #prune = min(prune, 100. / N)

    # save q2n to disk
    print 'saving q2n to disk'
    _o = open(tmp_path + '_dict.pkl', 'wb')
    cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
    _o.close()

    del q2n
    gc.collect()

    #shape = (N, N)
    shape = (block, block)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm_gpu(qry, shape, tmp_path, csr=False, cpu=cpu)
    #pruning(qry, tmp_path, cpu=cpu)
    pruning(qry, tmp_path, prune=prune, cpu=cpu)

    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i, os.listdir(tmp_path)
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        row_sum, fns, nnz = expand_gpu(
            qry, shape, tmp_path, True, I, prune, gpu)
        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm_gpu(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm_gpu(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu)

        #pruning(qry, tmp_path, cpu=cpu)
        pruning(qry, tmp_path, prune=prune, cpu=cpu)

        # if 0:
        if nnz < chunk / 4 and len(fns) / 4 > cpu:
            # if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4, len(fns)
            row_sum_new, fns_new, nnz_new, merged = merge_submat_gpu(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
                shape = (shape[0] * 2, shape[1] * 2)
                print 'merge_shape is', shape
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    '''
    g = load_matrix_gpu(fns[0], (N, N), True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        try:
            g = load_matrix_gpu(fn, (N, N), True)
        except:
            continue
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()
    '''

    g = load_matrix_gpu(fns[0], (N, N), True)
    for fn in fns[1:]:
        try:
            g += load_matrix_gpu(fn, (N, N), True)
        except:
            continue
        #ci = csgraph.connected_components(g)
        #cs = merge_connected(cs, ci)

    cs = csgraph.connected_components(g)
    del g
    gc.collect()

    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# reduce memory usage of cpu version of mcl
def mcl_lite(qry, tmp_path=None, xy=[], I=1.5, prune=1e-4, itr=100, rtol=1e-5, atol=1e-8, check=5, cpu=1, chunk=5 * 10**7, outfile=None, sym=False, gpu=1):
    if tmp_path == None:
        tmp_path = qry + '_tmpdir'

    os.system('rm -rf %s' % tmp_path)

    q2n, block = mat_split_gpu(qry, chunk=chunk, cpu=cpu, sym=sym)
    N = len(q2n)
    #prune = min(prune, 100. / N)

    # save q2n to disk
    print 'saving q2n to disk'
    _o = open(tmp_path + '_dict.pkl', 'wb')
    cPickle.dump(q2n, _o, cPickle.HIGHEST_PROTOCOL)
    _o.close()

    del q2n
    gc.collect()

    #shape = (N, N)
    shape = (block, block)
    # reorder matrix
    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=False, block=block, cpu=cpu)
    # norm
    fns, cvg, nnz = norm_gpu(qry, shape, tmp_path, csr=False, cpu=cpu)
    #pruning(qry, tmp_path, cpu=cpu)
    pruning(qry, tmp_path, prune=prune, cpu=cpu)

    # print 'finish norm', cvg
    # expension
    for i in xrange(itr):
        print '#iteration', i, os.listdir(tmp_path)
        # row_sum, fns = expend(qry, shape, tmp_path, True, prune=prune,
        # cpu=cpu)
        #row_sum, fns = expend(qry, shape, tmp_path, True, I, prune, cpu)
        # if i > 0 and i % (check * 2) == 0:
        #    #q2n, row_sum, fns, nnz = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block)
        #    #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True)

        #row_sum, fns, nnz = expand_gpu(qry, shape, tmp_path, True, I, prune, gpu)

        if i == 0:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu, fast=True)
        else:
            row_sum, fns, nnz = expand(
                qry, shape, tmp_path, True, I, prune, cpu)

        if i > 0 and i % check == 0:
            print 'reorder the matrix'
            fns, cvg, nnz = norm_gpu(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, check=True, cpu=cpu)
            #q2n, fns = mat_reorder(qry, q2n, shape=shape, chunk=chunk, csr=True, block=block, cpu=cpu)

        else:
            #os.system('rm %s/*.npz_old'%tmp_path)
            fns, cvg, nnz = norm_gpu(
                qry, shape, tmp_path, row_sum=row_sum, csr=True, cpu=cpu)

        #pruning(qry, tmp_path, cpu=cpu)
        pruning(qry, tmp_path, prune=prune, cpu=cpu)

        # if 0:
        if nnz < chunk / 4 and len(fns) / 4 > cpu:
            # if nnz < chunk / 4:
            print 'we try to merge 4 block into one', nnz, chunk / 4, len(fns)
            row_sum_new, fns_new, nnz_new, merged = merge_submat_gpu(
                fns, shape, csr=True, cpu=cpu)
            #row_sum_new, fns_new, nnz_new, merged = merge_submat(fns, shape, csr=True)
            if merged:
                row_sum, fns, nnz = row_sum_new, fns_new, nnz_new
                shape = (shape[0] * 2, shape[1] * 2)
                print 'merge_shape is', shape
            else:
                print 'we failed to merge'
        else:
            print 'current max nnz is', nnz, chunk, chunk / 4

        if cvg:
            # print 'yes, convergency'
            break

    # get connect components
    print 'construct from graph', fns
    g = load_matrix_gpu(fns[0], (N, N), True)
    cs = csgraph.connected_components(g)
    for fn in fns[1:]:
        try:
            g = load_matrix_gpu(fn, (N, N), True)
        except:
            continue
        ci = csgraph.connected_components(g)
        cs = merge_connected(cs, ci)

    del g
    gc.collect()

    # load q2n
    f = open(tmp_path + '_dict.pkl', 'rb')
    q2n = cPickle.load(f)
    f.close()
    os.system('rm %s_dict.pkl' % tmp_path)

    # print 'find components', cs
    groups = {}
    for k, v in q2n.iteritems():
        c = cs[1][v]
        try:
            groups[c].append(k)
        except:
            groups[c] = [k]

    del c
    gc.collect()
    if outfile and type(outfile) == str:
        _o = open(outfile, 'w')
    for v in groups.itervalues():
        out = '\t'.join(v)
        if outfile == None:
            print out
        else:
            _o.writelines([out, '\n'])
    if outfile and type(outfile) == str:
        _o.close()


# print the manual
def manual_print():
    print 'Usage:'
    print '    python %s -i foo.xyz -I 0.5 -a 8' % sys.argv[0]
    print 'Parameters:'
    print '  -i: adjacency matrix. A tab-delimited file which contain 3, 4, 12 or 14 columns'
    print '  -I: float. inflation parameter for mcl'
    print '  -p: float. cutoff of prune parameter'
    print '  -P: float. inverse of -p'

    print '  -R: float. recover parameter'
    print '  -S: float. select parameter'
    print '  -a: int. cpu number'
    print '  -b: int. chunk size. default value is 20000000'
    print '  -o: string. name of output file'
    print '  -d: T|F. is the graph directed? Default is True'
    print '  -g: int. how many gpus to use for speedup. Default is 0'
    print '  -r: T|F. resume the work. Default is F'
    print '  -m: int. memory usage limitation. Deaault is 4GB'
    print '  -A: str. which algorithm to use. currently, mcl and regularized-mcl are support. Default is mcl'


if __name__ == '__main__':

    argv = sys.argv
    # recommand parameter:
    args = {'-i': '', '-I': '1.5', '-a': '2', '-b': '20000000', '-o': None, '-d': 't', '-g': '0',
            '-r': 'f', '-m': '4', '-p': '1/4e3', '-P': '0', '-S': '700', '-R': '800', '-A': 'mcl'}

    N = len(argv)
    for i in xrange(1, N):
        k = argv[i]
        if k in args:
            try:
                v = argv[i + 1]
            except:
                break
            args[k] = v

        elif k[:2] in args and len(k) > 2:
            args[k[:2]] = k[2:]

        else:
            continue

    if args['-i'] == '':
        manual_print()
        raise SystemExit()

    try:
        qry, ifl, cpu, bch, ofn, sym, gpu, rsm, mem, pru, slc, rcv, PRU = args['-i'], float(eval(args['-I'])), int(eval(args['-a'])), int(eval(args['-b'])), args['-o'], args['-d'], int(
            eval(args['-g'])), args['-r'], float(eval(args['-m'])), float(eval(args['-p'])), int(eval(args['-S'])), int(eval(args['-R'])), float(eval(args['-P']))
        alg = args['-A']
        pru = PRU >= 1 and 1. / PRU or pru
        if sym.lower().startswith('f'):
            sym = False
        elif sym.lower().startswith('t'):
            sym = True
        else:
            manual_print()
            raise SystemExit()

        if rsm.lower().startswith('f'):
            rsm = False
        elif rsm.lower().startswith('t'):
            rsm = True
        else:
            manual_print()
            raise SystemExit()

    except:
        manual_print()
        raise SystemExit()

    # convert the relationship into numeric and split into small matrix
    # q2n, xy = mat_split(qry)
    # mul(qry, xy=xy, load=False)
    # mul(qry, load=False)
    # mul(qry, load=True)
    # q2n = mat_split(qry)
    # mul(qry, csr=False)
    gpu = min(cpu, gpu)
    #device = len(cuda.gpus.lst)
    #global CPU
    #CPU = cpu * 8

    # if has_gpu and gpu > 0 and device > 0:
    # if has_gpu and gpu > 0:
    tmp_dir = os.getcwd() + '/' + \
        ofn.split(os.sep)[-1].split(os.sep)[0] + '_tmpdir'
    os.system('mkdir %s' % tmp_dir)

    if gpu > 0:
        #mcl_gpu(qry, I=ifl, cpu=cpu, chunk=bch, outfile=ofn, sym=sym, gpu=gpu, mem=mem)
        mcl_gpu(qry, tmp_path=tmp_dir, I=ifl, cpu=cpu, chunk=bch,
                outfile=ofn, sym=sym, gpu=gpu, mem=mem)

    elif alg == 'mcl':
        #mcl(qry, I=ifl, cpu=cpu, chunk=bch, outfile=ofn, sym=sym, mem=mem, rsm=rsm)
        #mcl(qry, tmp_path=tmp_dir, I=ifl, cpu=cpu, chunk=bch, outfile=ofn, sym=sym, mem=mem, rsm=rsm)
        mcl(qry, tmp_path=tmp_dir, I=ifl, cpu=cpu, chunk=bch, outfile=ofn,
            sym=sym, mem=mem, rsm=rsm, prune=pru, select=slc, recover=rcv)

    else:
        #mcl(qry, I=ifl, cpu=cpu, chunk=bch, outfile=ofn, sym=sym, mem=mem, rsm=rsm)
        #mcl(qry, tmp_path=tmp_dir, I=ifl, cpu=cpu, chunk=bch, outfile=ofn, sym=sym, mem=mem, rsm=rsm)
        rmcl(qry, tmp_path=tmp_dir, I=ifl, cpu=cpu, chunk=bch, outfile=ofn,
             sym=sym, mem=mem, rsm=rsm, prune=pru, select=slc, recover=rcv)

        #mcl_lite(qry, I=ifl, cpu=cpu, chunk=bch, outfile=ofn, sym=sym)

    # preprocess(qry)
    raise SystemExit()

    # qry = sys.argv[1]
    # refs = sys.argv[2:]
    mats = [elem for elem in os.listdir('./') if elem.endswith('.bin')]
    for i in mats:
        x = load_matrix(i)
        sparse.save_npz(i, x)
        os.system('rm %s' % i)

    mats = [elem for elem in os.listdir('./') if elem.endswith('.bin.npz')]
    qrys = [elem for elem in mats if 'row_' in elem]
    refs = [elem for elem in mats if 'col_' in elem]

    qrys.sort()
    refs.sort()

    for qry in qrys:
        gc.collect()

        start = time()
        x, y = map(sparse.load_npz, [qry, refs[0]])
        print 'loading', time() - start

        start = time()
        z = x * y
        print 'mul by', refs[0], time() - start

        for ref in refs[1:]:
            start = time()
            # y = load_matrix(ref)
            y = sparse.load_npz(ref)
            print 'loading', time() - start

            start = time()
            z += x * y
            print 'mul by', ref, time() - start
