#!usr/bin/env pythnon
from scipy import sparse

N = D = 5000
P = 1e6/N/N
#x = sparse.random(10000, 10000, 1e-3, format='csr')
x = sparse.random(D, D, P, format='csr')
#x = sparse.load_npz('tmp.npz')

import sys
sys.path.append('../')
import mcl_sparse
from time import time




#N, D =x.shape
#P = 1e5 / N / D

for i in xrange(100):
    y = mcl_sparse.sparse.random(N, D, P, format='csr')
    st = time()
    z1 = mcl_sparse.csrmm_p_ez_fast(x, y, cpu=8)

    print time() - st, z1.nnz

    print '#' * 89

    st = time()
    z2 = mcl_sparse.csrmm_p_ez(x, y, cpu=1)

    print time() - st, z2.nnz

    z3 = x * y

    print z3.nnz

    dif = z3 - z1

    print 'diff nnz', dif.nnz
    print 'dif max min', dif.max(), dif.min()

    #break
