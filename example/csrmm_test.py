#!usr/bin/env pythnon
from scipy import sparse

#x = sparse.random(10000, 10000, 1e-3, format='csr')
x = sparse.load_npz('tmp.npz')

import sys
sys.path.append('../')
import mcl_sparse
from time import time



#z = mcl_sparse.csrmm_ez_ms_slow(x, x)
#print z.nnz

#z0 = mcl_sparse.csrmm_ez_ms_slow_p(x, x)

st = time()
z1 = mcl_sparse.csrmm_ez_ms_slow_p(x, x, cpu=2)

print time() - st, z1.nnz

print '#' * 89

st = time()
z2 = mcl_sparse.csrmm_ez_ms_slow_p(x, x, cpu=1)

print time() - st, z2.nnz

#z2 = x * x

#print z2.nnz
#dif = z1 - z0

#print dif.nnz


