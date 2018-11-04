#!usr/bin/env python

import sys
sys.path.append('../')

import mcl_sparse

x = mcl_sparse.sparse.random(10000, 10000, 1e-3, format='csr')
y = mcl_sparse.sparse.random(10000, 10000, 1e-3, format='csr')

z = mcl_sparse.csram_ez_ms(x, y)

print x.indptr
print y.indptr

zp = mcl_sparse.csram_p_ez(x, y, cpu=3)


dif = zp - z
print dif.nnz
