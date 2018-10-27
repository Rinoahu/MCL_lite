#!usr/bin/env python

from scipy import sparse
import sys
sys.path.append('../')
import mcl_sparse


x = sparse.random(10000, 10000, 1e-2, format='csr')

a = mcl_sparse.topks_p(x.indptr, x.indices, x.data, 10) 

b = mcl_sparse.topks(x.indptr, x.indices, x.data, 10) 


