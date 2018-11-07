#!usr/bin/env python

import sys
sys.path.append('../')
import mcl_sparse

#x = mcl_sparse.sparse.random(10000, 10000, 1e-3, format='csr')


#for i in xrange(100):
#    y = mcl_sparse.sparse.random(10000, 10000, 1e-3, format='csr')
#    z = x + y
#    zp = mcl_sparse.csram_p_ez(x, y, cpu=3)
#    dif = zp - z
#    print y.sum(), dif.nnz, dif.max(), dif.min()



for i in xrange(100):
	x0 = mcl_sparse.sparse.random(10000, 10000, 1.23e-3, format='csr')
	x1 = mcl_sparse.sparse.random(10000, 10000, 1.23e-3, format='csr')
	y = mcl_sparse.csram_p_ez(x0, x1, prefix='tmp.npy', cpu=12, disk=True)
	y1 = x0+x1
	ydif = y - y1

	z = mcl_sparse.csrmm_p_ez(y, y, prefix='tmp1.npy', cpu=12, disk=True)
	z1 = y1 * y1
	zdif = z - z1

	#print '+', ydif.max(), ydif.min(), 'x', zdif.max(), zdif.min()
	#a, b, c, d = ydif.max(), ydif.min(), zdif.max(), zdif.min()
	print '+', abs(ydif.data).max(), 'x', abs(zdif.data).max()
	

