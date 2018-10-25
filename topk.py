#!usr/bin/env python
from numba import njit
from scipy import sparse
import numpy as np

# get threshold of large k of colum


@njit(fastmath=True, cache=True)
def topk(indptr, indices, data, k):
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
                #print 'yes'
                ct[col] += 1

        print 'lo', lo[:10]
        print 'hi', hi[:10]
        print 'mi', mi[:10]
        print 'ct', ct[:10]


        #print 'hello'

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

        print 'vs', visit[:10]


        flag = np.any(visit)

    return mi, ct

N = 10000
x = sparse.random(N, N, 1e-2, format='csr')


#print x.min(0).todense()[0, :10]
#print x.max(0).todense()[0, :10]
print x.getnnz(0)


thres, ct = topk(x.indptr, x.indices, x.data, 10)

print thres
print ct
