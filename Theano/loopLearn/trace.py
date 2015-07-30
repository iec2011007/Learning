__author__ = 'root'

#Computing trace of matrix

import theano
from theano import function
import theano.tensor as T
import numpy as np

X = T.matrix('X')

result, update = theano.scan(lambda i,j,t_f : T.cast(X[i,j] + t_f, "float32"),
                             sequences=[T.arange(X.shape[0]),T.arange(X.shape[1])],
                             outputs_info=np.asarray(0, dtype="float32"))

compute_trace =function([X], [result])
x = np.eye(5, dtype=theano.config.floatX)
x[0] = np.arange(5, dtype=theano.config.floatX)
print compute_trace(x)[0]

print np.diagonal(x).sum()