__author__ = 'root'

import theano
from theano import function
import numpy as np
import theano.tensor as T

#Computing tanh(x(t).dot(W) + b) elementwise

X = T.matrix('x')
W = T.matrix('w')

b_sym = T.vector('b_sym')

result, update = theano.scan(lambda x_i : T.tanh(T.dot(x_i, W) + b_sym), sequences=X)
f = function([X,W, b_sym], result)

x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print f(x,w,b)
print np.tanh(x.dot(w)+b)
