__author__ = 'root'

#Program to compute the following using theano

#Computing tanh(x(t).dot(W) + b) elementwise

import theano
import theano.tensor as T
import numpy as np

x = T.matrix('x')
w = T.matrix('w')
b = T.vector('b')

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, w) + b), sequences=x)

compute_elemWise = theano.function([x,w,b], [results])

#testing values

x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2,2), dtype = theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
print "printing x, w ", x, w
b[1] = 2
print compute_elemWise(x, w, b)[0]

#Computing the same thing using numpy

print np.tanh(x.dot(w) + b)