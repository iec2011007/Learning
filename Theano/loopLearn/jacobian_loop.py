__author__ = 'root'

#Computing the Jacobian of y = tanh(v.dot(A)) wrt x
import theano
from theano import function
import numpy as np
import theano.tensor as T

v = T.vector()
A = T.matrix()

y = T.tanh(T.dot(v,A))

results, updates = theano.scan(lambda i: T.grad(y[i],v), sequences=[T.arange(y.shape[0])])

f = function([A, v], [results])
x = np.eye(5, dtype=theano.config.floatX)[0]
w = np.eye(5, 3, dtype=theano.config.floatX)
w[2] = np.ones((3), dtype=theano.config.floatX)
print f(w, x)[0]

# compare with numpy
print ((1 - np.tanh(x.dot(w)) ** 2) * w).T