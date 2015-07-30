__author__ = 'root'
#x(t) = tanh(x(t - 1).dot(W) + y(t).dot(U) + p(T - t).dot(V))

import theano
from theano import function
import theano.tensor as T
import numpy as np

X = T.vector('X')
W = T.matrix('W')
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")

result, update = theano.scan(lambda y,p,x_tm1 : T.tanh(T.dot(x_tm1, W) + T.dot(y, U) + T.dot(p, V)),sequences=[Y, P[::-1]], outputs_info=[X])

compute_seq = theano.function(inputs=[X, W, Y, U, P, V], outputs=[result])



# test values
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
w = np.ones((2, 2), dtype=theano.config.floatX)
y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype=theano.config.floatX)
p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=theano.config.floatX)

y = np.linspace(0,0.5,10).reshape((5,2))
p = np.linspace(0.5,1,10).reshape((5,2))

print y,p
print "Now Result"

print compute_seq(x, w, y, u, p, v)[0]



x_res = np.zeros((5, 2), dtype=theano.config.floatX)
x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4-i].dot(v))
print x_res