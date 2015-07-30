__author__ = 'root'
#Computing the sequence x(t) = x(t - 2).dot(U) + x(t - 1).dot(V) + tanh(x(t - 1).dot(W) + b)

import theano
from theano import function
import numpy as np
import theano.tensor as T

X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")
U = T.matrix("U")
V = T.matrix("V")
n_sym = T.iscalar("n_sym")

result, update = theano.scan(lambda xtm2,xtm1 : T.dot(xtm2,U) + T.dot(xtm1,V) + T.tanh(T.dot(xtm1,W)) + b_sym,
                             n_steps=n_sym, outputs_info=[dict(initial=X, taps=[-2, -1])])

compute_seq2 = theano.function(inputs=[X, U, V, W, b_sym, n_sym], outputs=[result])


x = np.zeros((2, 2), dtype=theano.config.floatX) # the initial value must be able to return x[-2]
x[1, 1] = 1
w = 0.5 * np.ones((2, 2), dtype=theano.config.floatX)
u = 0.5 * (np.ones((2, 2), dtype=theano.config.floatX) - np.eye(2, dtype=theano.config.floatX))
v = 0.5 * np.ones((2, 2), dtype=theano.config.floatX)
n = 10
b = np.ones((2), dtype=theano.config.floatX)

print compute_seq2(x, u, v, w, b, n)
x_res = np.zeros((10, 2))
x_res[0] = x[0].dot(u) + x[1].dot(v) + np.tanh(x[1].dot(w) + b)
x_res[1] = x[1].dot(u) + x_res[0].dot(v) + np.tanh(x_res[0].dot(w) + b)
x_res[2] = x_res[0].dot(u) + x_res[1].dot(v) + np.tanh(x_res[1].dot(w) + b)
for i in range(2, 10):
    x_res[i] = (x_res[i - 2].dot(u) + x_res[i - 1].dot(v) +
                np.tanh(x_res[i - 1].dot(w) + b))
print x_res