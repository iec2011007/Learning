__author__ = 'root'

# Computing norms of lines of X

import theano
import theano.tensor as T
import numpy as np

x = T.matrix('X')

results, updates = theano.scan(lambda x_i : T.sqrt((x_i**2).sum()), sequences=x)

compute_norm_lines = theano.function([x], [results])
x = np.diag(np.arange(1, 6, dtype=theano.config.floatX), 1)

print compute_norm_lines(x)[0]