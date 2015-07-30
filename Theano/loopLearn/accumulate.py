__author__ = 'root'

import theano
from theano import function
import numpy as np
import theano.tensor as T

k = theano.shared(0)
n_sym = T.iscalar("n_sym")

results, updates = theano.scan(lambda : {k:k+1}, n_steps=n_sym)
accumulator = function([n_sym], [], updates=updates)

print k.get_value()
accumulator(5)
print k.get_value()