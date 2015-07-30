__author__ = 'root'

import theano
from theano import function
import numpy as np
import theano.tensor as T

coefficients = T.vector('coeff')
x = T.scalar('x')

max_supported_coeff = 10000

results, updates = theano.scan(lambda coeff, power, free_var: coeff * (free_var ** power),
                               sequences=[coefficients, T.arange(max_supported_coeff)],
                               outputs_info=None, non_sequences=x)
results = results.sum()
calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=results)
test_coefficients = np.asarray([1, 0, 2], dtype=np.float32)
test_value = 3
print calculate_polynomial(test_coefficients, test_value)