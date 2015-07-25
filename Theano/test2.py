__author__ = 'root'

#Below code computes A**K in python using theano
# The variable A will be vector and K will be a constant yeilding a vector where each
#element has been raised to the power K

import theano
import theano.tensor as T
import numpy as np

k = T.iscalar('k')
A = T.vector('A')

result, updates = theano.scan(lambda prior_result, A : prior_result*A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

final_result = result[-1]

power = theano.function([A, k], outputs=final_result, updates=updates)

print power(range(10),2)