__author__ = 'root'

import theano
import theano.tensor as T
from theano import function
import numpy as np

#Sample Data
trX = np.linspace(-1,1,101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = T.scalar()
Y = T.scalar()

def model(X, w) :
    return X * w

w = theano.shared(np.asarray(0, dtype=theano.config.floatX))
y = model(X, w)

cost =  T.mean(T.sqr(y-Y))
gradients = T.grad(cost, w)
updates = [ [w, w- gradients*0.001] ]
train = theano.function([X,Y], cost, updates=updates)

for i in range(100):
    for x, y in zip(trX, trY):
        train(x, y)

print w.get_value() #something around 2

