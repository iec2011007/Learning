# Fill in the TODOs in this exercise, then run
# python 01_scalar.py to see if your solution works!
#
import numpy as np
from theano import function
import theano.tensor as T


def make_scalar():
    return T.scalar()

def log(x):
    return T.log(x)

def add(x, y) :
    return x+y

if __name__ == "__main__":
    a = make_scalar()
    b = make_scalar()
    c = log(b)
    d = add(a, c)
    f = function([a, b], d)
    a = np.cast[a.dtype](1.)
    b = np.cast[b.dtype](2.)
    actual = f(a,b)
    expected = 1. + np.log(2.)
    assert np.allclose(actual, expected)
    print "SUCCESS!"
