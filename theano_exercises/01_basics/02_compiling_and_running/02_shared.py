# Fill in the TODOs in this exercise, then run
# python 01_function.py to see if your solution works!
#
from collections import OrderedDict
import numpy as np
import theano
import  theano.tensor as T
from theano import function

def make_shared(shape):
    """
    Returns a theano shared variable containing a tensor of the specified
    shape.
    You can use any value you want.
    """
    return theano.shared(np.zeros(shape))

def exchange_shared(a, b):
    """
    a: a theano shared variable
    b: a theano shared variable
    Uses get_value and set_value to swap the values stored in a and b
    """
    t1 = a.get_value()
    t2 = b.get_value()
    a.set_value(t2)
    b.set_value(t1)

def make_exchange_func(a, b):
    """
    a: a theano shared variable
    b: a theano shared variable
    Returns f
    where f is a theano function, that, when called, swaps the
    values in a and b
    f should not return anything
    """
    #Finally Got the answer by the below code works for anything you want to do with shared variable
    #You can define a function and the new states of the variables are transferred through the
    #updates parameter in the theano.funciton
    updates = OrderedDict()
    updates[a] = b
    updates[b] = a
    return function([], updates=updates)



if __name__ == "__main__":
    a = make_shared((5, 4, 3))
    assert a.get_value().shape == (5, 4, 3)
    b = make_shared((5, 4, 3))
    assert a.get_value().shape == (5, 4, 3)
    a.set_value(np.zeros((5, 4, 3), dtype=a.dtype))
    b.set_value(np.ones((5, 4, 3), dtype=b.dtype))
    exchange_shared(a, b)
    assert np.all(a.get_value() == 1.)
    assert np.all(b.get_value() == 0.)
    f = make_exchange_func(a, b)
    rval = f()
    assert isinstance(rval, list)
    assert len(rval) == 0
    assert np.all(a.get_value() == 0.)
    assert np.all(b.get_value() == 1.)

    print "SUCCESS!"
