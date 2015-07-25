# Fill in the TODOs in this exercise, then run
# python 01_function.py to see if your solution works!
#
from theano import tensor as T
from theano import function

def evaluate(x, y, expr, x_value, y_value):
    """
    x: A theano variable
    y: A theano variable
    expr: A theano expression involving x and y
    x_value: A numpy value
    y_value: A numpy value

    Returns the value of expr when x_value is substituted for x
    and y_value is substituted for y
    """
    f = function([x,y],z)
    return f(x_value, y_value)

if __name__ == "__main__":
    x = T.iscalar()
    y = T.iscalar()
    z = x + y
    assert evaluate(x, y, z, 1, 2) == 3
    print "SUCCESS!"
