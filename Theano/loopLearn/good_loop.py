__author__ = 'root'



import theano
from theano import function
import numpy
import theano.tensor as T

locations = T.imatrix()
values = T.vector()
output_model = T.matrix()

def set_val_at_pos(a_location, a_value, output_model):
    zeros = T.zeros_like(output_model)
    zero_subtensor = zeros[a_location[0], a_location[1]]
    return T.set_subtensor(zero_subtensor, a_value)


results, updates = theano.scan(fn=set_val_at_pos, outputs_info=None, sequences=[locations, values], non_sequences=output_model)

assign_values_at_positions = function([locations,values, output_model], results)

test_locations = numpy.asarray([[1, 1], [2, 3]], dtype=numpy.int32)
test_values = numpy.asarray([42, 50], dtype=numpy.float32)
test_output_model = numpy.zeros((5, 5), dtype=numpy.float32)
print assign_values_at_positions(test_locations, test_values, test_output_model)