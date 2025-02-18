## import libraries
import sys
import pyrootutils
root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)



from src.entity import Entity
import pytest


def test_entity_backprop():
	#inputs x1, x2
	x1 = Entity(2, label='x1')
	x2 = Entity(0.0, label='x2')

	#weights w1, w2
	w1 = Entity(-3, label='w1')
	w2 = Entity(1, label='w2')

	#bias
	b = Entity(6.8813735870195432, label='b')

	#output
	x1w1 = x1 * w1; x1w1.label = 'x1*w1'
	x2w2 = x2 * w2; x2w2.label = 'x2*w2'
	x1w1_x2w2 = x1w1 + x2w2; x1w1_x2w2.label = 'x1*w1 + x2*w2'
	n = x1w1_x2w2 + b; n.label = 'n'
	o = n.tanh(); o.label = 'o'

	list_of_objects = [x1, x2, w1, w2, b, x1w1, x2w2, x1w1_x2w2, n, o]

	# creat a dict of grad for each object
	o.backprop()

	grad_dict = {}

	for obj in list_of_objects:
		grad_dict[obj.label] = round(obj.grad, 2)



	base_grad_dict = {'x1': -1.5, 'x2': 0.5, 'w1': 1.0, 'w2': 0.0, 'b': 0.5, 'x1*w1': 0.5, 'x2*w2': 0.5, 'x1*w1 + x2*w2': 0.5, 'n': 0.5, 'o': 1}


	for key in grad_dict:
		assert grad_dict[key] == base_grad_dict[key]


#test8 = test1 + test2 + test3 + test4 + test5 + test6 + test7; test8.label = 'test8'