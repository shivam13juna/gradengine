## import libraries
import sys
import pyrootutils
root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)



from src.neuralengine import Neuron, Layer, mlp
from src.entity import Entity
import pytest


def test_Neuron():
	x = [-1.2, 2.2, 1, 3]

	n = Neuron(3)

	output = n(x)

	assert isinstance(output, Entity)
	assert isinstance(output.data, float)


def test_Layer():
	x = [-1.2, 2.2, 1, 3]
	l = Layer(3, 2)
	output = l(x)

	assert isinstance(output, list)
	assert isinstance(output[0], Entity)
	assert isinstance(output[0].data, float) 


def test_mlp():

	x = [-1.2, 2.2, 1, 3]

	nn = mlp(3, [4, 4, 1])

	output = nn(x)

	print(output)


	assert isinstance(output, Entity)
	assert isinstance(output.data, float)
