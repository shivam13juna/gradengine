from src.entity import Entity
import numpy as np

class Neuron:

	def __init__(self, no_inputs):
		self.w = [Entity(np.random.normal(-1, 1)) for _ in range(no_inputs)]
		self.b = Entity(np.random.normal(-1, 1))

	def __call__(self, x):
		out = sum([w * x for w, x in zip(self.w, x)], self.b)
		tanh_out = out.tanh()
		return tanh_out
	
	def parameters(self):
		return self.w + [self.b]
	
class Layer:
	def __init__(self, no_inputs, no_neurons):
		self.layer = [Neuron(no_inputs) for _ in range(no_neurons)]

	def __call__(self, x):
		out = [n(x) for n in self.layer]
		return out[0] if len(out) == 1 else out
	
	def parameters(self):
		return [p for neuron in self.layer for p in neuron.parameters()]

class mlp:
	def __init__(self, no_inputs, layer_list):
		self.layer_list = [no_inputs] + layer_list
		self.nn = [Layer(self.layer_list[i-1], self.layer_list[i]) for i in range(1, len(self.layer_list))]

	def __call__(self, x):
		for layer in self.nn:
			x = layer(x)

		return x
	
	def parameters(self):
		return [p for layer in self.nn for p in layer.parameters()]


if __name__ == "__main__":
	x = [-1.2, 2.2, 1, 3]
	n = Neuron(3)
	print("output of neuron:", n(x))

	l = Layer(3, 2)
	print("output of layer:", l(x))

	nn = mlp(3, [4, 4, 1])
	print("output of mlp:", nn(x))
