import math


class Entity:

    def __init__(self, data, label="", _children=(), _op="", grad=0.0):
        self.data = data
        self._children = _children
        self.label = label
        self._op = _op
        self.grad = grad
        self.backward = lambda: None

    def __repr__(self):
        # return f"Entity data: {self.data}, label: {self.label}, _op: {self._op}, _children: {self._children}"
        return f"Entity data: {self.data}"

    def __str__(self):
        # return f"Entity data: {self.data}, label: {self.label}, _op: {self._op}, _children: {self._children}"
        return f"Entity data: {self.data}"

    # Defining how classes should add
    def __add__(self, other):
        other = other if isinstance(other, Entity) else Entity(other, label=str(other))

        ret = Entity(self.data + other.data, _children=(self, other), _op="+", label=f"{self.label} + {other.label}")

        def _backward():
            self.grad += ret.grad * 1
            other.grad += ret.grad * 1

        ret.backward = _backward

        return ret

    def __radd__(self, other):
        return self + other

    # Defining how classes should subtract
    def __sub__(self, other):
        other = other if isinstance(other, Entity) else Entity(other, label=str(other))

        ret = Entity(self.data - other.data, _children=(self, other), _op="-", label=f"{self.label} - {other.label}")

        def _backward():
            self.grad += ret.grad * 1
            other.grad += ret.grad * -1

        ret.backward = _backward

        return ret

    def __rsub__(self, other):
        return self - other

    # Defining how classes should multiply
    def __mul__(self, other):
        other = other if isinstance(other, Entity) else Entity(other, label=str(other))

        ret = Entity(self.data * other.data, _children=(self, other), _op="*", label=f"{self.label} * {other.label}")

        def _backward():
            self.grad += ret.grad * other.data
            other.grad += ret.grad * self.data

        ret.backward = _backward

        return ret

    def __rmul__(self, other):
        return self * other

    # Defining how classes should divide
    def __truediv__(self, other):
        other = other if isinstance(other, Entity) else Entity(other, label=str(other))

        ret = Entity(self.data / other.data, _children=(self, other), _op="/", label=f"{self.label} / {other.label}")

        def _backward():
            self.grad += ret.grad * (1 / other.data)
            other.grad += ret.grad * (-self.data / other.data**2)

        ret.backward = _backward

        return ret

    def __rtruediv__(self, other):
        return self / other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power can only be int/float"

        ret = Entity(math.pow(self.data, other), _children=(self,), _op="**")

        def _backward():
            self.grad += ret.grad * (other * (self.data ** (other - 1)))

        ret.backward = _backward

        return ret

    def tanh(self):

        ret = Entity(
            (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1),
            _children=(self,),
            _op="tanh",
        )

        def _backward():
            self.grad += ret.grad * (1 - ret.data**2)

        ret.backward = _backward

        return ret

    def exp(self):

        ret = Entity(math.exp(self.data), _children=(self,), _op="exp")

        def _backward():
            self.grad += ret.grad * ret.data

        ret.backward = _backward

        return ret

    def sigmoid(self):

        ret = Entity(1 / (1 + math.exp(-self.data)), _children=(self,), _op="sigmoid")

        def _backward():
            self.grad += ret.grad * ret.data * (1 - ret.data)

        ret.backward = _backward

        return ret

    def relu(self):

        ret = Entity(max(0, self.data), _children=(self,), _op="relu")

        def _backward():
            self.grad += ret.grad * (1 if self.data > 0 else 0)

        ret.backward = _backward

        return ret

    def backprop(self):

        nodes = set()
        topo = []

        def iterate(node):
            if node not in nodes:
                nodes.add(node)
                for child in node._children:
                    iterate(child)
                topo.append(node)

        iterate(self)

        self.grad = 1
        for node in reversed(topo):
            node.backward()

        return self
    

if __name__ == "__main__":
    # some tests

    test1 = Entity(5, 'test1')
    test2 = Entity(5, 'test2')

    test3 = test1 + test2; test3.label = 'test3'
    test4 = 5 + test3 ; test4.label = 'test4'

    test5 = test4/test3; test5.label = 'test5'
    test6 = test5 * test4; test6.label = 'test6'

    test7 = test6 ** 5; test7.label = 'test7'

    test8 = test1 + test2 + test3 + test4 + test5 + test6 + test7; test8.label = 'test8'


    to_test = [test3, test4, test5, test6, test7]

    for var in reversed(to_test):
        print('label:', var.label)
        print('data:', var.data)
        print('_children:', var._children)
        print('\n')

