"""Practice rewrite of the autograd engine. Fill in every TODO from scratch.

Tests: uv run pytest phase1_foundations/month01_autograd/test_practice_engine.py -q
"""
import math


class Value:
    """A scalar node in a dynamically built computation graph.

    Attributes the tests inspect:
        data      float, the forward value
        grad      float, d(output)/d(this node), starts at 0.0
        _prev     set of Value, the inputs that produced this node (empty for a leaf)
        _op       str, the operation that produced it ('' for a leaf)
        label     str, optional name
        _backward zero-argument callable; reads this node's grad and ACCUMULATES
                  into the grad of its inputs. Leaves do nothing.
    """

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0
        self._backward = lambda: None
        # TODO: set the attributes listed in the class docstring

    def __repr__(self):
        # TODO: e.g. "Value(data=2.0, grad=0.0)"
        return f"Value(data={self.data}, grad={self.grad})"

    # ---- binary ops: the other operand may be a plain int/float ----

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        ### forward pass
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():    
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            pass
        out._backward = _backward 
        return out
    def __mul__(self, other):
        # TODO: new node, '*', local backward
        other = other if isinstance(other, Value) else Value(other)
        ### forward pass
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():    
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            pass
        out._backward = _backward 
        return out

    def __pow__(self, other):
        #other = other if isinstance(other, Value) else Value(other)
        ### forward pass
        out = Value(self.data ** other, (self,), '**')
        def _backward():    
            self.grad += other * self.data ** (other - 1 ) * out.grad
            pass
        out._backward = _backward 
        return out

    # ---- derived ops: build them from the ones above where you can ----

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):

        other = other if isinstance(other, Value) else Value(other)
        ### forward pass
        out = Value(self.data * other.data ** (-1), (self, other), '/')
        def _backward():    
            self.grad += other.data ** -1  * out.grad 
            other.grad += -1 * self.data * other.data ** -2 * out.grad
            pass
        out._backward = _backward 
        return out


    # ---- reflected ops: make `3 + v`, `3 * v`, `3 - v`, `3 / v` and sum([...]) work ----

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        # TODO
        return self * other

    def __rsub__(self, other):
        return other + (-self)        

    def __rtruediv__(self, other):
        return other *  self ** -1
    # ---- nonlinearities ----

    def relu(self):
        # TODO: gradient at exactly 0 is 0
        out = Value (self.data if self.data > 0 else 0, (self,), 'relu')
        def _backward():
            self.grad += 1 * out.grad if out.data > 0 else 0
        out._backward = _backward
        return out
    def tanh(self):
        # TODO
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - (out.data)**2) * out.grad
        out._backward = _backward
        return out
    
        raise NotImplementedError

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += (out.data) * out.grad
        out._backward = _backward
        return out

    # ---- reverse-mode driver ----

    def backward(self):
        # TODO: order the graph topologically, seed this node's grad with 1.0,
        # then call _backward() on nodes in reverse topological order.
        # Stretch: make the traversal iterative so a 5000-node chain doesn't hit the recursion limit.
        self.grad = 1.0
        visited = []
        abandon, visited = self.backward_sub(visited)
        visited.reverse()
        for i in visited:
            i._backward()

    def backward_sub(self, visited):
        if self in visited:
            return [self, visited]
        for i in self._prev:
            abandon, visited = i.backward_sub(visited)
        visited.append(self)
        return [self,visited]
