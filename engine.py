class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        # Internal variables for computational graph construction (DAG nodes & edges)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # TODO: Implement local gradient accumulation for addition
            # Hint: derivative of (a + b) with respect to a and b is 1.0
            pass
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # TODO: Implement local gradient accumulation for multiplication
            # Hint: derivative of (a * b) with respect to a is b, and with respect to b is a
            pass
        out._backward = _backward
        
        return out

    def backward(self):
        # TODO: Implement topological sort (DFS) to order the computation graph
        # and execute _backward() in reverse topological order.
        pass