import math
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
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
            pass
        out._backward = _backward
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            # TODO: Implement local gradient accumulation for multiplication
            # Hint: derivative of (a * b) with respect to a is b, and with respect to b is a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            pass
        out._backward = _backward
        
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')
        
        def _backward():
            # TODO: Implement local gradient accumulation for addition
            # Hint: derivative of (a + b) with respect to a and b is 1.0
            self.grad += 1 * out.grad
            other.grad += -1 * out.grad
            pass
        out._backward = _backward 
        return out

    def __neg__(self):
        return self.__mul__(-1)

    def __rsub__(self, other):
        return -self + other

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data ** other.data, (self, other), '**')
        
        def _backward():
            # TODO: Implement local gradient accumulation for addition
            # Hint: derivative of (a + b) with respect to a and b is 1.0
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            pass
        out._backward = _backward 
        return out 
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)

    def relu(self):
        mask = (self.data > 0)
        if (mask):
            out = Value(self.data, (self,), 'relu') 
        else:
            out = Value(0, (self,), 'relu')
        def _backward():
            self.grad += mask * out.grad
            pass
        out._backward = _backward
        return out 

    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1
        visited = []
        topo = []
        abandon, visited, topo = self.backward_dfs(visited, topo)
        topo.reverse()
        for node in topo:
            node._backward()
        # TODO: Implement topological sort (DFS) to order the computation graph
        # and execute _backward() in reverse topological order.
        pass

    def backward_dfs(self, visited, topo):
        if self in visited:
            return [self, visited, topo]

        visited.append(self)
        for i in self._prev:
            abandon, visited, topo = i.backward_dfs(visited, topo)
        topo.append(self)
        return [self, visited, topo]