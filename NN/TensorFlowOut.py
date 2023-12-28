from graphviz import Digraph
import math

class Node:
  _n = 18
  _i = 0
  def __init__(self, data, _children=(), _op='', indexInLayer=0):
    self.data = data
    self.derivitive = 0.0
    self._backward = lambda: None
    # save the Node which making this operation
    self._prev = set(_children)
    # save the operation
    self._op = _op
    self.indexInLayer = indexInLayer

  def __add__(self, other):
    # to check the added number is a Node object
    other = other if isinstance(other, Node) else Node(other)
    # apply the add operation
    out = Node(self.data + other.data, (self, other), '+')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += out.derivitive
      other.derivitive += out.derivitive
      # if(self._i < self._n):
      #   print(f"add: ------ {self}")
    out._backward = _backward
    return out

  def __mul__(self, other):
    # to check the multiplayed number is a Node object
    other = other if isinstance(other, Node) else Node(other)
    # apply the multiplacation operation
    out = Node(self.data * other.data, (self, other), '*')
    # print(f"mul: ------- {self},  {out}")
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += other.data * out.derivitive
      other.derivitive += self.data * out.derivitive
      # if(self._i < self._n):
      #   print(f"mul: ------ {self._i},  {self}")
      #   print(f"mul2: ------ {out}")
        # print(f"mul children: {self._prev}")
    out._backward = _backward
    return out

  def __pow__(self, other):
    # only the integer and float number code be in the power
    assert isinstance(other, (int, float)), "only supporting int/float powers for now"
    # apply the power operation
    out = Node(self.data**other, (self,), f'**{other}')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += (other * self.data**(other-1)) * out.derivitive
    out._backward = _backward
    return out

  def log(self):
    # apply the log operation
    out = Node(math.log(self.data), (self, ), 'log')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += (1 / self.data) * out.derivitive
    out._backward = _backward
    return out

  def relu(self):
    # apply the ReLU activation function
    out = Node(0 if self.data < 0 else self.data, (self,), 'ReLU')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += (out.data > 0) * out.derivitive
    out._backward = _backward
    return out

  def tanh(self):
    # apply the tanh activation function
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Node(t, (self, ), 'tanh')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += (1 - t**2) * out.derivitive
    out._backward = _backward
    return out

  def softMax(self, exp_neurons):
    # apply the soft max activation function
    x = self.data
    e = math.exp(x)
    # the sum of all exp neurons
    sum_exp_neurons = sum(exp_neurons)
    t = e / sum_exp_neurons
    out = Node(t, (self, ), 'soft_max')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      for i in range(len(exp_neurons)):
        if i == self.indexInLayer:
          self.derivitive += (t * (1 - t)) * out.derivitive
          # self._i += 1
          # print()
          # print("first")
          # print(f"self: {self},  other: ({i}, {exp_neurons[i]})")
          # print()
        else:
          theOther = exp_neurons[i]
          other_y_hat = theOther/sum_exp_neurons
          self.derivitive += -t * other_y_hat * out.derivitive
          # print()
          # print("second")
          # print(f"other: {theOther}, other_exp: {other_y_hat}, my_exp: {t}, out_der: {out.derivitive}")
          # print(f"self: {self},  other: ({i}, {exp_neurons[i]})")
          # print()
      # if(self._i < self._n):
      # print(f"soft_max: ------ {self}")
    out._backward = _backward
    return out

  def crossEntropy(self, label):
    x = self.data
    prediction = math.log(x)
    cross_entropy = (-1 * label * prediction)
    out = Node(cross_entropy, (self, ), 'cross_entropy')
    # out.derivitive = 1
    # print()
    # print("cross")
    # print(out)
    # print()
    def _backward():
      self.derivitive += (prediction - label) * out.derivitive
      # print(f"cross----------: {self},  {out}")
    out._backward = _backward
    return out
  
  def sigmoid(self):
    # apply the sigmoid activation function
    x = self.data
    t = (1 / (1 + math.exp(-x)))
    out = Node(t, (self, ), 'sigmoid')
    # update the derivitive (which in in the backward phase of the backpropagation)
    def _backward():
      self.derivitive += (t * (1 - t)) * out.derivitive
    out._backward = _backward
    return out

  # the backword phase of the backpropagation algorithm
  # should be called on the final Node 
  def backward(self):
    # First: apply topological order all of the children in the graph
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)
    # Second: go one variable at a time and apply the chain rule to get its derivitive (the "_backword" function)
    self.derivitive = 1.0
    for node in reversed(topo):
      node._backward()
      # if self._i < self._n:
      # print(node)
      #   self._i += 0.5
    
  # for the negative Node
  def __neg__(self): 
    return self * -1
  
  # for the add in the revers order as in (number + Node.object) reverse it to (Node.object + number)
  def __radd__(self, other):
    return self + other
  
  # the subtraction is an add operation with a negative value for the other object
  def __sub__(self, other):
    return self + (-other)

  # for the subtraction in the revers order as in (number - Node.object) reverse it to (Node.object - number)
  def __rsub__(self, other): # other - self
    return other + (-self)
  
  # for the multiplacation in the revers order as in (number * Node.object) reverse it to (Node.object * number)
  def __rmul__(self, other):
    return self * other
  
  # for the division operation as a multiplacation with a value to the power -1
  def __truediv__(self, other): 
    return self * other**-1
  
  # for the multiplacation in the revers order as in (number / Node.object) reverse it to (Node.object / number)
  def __rtruediv__(self, other):
    return other * self**-1
  
  # for printing the Node value (the data and the derivitiveiant)
  def __repr__(self):
    return f"Node(index={self.indexInLayer}, data={self.data}, derivitive={self.derivitive})"
  
  
class Draw:
  def __init__():{}

  def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    def trace(root):
      # builds a set of all nodes and edges in a graph
      nodes, edges = set(), set()
      def build(v):
        if v not in nodes:
          nodes.add(v)
          for child in v._prev:
            edges.add((child, v))
            build(child)
      build(root)
      return nodes, edges
    
    nodes, edges = trace(root)
    for n in nodes:
      uid = str(id(n))
      # for any value in the graph, create a rectangular ('record') node for it
      dot.node(name = uid, label = "{ data %.4f | derivitive %.4f }" % (n.data, n.derivitive), shape='record')
      if n._op:
        # if this value is a result of some operation, create an op node for it
        dot.node(name = uid + n._op, label = n._op)
        # and connect this node to it
        dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
      # connect n1 to the op node of n2
      dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot