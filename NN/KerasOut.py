import random
import math
from NN.TensorFlowOut import Node

class KerasOut:
  def zero_derivitive(self):
    for param in self.parameters():
      param.derivitive = 0.0

  def parameters(self):
    return []
  
class Neuron(KerasOut):
  def __init__(self, NumberOfInputs, indexInLayer, activationFunction='linear'):
    # initialize the wight array and the bias for this single neuron and determine the activation function
    # the wight array will be of the size of the input to this neuron 
    # while the bias is one
    self.weight = [Node(random.uniform(-0.9, 0.9), indexInLayer=indexInLayer) for _ in range(NumberOfInputs)]
    self.bias = Node(random.uniform(-0.9, 0.9), indexInLayer=indexInLayer)
    self.activation_function = activationFunction
    self.indexInLayer = indexInLayer

  # for the forward phase of backpropagation
  def __call__(self, data):
    # if(not self.activation_function == 'soft_max'):
    f_wb = sum((weight_i*data_i for weight_i, data_i in zip(self.weight, data)), self.bias)
    # f_wb = 0
    # for weight_i in self.weight:
    #   for data_i in data:
    #     f_wb += weight_i * data_i
    #   f_wb += self.bias
    # if self.activation_function == "soft_max":
      # print(f"--------------------------------- {self.indexInLayer} -----------------------------------")
      # print(f_wb)
      # print(f_wb._prev)
    # else:
    #   f_wb = sum([data_i.exp() for data_i in data])
    if(self.activation_function == 'linear'):
      return f_wb
    elif(self.activation_function == 'sigmoid'):
      return f_wb.sigmoid()
    elif(self.activation_function == 'tanh'):
      return f_wb.tanh()
    elif(self.activation_function == 'relu'):
      return f_wb.relu()
    elif(self.activation_function == 'soft_max'):
      return f_wb
    else:
      return TypeError(f"activation function {self.activation_function} is not available")
  
  def parameters(self):
    # return all the parameters of this neuron as a list
    return self.weight + [self.bias]
  
  def __repr__(self):
    return f"{self.activation_function} Neuron ({self.indexInLayer},{len(self.weight)})"

class Layer(KerasOut):
  def __init__(self, numberOfInputsToLayer, NumberOfNeuronInLayer, activationFunction='linear'):
    self.neurons = [Neuron(numberOfInputsToLayer, i, activationFunction) for i in range(NumberOfNeuronInLayer)]
    self.activation_function = activationFunction
  
  # for the forward phase of backpropagation
  def __call__(self, data):
    LayerNeuronsF_wb = [neuron(data) for neuron in self.neurons]
    if self.activation_function == 'soft_max':
      # n = []
      # for i in range(len(self.neurons)):
      #   n.append(0) 
      # for i in range(len(self.neurons)):
      #   self.neurons[i] = self.neurons[0](data)
      #   n[i] = 1
      # a float numbers copy to calculate the exps
      lst = [math.exp(neuron.data) for neuron in LayerNeuronsF_wb]
      # calculate the soft max for each neuron in this layer    
      # print("\nbefor")
      # print(LayerNeuronsF_wb)
      # print()
      LayerNeuronsF_wb = [ neuron.softMax(lst) for neuron in LayerNeuronsF_wb]
      # print("\nafter")
      # print(LayerNeuronsF_wb)
      # print("DASDA\n")
    # return LayerNeuronsF_wb[0] if len(LayerNeuronsF_wb) == 1 else LayerNeuronsF_wb
    return LayerNeuronsF_wb
  
  def parameters(self):
    return [param for neuron in self.neurons for param in neuron.parameters()]

  def __repr__(self): 
    return f"Layer of [{', '.join(str(neuron) for neuron in self.neurons)}]"

class MultiLayerPerceptron(KerasOut):
  def __init__(self, numberOfFeaturesInInputLayer, NumberOfNeuronInEachLayer, EachLayerActivation):
    # if EachLayerActivation[-1] == 'soft_max':
    #   # max the last layer linear add a softmax after it
    #   EachLayerActivation[-1] = EachLayerActivation[-2]
    #   EachLayerActivation.append('soft_max')
    #   NumberOfNeuronInEachLayer.append(NumberOfNeuronInEachLayer[-1])
    numberOfLayers = len(NumberOfNeuronInEachLayer)
    NumberOfNeuronInAllLayers = [numberOfFeaturesInInputLayer] + NumberOfNeuronInEachLayer
    self.layers = [Layer(NumberOfNeuronInAllLayers[layer], NumberOfNeuronInAllLayers[layer+1], EachLayerActivation[layer]) for layer in range(numberOfLayers)]
    self.inputs = []
    self.label = []
    self.learningRate = 0.01
    self.regularization = False
    self.regularization_hyper = 0.001
    self.thresh_hold = 0.7
    self.cost_function = 'LL'
    self.batch_size = 1000

  # for the forward phase of backpropagation
  def __call__(self, data):
    for layer in self.layers:
      data = layer(data)
    return data
  
  def parameters(self):
    return [param for layer in self.layers for param in layer.parameters()]
  
  def prediction(self):
    return list(map(self, self.inputs))
  
  def sigmoid_accuracy(self):
    prediction = self.prediction()
    accuracy = [(y > self.thresh_hold) == (y_hat[0].data > self.thresh_hold) for y, y_hat in zip(self.label, prediction)]
    return sum(accuracy) / len(accuracy)
  
  def forwardPropagation(self):
    regularization_loss = 0.0
    data_loss = 0.0
    prediction = self.prediction()
    if self.cost_function == 'MSE':
      data_loss = [(y_hat[0] - y)**2 for y_hat, y in zip(prediction, self.label)]
      data_loss = sum(data_loss) * (1/len(data_loss))
    if self.cost_function == 'LL':
      data_loss = [( -(y * y_hat[0].log()) - ((1 - y) * (1 - y_hat[0]).log())) for y_hat, y in zip(prediction, self.label)]
      data_loss = sum(data_loss)
    if self.cost_function == 'CE':
      # print("\nLabel:")
      # print(self.label)
      for i in range(len(prediction)):
        for j in range(len(prediction[i])):
          # print("------------")
          # print("prediction: ")
          # print(prediction)
          # print("prediction [i]: ")
          # print(prediction[i])
          # print("prediction [i][j]: ")
          # print(prediction[i][j])
          # print("label: ")
          # print(self.label)
          # print("label [i]: ")
          # print(self.label[i])
          # print("label [i][j]: ")
          # print(self.label[i][j])
          # print("--------------")
          # print(math.log(prediction[i][j].data))
          # if self.label[i][j] == 0:
          #   continue
          data_loss += prediction[i][j].crossEntropy(self.label[i][j])
        # print(len(prediction[i]))
        # print(prediction[i])
        # data_loss += (-1 * self.label[i] * math.log(prediction[i]))
        # data_loss = [-1 * (y * math.log(y_hat)) for y, y_hat in zip(self.label, prediction[i])]
        # data_loss = sum(data_loss)
    if self.regularization:
      regularization_loss = self.regularization_hyper * sum((param * param for param in self.parameters()))
    total_loss = data_loss + regularization_loss
    accuracy = 0
    if self.layers[-1].activation_function == 'sigmoid':
      accuracy = self.sigmoid_accuracy()
    return total_loss, accuracy
  
  def backwardPropagation(self, loss):
    self.zero_derivitive()
    # print("\nloss")
    # print(loss)
    # print()
    loss.backward()
    # print(loss)
    # print("loss\n")

  def update(self):
    for param in self.parameters():
      # print("\nbefore:")
      # print(param)
      param.data -= self.learningRate * param.derivitive
      # print("after")
      # print(param)
      # print()

  def fit(self, data, label, costFunction='LL', learningRate=0.01, regularization=False, regularizationHyper=0.001, thresh_hold=0.7):
    self.inputs = [list(map(Node, row)) for row in data]
    self.label = label
    self.cost_function = costFunction
    self.learningRate = learningRate
    self.regularization = regularization
    self.regularization_hyper = regularizationHyper
    self.thresh_hold = thresh_hold
  
  def optimize(self, numberOfIterations):
    for iteration in range(numberOfIterations):
      loss, acc = self.forwardPropagation()
      self.backwardPropagation(loss)
      self.update()
      if iteration % 10 == 0:
        print(f"Iteration ({iteration}): loss: {loss}, accuray: {acc*100}%")
    return loss

  def __repr__(self):
    return f"MLP [{', '.join(str(layer) for layer in self.layers)}]"