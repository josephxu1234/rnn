import numpy as np
from numpy.random import randn


class RNN:
  def __init__(self, input_size, output_size, hidden_size=64):
    #Weights
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000

    #biases
    self.bh = np.zeros((hidden_size, 1))
    self.by = np.zeros((output_size, 1))

  def feedforward(self, inputs):
    """
    inputs is an array of one-hot encodings
    return the final output and hidden state
    """
    h = np.zeros((self.Whh.shape[0], 1))

    #cache the inputs and hidden state for backprop
    self.prev_inputs = inputs
    self.prev_hs = {0: h} #prev_hs = previous hidden state. 0 is first step, h is the hidden state at step 0

    #perform the passes of the RNN
    # @ symbol means matrix multiplication
    for idx, x_i in enumerate(inputs):
      h = np.tanh(self.Wxh @ x_i + self.Whh @ h + self.bh)
      self.prev_hs[idx + 1] = h #cache all hidden states in dictionary
    #get final output
    y = self.Why @ h + self.by

    return y, h

  def backprop(self, dL_dy, learn_rate = .02):
    """
    perform backwards pass of RNN
    dL_dy (dL/dy) has shape (output_size, 1)
    """
    n = len(self.prev_inputs)
    dL_dWhy = dL_dy@self.prev_hs[n].T #transpose to make dimesnions match??
    dL_dby = dL_dy



    #initialize gradients
    dL_dWxh = np.zeros(self.Wxh.shape)
    dL_dWhh = np.zeros(self.Whh.shape)
    dL_dbh = np.zeros(self.bh.shape)
    dy_dh = self.Why.T
    dL_dh = dy_dh @ dL_dy #calculate dL/dh for last h
    for t in reversed(range(n)):
      temp = (1 - self.prev_hs[t + 1] ** 2) * dL_dh #this value is calculated often. just cache it
      dL_dbh += temp
      dL_dWhh += temp @ self.prev_hs[t].T
      dL_dWxh += temp @ self.prev_inputs[t].T
      dL_dh = self.Whh @ temp
    for d in [dL_dWxh, dL_dWhh, dL_dWhy, dL_dbh, dL_dby]:
      np.clip(d, -1, 1, out=d) #prevent exploding gradient
    self.Whh -= learn_rate * dL_dWhh
    self.Wxh -= learn_rate * dL_dWxh
    self.Why -= learn_rate * dL_dWhy
    self.bh -= learn_rate * dL_dbh
    self.by -= learn_rate * dL_dby
