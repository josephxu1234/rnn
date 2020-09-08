import numpy as np
import random
from rnn import RNN
from data import train_data, test_data
import random

#construct of vocabulary of words that exist in our data:
vocab = list(set([word for phrase in train_data.keys() for word in phrase.split(" ")]))
vocab_size = len(vocab)
"""
assign integer index to represent each word in vocab
need to represent each word with an index bc RNNs can't understand words
we have to give them numbers
"""
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

"""
xi input to RNN is a vector. can use one-hot encoding
we have 18 unique words in the vocabulary, so each xi will be 18 dimensional one-hot vector

create_inputs returns array of one-hot vectors that represent the words in the input text string
"""
def create_inputs(text):
  res = []
  for word in text.split(' '):
    vec = np.zeros((vocab_size, 1)) #start off as array of zeros
    vec[word_to_idx[word]] = 1
    res.append(vec)
  return res

def softmax(x):
  return np.exp(x) / sum(np.exp(x))

"""

rnn = RNN(vocab_size, 2)
inputs = create_inputs("i am very good")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)
"""

rnn = RNN(vocab_size, 2)

def processData(data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping text to True or False.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = create_inputs(x)
    target = int(y)

    # feedforward
    out, _ = rnn.feedforward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)

# Training loop
for epoch in range(300):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

inputs = create_inputs("i am sad")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)

inputs = create_inputs("i am happy")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)
"""
inputs = create_inputs("Tintin was happy earlier")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)

inputs = create_inputs("Tintin was not happy earlier")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)

inputs = create_inputs("Mickey Rooney and I have a very close relationship")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)

inputs = create_inputs("Mickey Rooney is incredibly stupid")
out, h = rnn.feedforward(inputs)
probs = softmax(out)
print(probs)
"""
