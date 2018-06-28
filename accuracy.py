import numpy as np
import tflearn
import tensorflow as tf
import sys
import os
from decimal import getcontext, Decimal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def to_bits(str):
  '''
  Converts a string s to an array of bits of the composing characters
  '''
  result = []
  for c in str:
      bits = bin(ord(c))[2:]
      bits = '00000000'[len(bits):] + bits
      result.extend([int(b) for b in bits])
  return(np.array([result]))

def build_neural_network():
  '''
  Creates a TFLearn DNN with the same configuration as the model was trained.
  The model loaded is the one saved in the model/ folder.
  '''
  tf.reset_default_graph()
  tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
  net = tflearn.input_data(shape=[None, 128])
  net = tflearn.fully_connected(net, 128, activation='sigmoid', weights_init=tnorm)
  net = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=tnorm)
  regressor = tflearn.regression(net, optimizer='sgd', learning_rate=2., loss='mean_square')
  model = tflearn.DNN(regressor)
  model.load('model/toca.tflearn')
  return model

def main():
  '''
  Builds the neural network and plays the solver.
  '''
  tf.reset_default_graph()
  tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
  net = tflearn.input_data(shape=[None, 128])
  net = tflearn.fully_connected(net, 128, activation='sigmoid', weights_init=tnorm)
  net = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=tnorm)
  regressor = tflearn.regression(net, optimizer='sgd', learning_rate=2., loss='mean_square')
  model = tflearn.DNN(regressor)
  model.load('model/toca.tflearn')
  
  correct = 0
  incorrect = 0
  total = 0
  with open('dataset/test/words-r.txt','r') as f:
    for line in f:
      s = line.split(";")
      word = s[0]
      o = int(s[1])
      if len(word) > 16:
        continue
      word = word.rstrip()
      unrounded = model.predict(to_bits(word.ljust(16, ' ')))[0][0]

      if (round(unrounded) == o):
        correct += 1        
      else:
        incorrect += 1

      total += 1

  print("Current accuracy: " + str(round(correct / total * 100, 2)) + "%")

if __name__ == "__main__":
    main()