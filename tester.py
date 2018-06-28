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
  model = build_neural_network()

  print("----------------------------------------")
  print("          TOCA OU NÃO TOCA?             ")
  print("----------------------------------------")
  while True:
    line = input("> ")
    line = line.rstrip()

    if (len(line) < 3 or len(line) > 16):
      print("The word size must be between 3 and 16")
      continue

    unrounded = model.predict(to_bits(line.ljust(16, ' ')))[0][0]
    if (round(unrounded) == 1.0):
      print(line + " toca [" + str(np.round(unrounded, 3)) + "]")
    else:
      print(line + " não toca [" + str(np.round(unrounded, 3)) + "]")

if __name__ == "__main__":
    main()