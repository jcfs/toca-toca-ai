import numpy as np
import tflearn
import tensorflow as tf
import random

def to_bits(str):
  '''
  Converts a string s to an array of bits of the composing characters
  '''
  result = []
  for c in str:
      bits = bin(ord(c))[2:]
      bits = '00000000'[len(bits):] + bits
      result.extend([int(b) for b in bits])
  return result

def main():
  '''
  Main training function that will read the words from the list
  convert them to binary arrays and feed that information to a DNN.
  '''
  training = []

  with open('dataset/train/words-big-r.txt','r') as f:
    for line in f:
      s = line.split(";")
      word = s[0]
      o = int(s[1])
      if len(word) <= 16:
        word = word.ljust(16)
        r = to_bits(word)
        training.append([r, [o]])

  # shuffle our features and turn into np.array as tensorflow takes in numpy array
  random.shuffle(training)
  training = np.array(training)

  b_words = list(training[:,0])
  classification = list(training[:,1])

  # reset underlying graph data
  tf.reset_default_graph()
  # Build neural network
  tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
  net = tflearn.input_data(shape=[None, len(b_words[0])])
  net = tflearn.fully_connected(net, 128, activation='sigmoid', weights_init=tnorm)
  net = tflearn.fully_connected(net, 1, activation='sigmoid', weights_init=tnorm)
  regressor = tflearn.regression(net, optimizer='sgd', learning_rate=2., loss='mean_square')

  model = tflearn.DNN(regressor)
  # train the model
  model.fit(b_words, classification, n_epoch=2000, show_metric=True)
  # save the model
  model.save('model/toca.tflearn')

if __name__ == "__main__":
    main()