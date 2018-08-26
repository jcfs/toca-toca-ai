from flask import Flask, request
from flask_restful import Resource, Api

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

model = build_neural_network()
app = Flask(__name__)
api = Api(app)

class Classifier(Resource):
  def get(self, word):
    unrounded = model.predict(to_bits(word.ljust(16, ' ')))[0][0]
    if (round(unrounded) == 1.0):
      return "toca"
    else:
      return "nao toca"

api.add_resource(Classifier, '/classify/<word>')

if __name__ == "__main__":
  app.run(port='5002')
