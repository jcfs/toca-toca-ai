# 'Toca ou não toca' solver using a Tensorflow DNN

## Description
Solver for the _well known_ portuguese 'Toca não toca' that uses a pretrained model for a Deep Neural Network (DNN). The solver makes use of google's Tensorflow and its higher abstraction layer TFLearn.

The riddle description is as follows (in Portuguese)
```
Este enigma tem como chave o próprio nome, toca não toca.
Por exemplo: lábios toca, toca não toca.
```

The purpose of the riddle is to find the citeria that makes a word 'tocar' or 'não tocar'. To do that we must keep asking different words to the riddle master who will simply answer "_word_ toca" or "_word_ não toca" (btw, 'word' não toca).

The algorithm to the solution of the problem is easy and deterministic, and could be implemented in a couple of lines of code with 100% accuracy (without using any AI). The fun of it is that the model trained and commited in the project does not have a 100% accuracy (but it is close to it), therefore it can make mistakes just like a normal person would - turning the riddle more interesting ;)

The model trainer used a train dataset composed of ~24k words (taken from the bible) and a linear implementation to classify the words (meaning that if you look at the trainer.py code the solution for the riddle will be there - do that at your own will).

## Example usage

To run the trainer to train a new model:
```
$ python trainer.py
```

This will train a model over 1000 epochs using the dataset/train/words-big.txt dataset.

To run the tester:
```
$ python tester.py
----------------------------------------
          TOCA OU NÃO TOCA?             
----------------------------------------
> lábios
lábios toca [1.0]
> toca
toca não toca [0.043]
> a
The word size must be between 3 and 16
> 
```

## Instalation

This implementation will need the following software

* Python 3 / pip
* Tensorflow (https://www.tensorflow.org/)
* TFLean (http://tflearn.org/)

```
git clone https://github.com/jcfs/toca-toca-ai.git
```

```
cd toca-toca-ai
pip install -r requirements.txt
```
