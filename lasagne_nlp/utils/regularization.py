__author__ = 'max'

import theano.tensor as T
import theano.tensor.nlinalg as nlinalg


def dima(x):
    return -T.log(nlinalg.Det()(T.dot(x.T, x)))