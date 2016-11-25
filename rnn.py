import time
import sys
import argparse

import numpy as np
import theano.tensor as T
import theano
import lasagne

import lasagne.nonlinearities as nonlinearities
from lasagne.layers import RecurrentLayer

LENGTH = 100
NUM_UNITS = 1


def get_batch(batch_size):
    x = np.random.binomial(1, 0.5, [batch_size, LENGTH])
    x = 2 * x - 1
    y = np.zeros((batch_size,), dtype=np.int32)
    y[:] = x[:, 0]
    return np.reshape(x, [batch_size, LENGTH, 1]).astype(np.float32), y


def main():
    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    target_var = T.ivector(name='targets')

    layer_input = lasagne.layers.InputLayer(shape=(None, LENGTH, 1), input_var=input_var, name='input')

    layer_rnn = RecurrentLayer(layer_input, NUM_UNITS, nonlinearity=nonlinearities.tanh, only_return_final=True,
                               W_in_to_hid=lasagne.init.Constant(1), W_hid_to_hid=lasagne.init.Constant(2), b=None,
                               name='RNN')
    W = layer_rnn.W_hid_to_hid
    U = layer_rnn.W_in_to_hid

    output = lasagne.layers.get_output(layer_rnn)
    output = output.mean(axis=1)
    prediction = T.switch(T.gt(output, 0), 1, -1)
    acc = T.eq(prediction, target_var)
    acc = acc.sum()
    # get the output before activation function tanh
    epsilon = 1e-6
    prob = 0.5 * T.log((1 + output + epsilon) / (1 - output + epsilon))
    prob = nonlinearities.sigmoid(prob)
    loss = -0.5 * ((1 + target_var) * T.log(prob) + (1 - target_var) * T.log(1 - prob))
    loss = loss.sum()

    batch_size = 100
    learning_rate = 0.01
    steps_per_epoch = 1000
    params = lasagne.layers.get_all_params(layer_rnn, trainable=True)
    updates = lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], [loss, acc, W, U, output], updates=updates)

    for epoch in range(10000):
        print 'Epoch %d (learning rate=%.4f)' % (epoch, learning_rate)
        loss = 0.0
        correct = 0.0
        num_back = 0
        for step in range(steps_per_epoch):
            x, y = get_batch(batch_size)
            err, corr, w, u, pred = train_fn(x, y)
            # print x
            # print y
            # print pred
            loss += err
            correct += corr
            num_inst = (step + 1) * batch_size
            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'inst: %d loss: %.4f, corr: %d, acc: %.2f%%, W: %.6f, U: %.6f' % (
                num_inst, loss / num_inst, correct, correct * 100 / num_inst, w.sum(), u.sum())
            sys.stdout.write(log_info)
            num_back = len(log_info)
            # raw_input()
        # update training log after each epoch
        sys.stdout.write("\b" * num_back)
        assert num_inst == batch_size * steps_per_epoch
        print 'inst: %d loss: %.4f, corr: %d, acc: %.2f%%' % (num_inst, loss / num_inst, correct, correct * 100 / num_inst)


if __name__ == '__main__':
    main()
