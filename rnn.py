import time
import sys
import argparse

import numpy as np
import theano.tensor as T
import theano
import lasagne

import lasagne.nonlinearities as nonlinearities
from lasagne.layers import RecurrentLayer, Gate, LSTMLayer, GRULayer, DenseLayer

BATCH_SIZE = 128


def get_batch(batch_size, pos, binominal, length):
    if binominal:
        x = np.random.binomial(1, 0.5, [batch_size, length])
        x = 2 * x - 1
    else:
        x = np.random.uniform(-1.0, 1.0, [batch_size, length])
    y = np.zeros((batch_size,), dtype=np.int32)
    y[:] = (x[:, pos]) > 0.0
    return np.reshape(x, [batch_size, length, 1]).astype(np.float32), y


def construct_position_input(batch_size, length, num_units):
    index = T.zeros(shape=(batch_size, length), dtype='int32') + T.arange(length).dimshuffle('x', 0)
    index_layer = lasagne.layers.InputLayer(shape=(batch_size, length), input_var=index, name='position')
    embedding_layer = lasagne.layers.EmbeddingLayer(index_layer, input_size=length, output_size=num_units,
                                                    W=lasagne.init.Constant(0.), name='embedding')
    return embedding_layer


def train(layer_output, input_var, target_var, W, U, b, batch_size, length, position, binominal):
    predictions = lasagne.layers.get_output(layer_output)
    acc = lasagne.objectives.binary_accuracy(predictions, target_var)
    acc = acc.sum()

    loss = lasagne.objectives.binary_crossentropy(predictions, target_var)
    loss = loss.sum()

    learning_rate = 0.01
    steps_per_epoch = 1000
    params = lasagne.layers.get_all_params(layer_output, trainable=True)
    updates = lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate)
    # updates = lasagne.updates.adam(loss, params=params, learning_rate=learning_rate)
    train_fn = theano.function([input_var, target_var], [loss, acc, W, U, b, predictions], updates=updates)

    accuracies = np.zeros(1000)
    for epoch in range(1000):
        start_time = time.time()
        print 'Epoch %d (learning rate=%.4f)' % (epoch, learning_rate)
        loss = 0.0
        correct = 0.0
        num_back = 0
        for step in range(steps_per_epoch):
            x, y = get_batch(batch_size, position, binominal, length)
            err, corr, w, u, b, pred = train_fn(x, y)
            # print x
            # print y
            # print pred
            loss += err
            correct += corr
            num_inst = (step + 1) * batch_size
        #     # update log
        #     sys.stdout.write("\b" * num_back)
        #     log_info = 'inst: %d loss: %.4f, corr: %d, acc: %.2f%%, W: %.6f, U: %.6f, b: %.6f' % (
        #         num_inst, loss / num_inst, correct, correct * 100 / num_inst, w, u, b)
        #     sys.stdout.write(log_info)
        #     num_back = len(log_info)
        #     # raw_input()
        # # update training log after each epoch
        # sys.stdout.write("\b" * num_back)
        assert num_inst == batch_size * steps_per_epoch
        accuracies[epoch] = correct * 100 / num_inst
        print 'inst: %d loss: %.4f, corr: %d, acc: %.2f%%, time: %.2fs' % (
            num_inst, loss / num_inst, correct, correct * 100 / num_inst, time.time() - start_time)

    return accuracies[-100:].mean()

def exe_rnn(use_embedd, length, num_units, position, binominal):
    batch_size = BATCH_SIZE

    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    target_var = T.ivector(name='targets')

    layer_input = lasagne.layers.InputLayer(shape=(None, length, 1), input_var=input_var, name='input')
    if use_embedd:
        layer_position = construct_position_input(batch_size, length, num_units)
        layer_input = lasagne.layers.concat([layer_input, layer_position], axis=2)

    layer_rnn = RecurrentLayer(layer_input, num_units, nonlinearity=nonlinearities.tanh, only_return_final=True,
                               W_in_to_hid=lasagne.init.GlorotUniform(), W_hid_to_hid=lasagne.init.GlorotUniform(),
                               b=lasagne.init.Constant(0.), name='RNN')
    W = layer_rnn.W_hid_to_hid.sum()
    U = layer_rnn.W_in_to_hid.sum()
    b = layer_rnn.b.sum()

    layer_output = DenseLayer(layer_rnn, num_units=1, nonlinearity=nonlinearities.sigmoid, name='output')

    return train(layer_output, input_var, target_var, W, U, b, batch_size, length, position, binominal)


def exe_lstm(use_embedd, length, num_units, position, binominal):
    batch_size = BATCH_SIZE

    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    target_var = T.ivector(name='targets')

    layer_input = lasagne.layers.InputLayer(shape=(None, length, 1), input_var=input_var, name='input')
    if use_embedd:
        layer_position = construct_position_input(batch_size, length, num_units)
        layer_input = lasagne.layers.concat([layer_input, layer_position], axis=2)

    ingate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                  W_cell=lasagne.init.Uniform(range=0.1))

    outgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                   W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                      W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

    layer_lstm = LSTMLayer(layer_input, num_units, ingate=ingate, forgetgate=forgetgate, cell=cell, outgate=outgate,
                           peepholes=False, nonlinearity=nonlinearities.tanh, only_return_final=True, name='LSTM')

    W = layer_lstm.W_hid_to_cell.sum()
    U = layer_lstm.W_in_to_cell.sum()
    b = layer_lstm.b_cell.sum()

    layer_output = DenseLayer(layer_lstm, num_units=1, nonlinearity=nonlinearities.sigmoid, name='output')

    return train(layer_output, input_var, target_var, W, U, b, batch_size, length, position, binominal)

def exe_gru(use_embedd, length, num_units, position, binominal):
    batch_size = BATCH_SIZE

    input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
    target_var = T.ivector(name='targets')

    layer_input = lasagne.layers.InputLayer(shape=(None, length, 1), input_var=input_var, name='input')
    if use_embedd:
        layer_position = construct_position_input(batch_size, length, num_units)
        layer_input = lasagne.layers.concat([layer_input, layer_position], axis=2)

    resetgate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None)

    updategate = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None)

    hiden_update = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        b=lasagne.init.Constant(0.), nonlinearity=nonlinearities.tanh)

    layer_gru = GRULayer(layer_input, num_units, resetgate=resetgate, updategate=updategate, hidden_update=hiden_update,
                         only_return_final=True, name='GRU')

    W = layer_gru.W_hid_to_hidden_update.sum()
    U = layer_gru.W_in_to_hidden_update.sum()
    b = layer_gru.b_hidden_update.sum()

    layer_output = DenseLayer(layer_gru, num_units=1, nonlinearity=nonlinearities.sigmoid, name='output')

    return train(layer_output, input_var, target_var, W, U, b, batch_size, length, position, binominal)


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN')
    parser.add_argument('--architec', choices=['rnn', 'lstm', 'gru'], help='architecture of rnn', required=True)
    parser.add_argument('--num_units', type=int, default=2, help='Number of units')
    parser.add_argument('--use_embedd', action='store_true', help='If use positional embedding')
    parser.add_argument('--binominal', action='store_true', help='If the data are sampled from bi-nominal distribution.')
    args = parser.parse_args()

    architec = args.architec
    NUM_UNITS = args.num_units
    USE_EMBEDD = args.use_embedd
    BINOMINAL = args.binominal

    if architec == 'rnn':
        exe = exe_rnn
    elif architec == 'lstm':
        exe = exe_lstm
    elif architec == 'gru':
        exe = exe_gru
    else:
        pass

    filename = '%s.%s.dim=%d.embedd=%s' % ('binominal' if BINOMINAL else 'uniform', architec, NUM_UNITS, USE_EMBEDD)
    fp = open(filename, 'w')
    print 'data: %s' % ('binominal' if BINOMINAL else 'uniform')
    num_runs = 100
    for length in [5, 10, 20, 40, 50]:
        result = 0.
        position = 0
        print 'architecture: %s (dim=%d, length=%d, postion=%d, embedd=%s)' % (architec, NUM_UNITS, length, position, USE_EMBEDD)
        fp.write('length=%d, pos=%d:\n' % (length, position))
        fp.flush()
        for run in range(num_runs):
            acc = exe(USE_EMBEDD, length, NUM_UNITS, position, BINOMINAL)
            fp.write('%.2f, ' % acc)
            result = result + acc
            fp.flush()

        fp.write('%.2f\n\n' % (result / num_runs))
        fp.flush()

        position = (length - 1) / 2
        fp.write('length=%d, pos=%d:\n' % (length, position))
        fp.flush()
        for run in range(num_runs):
            acc = exe(USE_EMBEDD, length, NUM_UNITS, position, BINOMINAL)
            fp.write('%.2f, ' % acc)
            result = result + acc
            fp.flush()

        fp.write('%.2f\n\n' % (result / num_runs))
        fp.flush()

    fp.close()


if __name__ == '__main__':
    main()
