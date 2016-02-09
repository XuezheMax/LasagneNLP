__author__ = 'max'

import lasagne
import lasagne.nonlinearities as nonlinearities
from lasagne.layers import Gate


def build_BiRNN(incoming, num_units, mask=None, grad_clipping=0, nonlinearity=nonlinearities.tanh,
                precompute_input=True):
    # construct the forward and backward rnns. Now, Ws are initialized by He initializer with default arguments.
    # Need to try other initializers for specific tasks.
    rnn_forward = lasagne.layers.RecurrentLayer(incoming, num_units,
                                                mask_input=mask, grad_clipping=grad_clipping,
                                                nonlinearity=nonlinearity, precompute_input=precompute_input,
                                                W_in_to_hid=lasagne.init.HeUniform(),
                                                W_hid_to_hid=lasagne.init.HeUniform())
    rnn_backward = lasagne.layers.RecurrentLayer(incoming, num_units,
                                                 mask_input=mask, grad_clipping=grad_clipping,
                                                 nonlinearity=nonlinearity, precompute_input=precompute_input,
                                                 W_in_to_hid=lasagne.init.HeUniform(),
                                                 W_hid_to_hid=lasagne.init.HeUniform(), backwards=True)

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([rnn_forward, rnn_backward], axis=2, name="bi-rnn")

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat


def build_BiLSTM(incoming, num_units, mask=None, grad_clipping=0, precompute_input=True, peepholes=False):
    # construct the forward and backward rnns. Now, Ws are initialized by Glorot initializer with default arguments.
    # Need to try other initializers for specific tasks.
    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.GlorotUniform())
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.GlorotUniform())
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                            nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                            precompute_input=precompute_input,
                                            ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward)

    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.GlorotUniform())
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.GlorotUniform())
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, num_units, mask_input=mask, grad_clipping=grad_clipping,
                                             nonlinearity=nonlinearities.tanh, peepholes=peepholes,
                                             precompute_input=precompute_input, backwards=True,
                                             ingate=ingate_backward, outgate=outgate_backward,
                                             forgetgate=forgetgate_backward, cell=cell_backward)

    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")

    # the shape of BiRNN output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    return concat
