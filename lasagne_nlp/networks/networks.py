__author__ = 'max'

import lasagne
import lasagne.nonlinearities as nonlinearities


def build_BiRNN(incoming, num_units, mask=None, grad_clipping=0, nonlinearity=nonlinearities.tanh,
                precompute_input=True):
    # construct the forward and backward rnns. Now, Ws are initialized by Glorot initializer with default arguments.
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
