__author__ = 'max'

import theano.tensor as T

from lasagne.layers import MergeLayer
from lasagne import init

__all__ = [
    "DepParserLayer",
]


class DepParserLayer(MergeLayer):
    """

    """

    def __init__(self, incoming, num_labels, mask_input=None, W_h=init.GlorotUniform(), W_c=init.GlorotUniform(),
                 b=init.Constant(0.), **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.
        # We will just provide the layer input as incomings, unless a mask input was provided.
        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1

        super(DepParserLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels
        num_inputs = self.input_shape[2]

        # add parameters
        self.W_h = self.add_param(W_h, (num_inputs, self.num_labels), name='W_h')

        self.W_c = self.add_param(W_c, (num_inputs, self.num_labels), name='W_c')

        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels,), name='b', regularizable=False)

    def get_output_shape_for(self, input_shapes):
        """

        :param input_shapes:
        :return: the shape of output [batch_size, length, length, num_labels]
        """
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], input_shape[1], self.num_labels

    def get_output_for(self, inputs, **kwargs):
        """

        :param inputs: inputs: list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        :return: theano.TensorType
            Symbolic output variable.
        """
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        # compute head part by tensor dot ([batch, length, input] * [input, num_label]
        # the shape of s_h should be [batch, length, num_label]
        s_h = T.tensordot(input, self.W_h, axes=[[2], [0]])

        if self.b is not None:
            b_shuffled = self.b.dimshuffle('x', 'x', 0)
            s_h = s_h + b_shuffled

        # compute child part by tensor dot ([batch, length, input] * [input, num_label]
        # the shape of s_c should be [batch, length, num_label]
        s_c = T.tensordot(input, self.W_c, axes=[[2], [0]])

        # compute out
        input_shape = input.shape
        out = T.cast(T.alloc(0.0, input_shape[0], input_shape[1], input_shape[1], self.num_labels), 'floatX')
        out = out + s_h.dimshuffle(0, 1, 'x', 2)
        out = out + s_c.dimshuffle(0, 'x', 1, 2)

        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
            mask_shuffled = mask.dimshuffle(0, 'x', 1, 'x')
            out = out * mask_shuffled
        return out
