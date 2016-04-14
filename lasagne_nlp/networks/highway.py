__author__ = 'max'

import theano.tensor as T
import numpy as np
from lasagne import init
from lasagne.layers import Layer
import lasagne.nonlinearities as nonlinearities

__all__ = [
    "HighwayDenseLayer",
]


class HighwayDenseLayer(Layer):
    """
    lasagne_nlp.networks.highway.HighwayDenseLayer(incoming, W_h=init.GlorotUniform(), b_h=init.Constant(0.),
        W_t=init.GlorotUniform(), b_t=init.Constant(0.), **kwargs)

    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape

    W_h : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_inputs)``.
        See :func:`lasagne.utils.create_param` for more information.

    b_h : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_inputs,)``.
        See :func:`lasagne.utils.create_param` for more information.

    W_t : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a matrix with shape ``(num_inputs, num_inputs)``.
        See :func:`lasagne.utils.create_param` for more information.

    b_t : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_inputs,)``.
        See :func:`lasagne.utils.create_param` for more information.

    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    """

    def __init__(self, incoming, W_h=init.GlorotUniform(), b_h=init.Constant(0.), W_t=init.GlorotUniform(),
                 b_t=init.Constant(0.), nonlinearity=nonlinearities.rectify, **kwargs):
        super(HighwayDenseLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W_h = self.add_param(W_h, (num_inputs, num_inputs), name="W_h")
        if b_h is None:
            self.b_h = None;
        else:
            self.b_h = self.add_param(b_h, (num_inputs,), name="b_h", regularizable=False)

        self.W_t = self.add_param(W_t, (num_inputs, num_inputs), name="W_t")
        if b_t is None:
            self.b_t = None;
        else:
            self.b_t = self.add_param(b_t, (num_inputs,), name="b_t", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        # if the input has more than two dimensions, flatten it into a
        # batch of feature vectors.
        input_reshape = input.flatten(2) if input.ndim > 2 else input

        activation = T.dot(input_reshape, self.W_h)
        if self.b_h is not None:
            activation = activation + self.b_h.dimshuffle('x', 0)
            activation = self.nonlinearity(activation)

        transform = T.dot(input_reshape, self.W_t)
        if self.b_t is not None:
            transform = transform + self.b_t.dimshuffle('x', 0)
            transform = nonlinearities.sigmoid(transform)

        carry = 1.0 - transform

        output = activation * transform + input_reshape * carry
        # reshape output back to orignal input_shape
        if input.ndim > 2:
            output = output.reshape(input.shape)

        return output
