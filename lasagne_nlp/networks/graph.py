__author__ = 'max'

import theano.tensor as T

from lasagne.layers import MergeLayer
from lasagne import init
import lasagne.nonlinearities as nonlinearities
from theano.tensor.sort import argsort

__all__ = [
    "GraphConvLayer",
]


class GraphConvLayer(MergeLayer):
    """
    lasagne_nlp.networks.graph.GraphConvLayer(incoming_vertex, incoming_edge, num_filters, filter_size,
                 W=init.GlorotUniform(), b=init.Constant(0.), nonlinearity=nonlinearities.rectify, **kwargs)

    Parameters
    ----------
    incoming_vertex : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
        The output of this layer should be a 3D tensor with shape
        ``(batch_size, number_input_channels, number_vertex)``
    incoming_edge : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
        The output of this layer should be a 4D tensor with shape
        ``(batch_size, number_distance_metric, number_vertex, number_vertex)``
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or an `n`-element tuple specifying the size of the filters.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor with shape ``(number_distance_metric * filter_size * number_input_channels, num_filters)``,
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    """

    def __init__(self, incoming_vertex, incoming_edge, num_filters, filter_size, W=init.GlorotUniform(),
                 b=init.Constant(0.), nonlinearity=nonlinearities.rectify, **kwargs):
        self.vertex_shape = incoming_vertex.output_shape
        self.edge_shape = incoming_edge.output_shape

        self.input_shape = incoming_vertex.output_shape
        incomings = [incoming_vertex, incoming_edge]
        self.vertex_incoming_index = 0
        self.edge_incoming_index = 1
        super(GraphConvLayer, self).__init__(incomings, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_filters = num_filters
        self.filter_size = filter_size

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_filters,), name="b", regularizable=False)

    def get_W_shape(self):
        """Get the shape of the weight matrix `W`.

        Returns
        -------
        tuple of int
            The shape of the weight matrix.
        """
        num_input_channels = self.vertex_shape[1]
        num_dist_metrics = self.edge_shape[1]
        return num_dist_metrics * num_input_channels * self.filter_size, self.num_filters

    def get_output_shape_for(self, input_shapes):
        vertex_shape = input_shapes[self.vertex_incoming_index]
        return vertex_shape[0], self.num_filters, vertex_shape[2]

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        :param inputs: list of theano.TensorType
            `inputs[0]` should always be the symbolic vertex variable.
            `inputs[1]` should always be the symbolic edge variable.
        :return: theano.TensorType
            Symbolic output variable.
        """

        vertex = inputs[self.vertex_incoming_index]
        # shuffle vertex to shape [batch, n, channel]
        vertex = vertex.dimshuffle(0, 2, 1)
        # get each dimension
        vertex_shape = vertex.shape
        batch_size = vertex_shape[0]
        num_vertex = vertex_shape[1]
        num_channel = vertex_shape[2]
        num_dist_metrics = self.edge_shape[1]
        filter_size = self.filter_size
        num_filters = self.num_filters

        # vertex_conv shape [batch, n, n, channel]
        vertex_conv = T.cast(T.alloc(0.0, batch_size, num_vertex, num_vertex, num_channel), 'floatX')
        vertex_conv = vertex_conv + vertex.dimshuffle(0, 'x', 1, 2)
        # reshape vertex_conv to [batch * n, n, channel]
        vertex_conv = T.reshape(vertex_conv, (batch_size * num_vertex, num_vertex, num_channel))

        edge = inputs[self.edge_incoming_index]
        edge_sorted_indices = argsort(edge, axis=3)
        # take last filter_size indices. the shape of edge_sorted_indices is [batch, d, n, k]
        edge_sorted_indices = edge_sorted_indices[:, :, :, :filter_size]
        # shuffle indices to shape [batch, n, d, k]
        edge_sorted_indices = edge_sorted_indices.dimshuffle(0, 2, 1, 3)
        # reshape indices to shape [batch * n, d * k]
        edge_sorted_indices = T.reshape(edge_sorted_indices, (batch_size * num_vertex, num_dist_metrics * filter_size))

        # compute conv_tensor with shape [d * k, batch * n, channel]
        conv = vertex_conv[T.arange(batch_size * num_vertex), edge_sorted_indices.T, :]
        # shuffle conv to [batch * n, d * k, channel]
        conv = conv.dimshuffle(1, 0, 2)
        # reshape conv to [batch * n, d * k * channel]
        conv = T.reshape(conv, (batch_size * num_vertex, num_dist_metrics * filter_size * num_channel))

        # dot conv with W ([batch * n, d * k * channel] x [d * k * channel, num_filters] = [batch * n, num_filters]
        activation = T.dot(conv, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        # apply nonlinear function
        activation = self.nonlinearity(activation)
        # reshape activation back to [batch, n, num_filters]
        activation = T.reshape(activation, (batch_size, num_vertex, num_filters))
        # shuffle it to [batch, num_filters, n]
        return activation.dimshuffle(0, 2, 1)
