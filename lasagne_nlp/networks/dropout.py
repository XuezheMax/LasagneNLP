__author__ = 'max'

import theano
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.layers import Layer

__all__ = [
    "GaussianDropoutLayer",
]


class GaussianDropoutLayer(Layer):
    """Gaussian Dropout layer

    Multiply values by gaussian variables with mean 1.0 and standard variance sigma. See notes for disabling dropout
    during testing.

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        the layer feeding into this layer, or the expected input shape
    sigma : float or scalar tensor
        The standard variance for gaussian distribution
    """

    def __init__(self, incoming, sigma=1.0, **kwargs):
        super(GaussianDropoutLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(get_rng().randint(1, 2147462579))
        self.sigma = sigma

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true dropout and scaling is disabled, see notes
        """
        if deterministic or self.sigma == 0:
            return input
        else:

            # use nonsymbolic shape for dropout mask if possible
            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.normal(input_shape, avg=1.0, std=self.sigma, dtype=theano.config.floatX)

gaussian_dropout = GaussianDropoutLayer # shortcut