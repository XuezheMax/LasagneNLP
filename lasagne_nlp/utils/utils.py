__author__ = 'max'

import logging
import sys
import numpy as np
import lasagne


def get_logger(name, level=logging.INFO, handler=sys.stdout,
               formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches(inputs, targets, masks, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    assert len(inputs) == len(masks)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], masks[excerpt]


def create_updates(loss, params, update_algo, learning_rate, momentum=None):
    """
    create updates for training
    :param loss: loss for gradient
    :param params: parameters for update
    :param update_algo: update algorithm
    :param learning_rate: learning rate
    :param momentum: momentum
    :return: updates
    """

    if update_algo == 'sgd':
        return lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate)
    elif update_algo == 'momentum':
        return lasagne.updates.momentum(loss, params == params, learning_rate=learning_rate, momentum=momentum)
    elif update_algo == 'nesterov':
        return lasagne.updates.nesterov_momentum(loss, params=params, learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError('unkown update algorithm: %s' % update_algo)
