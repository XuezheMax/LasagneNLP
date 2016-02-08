__author__ = 'max'

import logging
import sys
import numpy as np
import lasagne
from gensim.models.word2vec import Word2Vec
import gzip
import theano.config.floatX as floatX


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


def load_word_embedding_dict(embedding, embedding_path, logger):
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim
    elif embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        with gzip.open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert(embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim
    elif embedding == 'senna':
        return None
    else:
        raise ValueError("embedding should choose from [word2vec, senna]")

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
        return lasagne.updates.momentum(loss, params=params, learning_rate=learning_rate, momentum=momentum)
    elif update_algo == 'nesterov':
        return lasagne.updates.nesterov_momentum(loss, params=params, learning_rate=learning_rate, momentum=momentum)
    else:
        raise ValueError('unkown update algorithm: %s' % update_algo)
