__author__ = 'max'

import logging
import sys
import numpy as np
import lasagne
from gensim.models.word2vec import Word2Vec
import gzip
import theano


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


def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimention, caseless
    """
    if embedding == 'word2vec':
        # loading word2vec
        logger.info("Loading word2vec ...")
        word2vec = Word2Vec.load_word2vec_format(embedding_path, binary=True)
        embedd_dim = word2vec.vector_size
        return word2vec, embedd_dim, False
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
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'senna':
        # loading Senna
        logger.info("Loading Senna ...")
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
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    elif embedding == 'random':
        # loading random embedding table
        logger.info("Loading Random ...")
        embedd_dict = dict()
        words = word_alphabet.get_content()
        scale = np.sqrt(3.0 / embedd_dim)
        for word in words:
            embedd_dict[word] = np.random.uniform(-scale, scale, [1, embedd_dim])
        return embedd_dict, embedd_dim, False
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


def iterate_minibatches(inputs, targets, masks=None, char_inputs=None, batch_size=10, shuffle=False):
    assert len(inputs) == len(targets)
    if masks is not None:
        assert len(inputs) == len(masks)
    if char_inputs is not None:
        assert len(inputs) == len(char_inputs)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs), batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt], (None if masks is None else masks[excerpt]), \
              (None if char_inputs is None else char_inputs[excerpt])


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
    elif update_algo == 'adadelta':
        return lasagne.updates.adadelta(loss, params=params)
    elif update_algo == 'adam':
        return lasagne.updates.adam(loss, params=params, learning_rate=learning_rate)
    else:
        raise ValueError('unkown update algorithm: %s' % update_algo)


def get_all_params_by_name(layer, name=None, **tags):
    # tags['trainable'] = tags.get('trainable', True)
    # tags['regularizable'] = tags.get('regularizable', True)
    params = lasagne.layers.get_all_params(layer, **tags)
    if name is None:
        return params
    else:
        name_set = set(name) if isinstance(name, list) else set([name, ])
        return [param for param in params if param.name in name_set]


def output_predictions(predictions, targets, masks, filename, label_alphabet, is_flattened=True):
    batch_size, max_length = targets.shape
    with open(filename, 'a') as file:
        for i in range(batch_size):
            for j in range(max_length):
                if masks[i, j] > 0.:
                    prediction = predictions[i * max_length + j] + 1 if is_flattened else predictions[i, j] + 1
                    file.write('_ %s %s\n' % (label_alphabet.get_instance(targets[i, j] + 1),
                                              label_alphabet.get_instance(prediction)))
            file.write('\n')

