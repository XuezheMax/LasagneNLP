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


def iterate_minibatches(inputs, targets, types=None, masks=None, char_inputs=None, batch_size=10, shuffle=False):
    assert len(inputs) == len(targets)
    if types is not None:
        assert len(inputs) == len(types)
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
        yield inputs[excerpt], targets[excerpt], (None if types is None else types[excerpt]), \
              (None if masks is None else masks[excerpt]), (None if char_inputs is None else char_inputs[excerpt])


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


def decode_MST(energies, masks):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: Theano 4D tensor
        energies of each edge. the shape is [batch_size, n_steps, n_steps, num_labels],
        where the summy root is at index 0.
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_steps].
    :return:
    """

    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if added[i] or not curr_nodes[i]:
                continue

            # init cycle
            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i] = True
            findcycle = True
            l = i

            while not par[l] in tmp_cycle:
                l = par[l]
                if added[l]:
                    findcycle = False
                    break
                added[l] = True
                tmp_cycle.add(l)

            if findcycle:
                lorg = l
                cycle.add(lorg)
                l = par[lorg]
                while l != lorg:
                    cycle.add(l)
                    l = par[l]
                break

        return findcycle, cycle

    def chuLiuEdmonds():
        par = np.zeros([length], dtype=np.int32)
        # create best graph
        par[0] = -1
        for i in range(length):
            # only interested at current nodes
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue

                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)
        # no cycles, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue

                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr
            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        id = 0
        for cyc_node in cycle:
            cyc_nodes[id] = cyc_node
            id += 1
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1

            for j in range(cyc_len):
                j1 = cyc_nodes[j]
                if score_matrix[j1, i] > max1:
                    max1 = score_matrix[j1, i]
                    wh1 = j1

                scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]

                if scr > max2:
                    max2 = scr
                    wh2 = j1

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)

        for i in range(1, cyc_len):
            cyc_node = cyc_nodes[i]
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != par[wh]:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    input_shape = energies.shape
    batch_size = input_shape[0]
    max_length = input_shape[1]

    pars = np.zeros([batch_size, max_length], dtype=np.int32)
    types = np.zeros([batch_size, max_length], dtype=np.int32)
    for i in range(batch_size):
        energy = energies[i]
        mask = masks[i]

        # calc the realy length of this instance
        length = 0
        while length < max_length and mask[length] == 1:
            length += 1

        # calc real energy matrix shape = [length, length, num_labels - 1] (remove the label for root symbol).
        energy = energy[:length, :length, 1:]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=2)
        # get original score matrix
        orig_score_matrix = energy.max(axis=2)
        # initialize score matrix to original score matrix
        score_matrix = np.array(orig_score_matrix, copy=True)

        oldI = np.zeros([length, length], dtype=np.int32)
        oldO = np.zeros([length, length], dtype=np.int32)
        curr_nodes = np.zeros([length], dtype=np.bool)
        reps = []

        for s in range(length):
            orig_score_matrix[s, s] = 0.0
            score_matrix[s, s] = 0.0
            curr_nodes[s] = True
            reps.append(set())
            reps[s].add(s)
            for t in range(s + 1, length):
                oldI[s, t] = s
                oldO[s, t] = t

                oldI[t, s] = t
                oldO[t, s] = s

        final_edges = dict()
        chuLiuEdmonds()
        par = np.zeros([max_length], np.int32)
        type = np.zeros([max_length], np.int32)
        type[0] = 0

        for ch, pr in final_edges.items():
            par[ch] = pr
            if ch != 0:
                type[ch] = label_id_matrix[pr, ch]

        pars[i] = par
        types[i] = type

    return pars, types
