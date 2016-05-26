__author__ = 'max'

import time
import sys
import argparse
import re
from collections import OrderedDict
from lasagne_nlp.utils import utils
import lasagne_nlp.utils.data_processor as data_processor
from lasagne_nlp.utils.objectives import parser_loss, crf_loss, crf_accuracy
from lasagne_nlp.networks.crf import CRFLayer
from lasagne_nlp.networks.parser import DepParserLayer
import lasagne
import theano
import theano.tensor as T
from lasagne_nlp.networks.networks import build_BiLSTM_CNN, build_BiGRU_CNN

import numpy as np


def is_uni_punctuation(word):
    match = re.match("^[^\w\s]+$]", word, flags=re.UNICODE)
    return match is not None


def is_punctuation(word, pos, punct_set=None):
    if punct_set is None:
        return is_uni_punctuation(word)
    else:
        return pos in punct_set


def eval_pos(inputs, poss, poss_pred, masks, filename, word_alphabet, pos_alphabet):
    batch_size, max_length = inputs.shape
    corr = 0.
    total = 0.
    with open(filename, 'a') as file:
        for i in range(batch_size):
            for j in range(1, max_length):
                if masks[i, j] > 0.:
                    word = word_alphabet.get_instance(inputs[i, j])
                    pos_pred = pos_alphabet.get_instance(poss_pred[i, j] + 1)
                    pos = pos_alphabet.get_instance(poss[i, j] + 1)
                    total += 1
                    corr += 1 if poss[i, j] == poss_pred[i, j] else 0
                    file.write(
                        '%d\t%s\t%s\t%s\n' % (j, word.encode('utf8'), pos_pred.encode('utf8'), pos.encode('utf8')))
            file.write('\n')
    return corr, total


def eval_parsing(inputs, poss, pars_pred, types_pred, heads, types, masks, filename, word_alphabet, pos_alphabet,
                 type_alphabet, punct_set=None):
    batch_size, max_length = inputs.shape
    ucorr = 0.
    lcorr = 0.
    total = 0.
    ucorr_nopunc = 0.
    lcorr_nopunc = 0.
    total_nopunc = 0.
    with open(filename, 'a') as file:
        for i in range(batch_size):
            for j in range(1, max_length):
                if masks[i, j] > 0.:
                    word = word_alphabet.get_instance(inputs[i, j])
                    pos = pos_alphabet.get_instance(poss[i, j] + 1)
                    type = type_alphabet.get_instance(types_pred[i, j] + 1)
                    total += 1
                    ucorr += 1 if heads[i, j] == pars_pred[i, j] else 0
                    lcorr += 1 if heads[i, j] == pars_pred[i, j] and types[i, j] == types_pred[i, j] else 0

                    if not is_punctuation(word, pos, punct_set):
                        total_nopunc += 1
                        ucorr_nopunc += 1 if heads[i, j] == pars_pred[i, j] else 0
                        lcorr_nopunc += 1 if heads[i, j] == pars_pred[i, j] and types[i, j] == types_pred[i, j] else 0

                    file.write('%d\t%s\t_\t_\t%s\t_\t%d\t%s\n' % (
                        j, word.encode('utf8'), pos.encode('utf8'), pars_pred[i, j], type.encode('utf8')))
            file.write('\n')
    return ucorr, lcorr, total, ucorr_nopunc, lcorr_nopunc, total_nopunc


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
            if findcycle:
                break

            if added[i] or not curr_nodes[i]:
                continue

            # init cycle
            tmp_cycle = set()
            tmp_cycle.add(i)
            added[i] = True
            findcycle = True
            l = i

            while par[l] not in tmp_cycle:
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
        for i in range(1, length):
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
        while l != wh:
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
        while length < max_length and mask[length] > 0.5:
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
        type = np.ones([max_length], np.int32)
        type[0] = 0

        for ch, pr in final_edges.items():
            par[ch] = pr
            if ch != 0:
                type[ch] = label_id_matrix[pr, ch] + 1

        pars[i] = par
        types[i] = type

    return pars, types


def iterate_minibatches(inputs, pos=None, heads=None, types=None, masks=None, char_inputs=None, batch_size=10,
                        shuffle=False):
    if pos is not None:
        assert len(inputs) == len(pos)
    if heads is not None:
        assert len(inputs) == len(heads)
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
        yield inputs[excerpt], (None if pos is None else pos[excerpt]), \
              (None if heads is None else heads[excerpt]), \
              (None if types is None else types[excerpt]), \
              (None if masks is None else masks[excerpt]), \
              (None if char_inputs is None else char_inputs[excerpt])


def build_network(mode, input_var, char_input_var, mask_var,
                  max_length, max_char_length, alphabet_size, char_alphabet_size,
                  embedd_table, embedd_dim, char_embedd_table, char_embedd_dim,
                  num_units, num_filters, grad_clipping, peepholes, dropout, num_pos, num_types,
                  rnn, in_to_out, logger):
    def construct_input_layer():
        layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
        layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=alphabet_size,
                                                        output_size=embedd_dim,
                                                        W=embedd_table, name='embedding')
        return layer_embedding

    def construct_char_input_layer():
        layer_char_input = lasagne.layers.InputLayer(shape=(None, max_length, max_char_length),
                                                     input_var=char_input_var, name='char-input')
        layer_char_input = lasagne.layers.reshape(layer_char_input, (-1, [2]))
        layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char_input, input_size=char_alphabet_size,
                                                             output_size=char_embedd_dim, W=char_embedd_table,
                                                             name='char_embedding')
        layer_char_input = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))
        return layer_char_input

    def build_network_for_pos():
        return CRFLayer(layer_bottom, num_pos, mask_input=layer_mask, name='crf'), None, layer_bottom

    def build_network_for_parsing():
        return None, DepParserLayer(layer_bottom, num_types, mask_input=layer_mask, name='parser'), layer_bottom

    def build_network_for_both():
        layer_crf = CRFLayer(layer_bottom, num_pos, mask_input=layer_mask, name='crf')
        layer_parser = DepParserLayer(layer_bottom, num_types, mask_input=layer_mask, name='parser')
        return layer_crf, layer_parser, layer_bottom

    # construct input and mask layers
    layer_incoming1 = construct_char_input_layer()
    layer_incoming2 = construct_input_layer()

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=mask_var, name='mask')

    if rnn == 'LSTM':
        layer_bottom = build_BiLSTM_CNN(layer_incoming1, layer_incoming2, num_units, mask=layer_mask,
                                        grad_clipping=grad_clipping, peepholes=peepholes, num_filters=num_filters,
                                        dropout=dropout, in_to_out=in_to_out)
    elif rnn == 'GRU':
        layer_bottom = build_BiGRU_CNN(layer_incoming1, layer_incoming2, num_units, mask=layer_mask,
                                       grad_clipping=grad_clipping, num_filters=num_filters,
                                       dropout=dropout, in_to_out=in_to_out)
    else:
        raise ValueError('unknown rnn type: %s' % rnn)

    if mode == 'pos':
        return build_network_for_pos()
    elif mode == 'parse':
        return build_network_for_parsing()
    elif mode == 'both':
        return build_network_for_both()
    else:
        raise ValueError('unknown mode: %s' % mode)


def create_updates(loss, layer_top, layer_bottom, learning_rate_top, learning_rate_bottom, momentum, grad_clipping,
                   max_norm, update_algo):
    # get all the parameters
    params = lasagne.layers.get_all_params(layer_top, trainable=True)
    if update_algo == 'adadelta':
        updates = utils.create_updates(loss, params, update_algo, learning_rate_top, momentum=momentum)
        if max_norm:
            params_constraint = utils.get_all_params_by_name(layer_top,
                                                             name=['cnn.W', 'crf.W', 'parser.W_h', 'parser.W_c'],
                                                             trainable=True)
            for param in params_constraint:
                assert param in updates
                updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=max_norm)
        return updates

    # get parameters from bottom layers
    params_bottom = lasagne.layers.get_all_params(layer_bottom, trainable=True)

    # compute gradients
    grads = lasagne.updates.get_or_compute_grads(loss, params)
    # clip gradients
    if grad_clipping:
        clipped_grads = lasagne.updates.total_norm_constraint(grads, grad_clipping)
    else:
        clipped_grads = grads

    # create updates
    updates = OrderedDict()
    for param, grad in zip(params, clipped_grads):
        learning_rate = learning_rate_bottom if param in params_bottom else learning_rate_top
        updates[param] = param - learning_rate * grad

    # apply momentum term
    if update_algo == 'momentum':
        updates = lasagne.updates.apply_momentum(updates, momentum=momentum)
    elif update_algo == 'nesterov':
        updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=momentum)
    else:
        if update_algo != 'sgd':
            raise ValueError('unkown update algorithm: %s' % update_algo)

    if max_norm:
        params_constraint = utils.get_all_params_by_name(layer_top, name=['cnn.W', 'crf.W', 'parser.W_h', 'parser.W_c'],
                                                         trainable=True)
        for param in params_constraint:
            assert param in updates
            updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=max_norm)

    return updates


def perform_pos(layer_crf, layer_bottom, input_var, char_input_var, pos_var, mask_var, X_train, POS_train, mask_train,
                X_dev, POS_dev, mask_dev, X_test, POS_test, mask_test, C_train, C_dev, C_test,
                num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                pos_alphabet, tmp_dir, logger):
    logger.info('Performing mode: pos')
    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train = lasagne.layers.get_output(layer_crf)
    energies_eval = lasagne.layers.get_output(layer_crf, deterministic=True)

    loss_train = crf_loss(energies_train, pos_var, mask_var).mean()
    loss_eval = crf_loss(energies_eval, pos_var, mask_var).mean()
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(layer_crf, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    _, corr_train = crf_accuracy(energies_train, pos_var)
    corr_train = (corr_train * mask_var).sum(dtype=theano.config.floatX)
    prediction_eval, _ = crf_accuracy(energies_eval, pos_var)

    learning_rate_top = 1.0 if update_algo == 'adadelta' else learning_rate_top
    learning_rate_bottom = 1.0 if update_algo == 'adadelta' else learning_rate_bottom
    updates = create_updates(loss_train, layer_crf, layer_bottom, learning_rate_top, learning_rate_bottom, momentum,
                             grad_clipping, max_norm, update_algo)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, pos_var, mask_var, char_input_var], [loss_train, corr_train, num_tokens],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, pos_var, mask_var, char_input_var],
                              [loss_eval, num_tokens, prediction_eval])

    # Finally, launch the training loop.
    logger.info("Start training: %s with regularization: %s(%f), (#training data: %d, batch size: %d)..." \
                % (update_algo, regular, (0.0 if regular == 'none' else gamma), num_data, batch_size))
    num_batches = num_data / batch_size
    num_epochs = 1000
    best_loss = 1e+12
    best_acc = 0.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    best_loss_test_err = 0.
    best_loss_test_corr = 0.
    best_acc_test_err = 0.
    best_acc_test_corr = 0.
    stop_count = 0
    lr_top = learning_rate_top
    lr_bottom = learning_rate_bottom
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=(%.5f, %.5f), decay rate=(%.4f, %.4f), momentum=%.4f): ' % (
            epoch, lr_bottom, lr_top, decay_rate_bottom, decay_rate_top, momentum)
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, pos=POS_train, masks=mask_train, char_inputs=C_train,
                                         batch_size=batch_size, shuffle=True):
            inputs, pos, _, _, masks, char_inputs = batch
            err, corr, num = train_fn(inputs, pos, masks, char_inputs)
            train_err += err * inputs.shape[0]
            train_corr += corr
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data,
                train_err / train_inst, train_corr * 100 / train_total, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_data
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err / num_data, train_corr * 100 / train_total, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        dev_inst = 0
        for batch in iterate_minibatches(X_dev, pos=POS_dev, masks=mask_dev, char_inputs=C_dev,
                                         batch_size=batch_size):
            inputs, pos, _, _, masks, char_inputs = batch
            err, num, predictions = eval_fn(inputs, pos, masks, char_inputs)
            dev_err += err * inputs.shape[0]
            corr, total = eval_pos(inputs, pos, predictions, masks, tmp_dir + '/dev_pos%d' % epoch, word_alphabet,
                                   pos_alphabet)
            dev_corr += corr
            dev_total += total
            dev_inst += inputs.shape[0]

        print 'dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            dev_err / dev_inst, dev_corr, dev_total, dev_corr * 100 / dev_total)

        if best_loss < dev_err and best_acc > dev_corr / dev_total:
            stop_count += 1
        else:
            update_loss = False
            update_acc = False
            stop_count = 0
            if best_loss > dev_err:
                update_loss = True
                best_loss = dev_err
                best_epoch_loss = epoch
            if best_acc < dev_corr / dev_total:
                update_acc = True
                best_acc = dev_corr / dev_total
                best_epoch_acc = epoch

            # evaluate on test data when better performance detected
            test_err = 0.0
            test_corr = 0.0
            test_total = 0
            test_inst = 0
            for batch in iterate_minibatches(X_test, pos=POS_test, masks=mask_test, char_inputs=C_test,
                                             batch_size=batch_size):
                inputs, pos, _, _, masks, char_inputs = batch
                err, num, predictions = eval_fn(inputs, pos, masks, char_inputs)
                test_err += err * inputs.shape[0]
                corr, total = eval_pos(inputs, pos, predictions, masks, tmp_dir + '/test_pos%d' % epoch, word_alphabet,
                                       pos_alphabet)
                test_corr += corr
                test_total += total
                test_inst += inputs.shape[0]

            print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
                test_err / test_inst, test_corr, test_total, test_corr * 100 / test_total)

            if update_loss:
                best_loss_test_err = test_err
                best_loss_test_corr = test_corr
            if update_acc:
                best_acc_test_err = test_err
                best_acc_test_corr = test_corr

        # stop if dev acc decrease 3 time straightly.
        if stop_count == patience:
            break

        # re-compile a function with new learning rate for training
        if update_algo != 'adadelta':
            lr_top = learning_rate_top / (1.0 + epoch * decay_rate_top)
            lr_bottom = learning_rate_bottom / (1.0 + epoch * decay_rate_bottom)
            updates = create_updates(loss_train, layer_crf, layer_bottom, lr_top, lr_bottom, momentum, grad_clipping,
                                     max_norm, update_algo)
            train_fn = theano.function([input_var, pos_var, mask_var, char_input_var],
                                       [loss_train, corr_train, num_tokens],
                                       updates=updates)

    # print best performance on test data.
    logger.info("final best loss test performance (at epoch %d)" % best_epoch_loss)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_loss_test_err / test_inst, best_loss_test_corr, test_total, best_loss_test_corr * 100 / test_total)
    logger.info("final best acc test performance (at epoch %d)" % best_epoch_acc)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_acc_test_err / test_inst, best_acc_test_corr, test_total, best_acc_test_corr * 100 / test_total)


def perform_parse(layer_parser, layer_bottom, input_var, char_input_var, head_var, type_var, mask_var,
                  X_train, POS_train, Head_train, Type_train, mask_train, X_dev, POS_dev, Head_dev, Type_dev, mask_dev,
                  X_test, POS_test, Head_test, Type_test, mask_test, C_train, C_dev, C_test,
                  num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                  decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                  pos_alphabet, type_alphabet, tmp_dir, punct_set, logger):
    logger.info('Performing mode: parse')
    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get output of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train = lasagne.layers.get_output(layer_parser)
    energies_eval = lasagne.layers.get_output(layer_parser, deterministic=True)

    loss_train = parser_loss(energies_train, head_var, type_var, mask_var).mean()
    loss_eval = parser_loss(energies_eval, head_var, type_var, mask_var).mean()
    # loss_train, _, _, _, _, _ = parser_loss(energies_train, head_var, type_var, mask_var)
    # loss_eval, E_eval, D_eval, L_eval, partition_eval, target_energy_eval = parser_loss(energies_eval, head_var,
    #                                                                                     type_var, mask_var)
    loss_train = loss_train.mean()
    loss_eval = loss_eval.mean()

    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(layer_parser, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    learning_rate_top = 1.0 if update_algo == 'adadelta' else learning_rate_top
    learning_rate_bottom = 1.0 if update_algo == 'adadelta' else learning_rate_bottom
    updates = create_updates(loss_train, layer_parser, layer_bottom, learning_rate_top, learning_rate_bottom, momentum,
                             grad_clipping, max_norm, update_algo)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, head_var, type_var, mask_var, char_input_var],
                               [loss_train, num_tokens], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, head_var, type_var, mask_var, char_input_var],
                              [loss_eval, num_tokens, energies_eval])

    # Finally, launch the training loop.
    logger.info("Start training: %s with regularization: %s(%f), (#training data: %d, batch size: %d)..." \
                % (update_algo, regular, (0.0 if regular == 'none' else gamma), num_data, batch_size))
    num_batches = num_data / batch_size
    num_epochs = 1000

    best_loss = 1e+12
    best_uas = 0.0
    best_las = 0.0
    best_uas_nopunt = 0.0
    best_las_nopunt = 0.0

    best_epoch_loss = 0
    best_epoch_uas = 0
    best_epoch_las = 0
    best_epoch_uas_nopunc = 0
    best_epoch_las_nopunc = 0

    best_loss_test_err = 0.
    best_loss_test_ucorr = 0.
    best_loss_test_lcorr = 0.
    best_loss_test_ucorr_nopunc = 0.
    best_loss_test_lcorr_nopunc = 0.

    best_uas_test_err = 0.
    best_uas_test_ucorr = 0.
    best_uas_test_lcorr = 0.
    best_uas_test_ucorr_nopunc = 0.
    best_uas_test_lcorr_nopunc = 0.

    best_las_test_err = 0.
    best_las_test_ucorr = 0.
    best_las_test_lcorr = 0.
    best_las_test_ucorr_nopunc = 0.
    best_las_test_lcorr_nopunc = 0.

    best_uas_nopunt_test_err = 0.
    best_uas_nopunt_test_ucorr = 0.
    best_uas_nopunt_test_lcorr = 0.
    best_uas_nopunt_test_ucorr_nopunc = 0.
    best_uas_nopunt_test_lcorr_nopunc = 0.

    best_las_nopunt_test_err = 0.
    best_las_nopunt_test_ucorr = 0.
    best_las_nopunt_test_lcorr = 0.
    best_las_nopunt_test_ucorr_nopunc = 0.
    best_las_nopunt_test_lcorr_nopunc = 0.

    stop_count = 0
    lr_top = learning_rate_top
    lr_bottom = learning_rate_bottom

    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=(%.5f, %.5f), decay rate=(%.4f, %.4f), momentum=%.4f): ' % (
            epoch, lr_bottom, lr_top, decay_rate_bottom, decay_rate_top, momentum)
        train_err = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, pos=POS_train, heads=Head_train, types=Type_train, masks=mask_train,
                                         char_inputs=C_train, batch_size=batch_size, shuffle=True):
            inputs, _, heads, types, masks, char_inputs = batch
            err, num = train_fn(inputs, heads, types, masks, char_inputs)
            train_err += err * inputs.shape[0]
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data, train_err / train_inst, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_data
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, time: %.2fs' % (min(train_batches * batch_size, num_data), num_data,
                                                        train_err / num_data, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_ucorr = 0.0
        dev_lcorr = 0.0
        dev_ucorr_nopunc = 0.0
        dev_lcorr_nopunc = 0.0
        dev_total = 0
        dev_total_nopunc = 0
        dev_inst = 0

        for batch in iterate_minibatches(X_dev, pos=POS_dev, heads=Head_dev, types=Type_dev, masks=mask_dev,
                                         char_inputs=C_dev, batch_size=batch_size):
            inputs, poss, heads, types, masks, char_inputs = batch
            err, num, energies = eval_fn(inputs, heads, types, masks, char_inputs)
            dev_err += err * inputs.shape[0]
            pars_pred, types_pred = decode_MST(energies, masks)
            ucorr, lcorr, total, ucorr_nopunc, \
            lcorr_nopunc, total_nopunc = eval_parsing(inputs, poss, pars_pred, types_pred, heads, types, masks,
                                                      tmp_dir + '/dev_parse%d' % epoch, word_alphabet,
                                                      pos_alphabet, type_alphabet, punct_set=punct_set)
            dev_inst += inputs.shape[0]

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total += total

            dev_ucorr_nopunc += ucorr_nopunc
            dev_lcorr_nopunc += lcorr_nopunc
            dev_total_nopunc += total_nopunc

            # np.set_printoptions(precision=5, suppress=True, linewidth=np.nan, threshold=np.nan)
            # length = masks[0].sum()
            # # print 'E:'
            # # print E[0, :length, :length]
            # # print 'D:'
            # # print D[0, :length, :length]
            # # print 'L:'
            # # print L[0, :length, :length]
            # print 'Z= %.5f, score= %.5f' % (Z[0], score[0])
            # energy = energies[0]
            # energy_max = energy[:length, :length, 1:].max(axis=2)
            # weight_pred1 = 0.0
            # weight_pred2 = 0.0
            # weight_gold = 0.0
            # for ch in range(1, length):
            #     weight_pred1 += energy_max[pars_pred[0, ch], ch]
            #     weight_pred2 += energy[pars_pred[0, ch], ch, types_pred[0, ch]]
            #     weight_gold += energy[heads[0, ch], ch, types[0, ch]]
            # print length
            # # print energy_max
            # print pars_pred[0, :length], weight_pred1, weight_pred2
            # print heads[0, :length], weight_gold, weight_pred1 - weight_gold
            # # raw_input()

        print 'dev loss: %.4f' % (dev_err / dev_inst)
        print 'Wi Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
            dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total)
        print 'Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
            dev_lcorr_nopunc * 100 / dev_total_nopunc)

        # re-compile a function with new learning rate for training
        if update_algo != 'adadelta':
            lr_top = learning_rate_top / (1.0 + epoch * decay_rate_top)
            lr_bottom = learning_rate_bottom / (1.0 + epoch * decay_rate_bottom)
            updates = create_updates(loss_train, layer_parser, layer_bottom, lr_top, lr_bottom, momentum, grad_clipping,
                                     max_norm, update_algo)
            train_fn = theano.function([input_var, head_var, type_var, mask_var, char_input_var],
                                       [loss_train, num_tokens], updates=updates)


def perform_both(layer_crf, layer_parser, layer_bottom, input_var, char_input_var, pos_var, head_var, type_var, mask_var,
                 X_train, POS_train, Head_train, Type_train, mask_train, X_dev, POS_dev, Head_dev, Type_dev, mask_dev,
                 X_test, POS_test, Head_test, Type_test, mask_test, C_train, C_dev, C_test,
                 num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                 decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                 pos_alphabet, type_alphabet, tmp_dir, punct_set, eta, logger):
    logger.info('Performing mode: both')
    # compute loss
    num_tokens = mask_var.sum(dtype=theano.config.floatX)

    # get outpout of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train_crf = lasagne.layers.get_output(layer_crf)
    energies_eval_crf = lasagne.layers.get_output(layer_crf, deterministic=True)

    loss_train_crf = crf_loss(energies_train_crf, pos_var, mask_var).mean()
    loss_eval_crf = crf_loss(energies_eval_crf, pos_var, mask_var).mean()

    pos_pred_eval, _ = crf_accuracy(energies_eval_crf, pos_var)

    # get output of bi-lstm-cnn-crf shape [batch, length, num_labels, num_labels]
    energies_train_parser = lasagne.layers.get_output(layer_parser)
    energies_eval_parser = lasagne.layers.get_output(layer_parser, deterministic=True)

    loss_train_parser = parser_loss(energies_train_parser, head_var, type_var, mask_var).mean()
    loss_eval_parser = parser_loss(energies_eval_parser, head_var, type_var, mask_var).mean()

    loss_train = loss_eval_crf * eta + loss_train_parser
    loss_eval = loss_eval_crf * eta + loss_eval_parser

    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(layer_parser, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    layer_merge = lasagne.layers.MergeLayer([layer_crf, layer_parser], name='merge')
    updates = create_updates(loss_train, layer_merge, layer_bottom, learning_rate_top, learning_rate_bottom, momentum,
                             grad_clipping, max_norm, update_algo)
    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, pos_var, head_var, type_var, mask_var, char_input_var],
                               [loss_train, loss_train_crf, loss_train_parser, num_tokens], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, pos_var, head_var, type_var, mask_var, char_input_var],
                              [loss_eval, loss_eval_crf, loss_eval_parser, num_tokens, pos_pred_eval,
                               energies_eval_parser])

    # Finally, launch the training loop.
    logger.info("Start training: %s with regularization: %s(%f), (#training data: %d, batch size: %d, eta: %.2f)..." \
                % (update_algo, regular, (0.0 if regular == 'none' else gamma), num_data, batch_size, eta))
    num_batches = num_data / batch_size
    num_epochs = 1000

    lr_top = learning_rate_top
    lr_bottom = learning_rate_bottom

    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=(%.5f, %.5f), decay rate=(%.4f, %.4f), momentum=%.4f): ' % (
            epoch, lr_bottom, lr_top, decay_rate_bottom, decay_rate_top, momentum)
        train_err = 0.0
        train_err_crf = 0.0;
        train_err_parser = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, pos=POS_train, heads=Head_train, types=Type_train, masks=mask_train,
                                         char_inputs=C_train, batch_size=batch_size, shuffle=True):
            inputs, poss, heads, types, masks, char_inputs = batch
            err, err_crf, err_parser, num = train_fn(inputs, poss, heads, types, masks, char_inputs)
            train_err += err * inputs.shape[0]
            train_err_crf += err_crf * inputs.shape[0]
            train_err_parser += err_parser * inputs.shape[0]
            train_total += num
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, crf loss: %.4f, parse loss: %.4f time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data, train_err / train_inst, train_err_crf / train_inst,
                train_err_parser / train_inst, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_data
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, crf loss: %.4f, parse loss: %.4f, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data, train_err / num_data, train_err_crf / num_data,
            train_err_parser / num_data, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_err_crf = 0.0
        dev_err_parser = 0.0
        dev_pcorr = 0.0
        dev_ucorr = 0.0
        dev_lcorr = 0.0
        dev_ucorr_nopunc = 0.0
        dev_lcorr_nopunc = 0.0
        dev_total = 0
        dev_total_nopunc = 0
        dev_inst = 0

        for batch in iterate_minibatches(X_dev, pos=POS_dev, heads=Head_dev, types=Type_dev, masks=mask_dev,
                                         char_inputs=C_dev, batch_size=batch_size):
            inputs, poss, heads, types, masks, char_inputs = batch
            err, err_crf, err_parser, num, pos_pred, energies_parser = eval_fn(inputs, poss, heads, types,
                                                                               masks, char_inputs)
            dev_err += err * inputs.shape[0]
            dev_err_crf += err_crf * inputs.shape[0]
            dev_err_parser += err_parser * inputs.shape[0]
            pars_pred, types_pred = decode_MST(energies_parser, masks)
            pcorr, total = eval_pos(inputs, poss, pos_pred, masks, tmp_dir + '/dev_pos%d' % epoch, word_alphabet,
                                    pos_alphabet)
            ucorr, lcorr, total, ucorr_nopunc, \
            lcorr_nopunc, total_nopunc = eval_parsing(inputs, poss, pars_pred, types_pred, heads, types, masks,
                                                      tmp_dir + '/dev_parse%d' % epoch, word_alphabet, pos_alphabet,
                                                      type_alphabet, punct_set=punct_set)
            dev_inst += inputs.shape[0]

            dev_pcorr += pcorr

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total += total

            dev_ucorr_nopunc += ucorr_nopunc
            dev_lcorr_nopunc += lcorr_nopunc
            dev_total_nopunc += total_nopunc

        print 'dev loss: %.4f, crf loss: %.4f, parse loss: %.4f' % (
            dev_err / dev_inst, dev_err_crf / dev_inst, dev_err_parser / dev_inst)

        print 'POS: corr: %d, total: %d, acc: %.2f' % (dev_pcorr, dev_total, dev_pcorr * 100 / dev_total)

        print 'Wi Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
            dev_ucorr, dev_lcorr, dev_total, dev_ucorr * 100 / dev_total, dev_lcorr * 100 / dev_total)

        print 'Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%' % (
            dev_ucorr_nopunc, dev_lcorr_nopunc, dev_total_nopunc, dev_ucorr_nopunc * 100 / dev_total_nopunc,
            dev_lcorr_nopunc * 100 / dev_total_nopunc)

        # re-compile a function with new learning rate for training
        if update_algo != 'adadelta':
            lr_top = learning_rate_top / (1.0 + epoch * decay_rate_top)
            lr_bottom = learning_rate_bottom / (1.0 + epoch * decay_rate_bottom)
            updates = create_updates(loss_train, layer_merge, layer_bottom, lr_top, lr_bottom, momentum, grad_clipping,
                                     max_norm, update_algo)
            train_fn = theano.function([input_var, pos_var, head_var, type_var, mask_var, char_input_var],
                                       [loss_train, loss_train_crf, loss_train_parser, num_tokens], updates=updates)


def main():
    parser = argparse.ArgumentParser(description='Tuning with neural MST parser')
    parser.add_argument('--mode', choices=['pos', 'parse', 'both'], help='mode for tasks', required=True)
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna', 'random'], help='Embedding for words',
                        required=True)
    parser.add_argument('--rnn', choices=['LSTM', 'GRU'], help='RNN architecture', default='GRU')
    parser.add_argument('--embedding_dict', default=None, help='path for embedding dict')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=200, help='Number of hidden units in LSTM')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--learning_rate_bottom', type=float, default=0.01, help='Learning rate for bottom layers')
    parser.add_argument('--learning_rate_top', type=float, default=0.001, help='Learning rate for top layers')
    parser.add_argument('--decay_rate_bottom', type=float, default=0.05, help='Decay rate of bottom layers')
    parser.add_argument('--decay_rate_top', type=float, default=0.05, help='Decay rate of top layers')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    parser.add_argument('--max_norm', type=float, default=2.0, help='weight for max-norm regularization')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    parser.add_argument('--peepholes', action='store_true', help='Peepholes for LSTM')
    parser.add_argument('--in_to_out', action='store_true', help='input to output')
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true', help='Apply dropout layers')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--punctuation', default=None, help='List of punctuations separated by whitespace')
    parser.add_argument('--eta', type=float, default=1.0, help='relative weight for pos crf.')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--test')
    parser.add_argument('--tmp', default='tmp', help='Directory for temp files.')

    args = parser.parse_args()

    logger = utils.get_logger("Neural MSTParser")

    mode = args.mode
    rnn = args.rnn
    regular = args.regular
    embedding = args.embedding
    embedding_path = args.embedding_dict
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    tmp_dir = args.tmp
    update_algo = args.update
    grad_clipping = args.grad_clipping
    max_norm = args.max_norm
    peepholes = args.peepholes
    in_to_out = args.in_to_out
    num_filters = args.num_filters
    num_units = args.num_units
    gamma = args.gamma
    eta = args.eta
    dropout = args.dropout
    punctuation = args.punctuation
    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation.split())
        logger.info("punctuations: %s" % ' '.join(punct_set))

    X_train, POS_train, Head_train, Type_train, mask_train, \
    X_dev, POS_dev, Head_dev, Type_dev, mask_dev, \
    X_test, POS_test, Head_test, Type_test, mask_test, \
    embedd_table, word_alphabet, pos_alphabet, type_alphabet, \
    C_train, C_dev, C_test, char_embedd_table = data_processor.load_dataset_parsing(train_path, dev_path, test_path,
                                                                                    embedding=embedding,
                                                                                    embedding_path=embedding_path)

    num_pos = pos_alphabet.size() - 1
    num_types = type_alphabet.size() - 1

    logger.info("constructing network...")
    # create variables
    pos_var = T.imatrix(name='pos')
    head_var = T.imatrix(name='heads')
    type_var = T.imatrix(name='types')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    input_var = T.imatrix(name='inputs')
    num_data, max_length = X_train.shape
    alphabet_size, embedd_dim = embedd_table.shape

    char_input_var = T.itensor3(name='char-inputs')
    num_data_char, max_sent_length, max_char_length = C_train.shape
    char_alphabet_size, char_embedd_dim = char_embedd_table.shape
    assert (max_length == max_sent_length)
    assert (num_data == num_data_char)
    batch_size = args.batch_size
    learning_rate_bottom = args.learning_rate_bottom
    learning_rate_top = args.learning_rate_top
    decay_rate_bottom = args.decay_rate_bottom
    decay_rate_top = args.decay_rate_top
    momentum = args.momentum
    patience = args.patience

    layer_crf, layer_parser, layer_bottom = build_network(mode, input_var, char_input_var, mask_var, max_length,
                                                         max_char_length, alphabet_size,
                                                         char_alphabet_size, embedd_table, embedd_dim,
                                                         char_embedd_table, char_embedd_dim, num_units,
                                                         num_filters, grad_clipping, peepholes, dropout, num_pos,
                                                         num_types, rnn, in_to_out, logger)

    logger.info('RNN: %s, num_units: %d, num_filters: %d, clip: %.1f, max_norm: %.1f, peepholes: %s, in_to_out: %s' % (
        rnn, num_units, num_filters, grad_clipping, max_norm, peepholes, in_to_out))

    if mode == 'pos':
        perform_pos(layer_crf, layer_bottom, input_var, char_input_var, pos_var, mask_var, X_train, POS_train,
                    mask_train,
                    X_dev, POS_dev, mask_dev, X_test, POS_test, mask_test, C_train, C_dev, C_test,
                    num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                    decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                    pos_alphabet, tmp_dir, logger)
    elif mode == 'parse':
        perform_parse(layer_parser, layer_bottom, input_var, char_input_var, head_var, type_var, mask_var,
                      X_train, POS_train, Head_train, Type_train, mask_train,
                      X_dev, POS_dev, Head_dev, Type_dev, mask_dev,
                      X_test, POS_test, Head_test, Type_test, mask_test, C_train, C_dev, C_test,
                      num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                      decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                      pos_alphabet, type_alphabet, tmp_dir, punct_set, logger)
    elif mode == 'both':
        perform_both(layer_crf, layer_parser, layer_bottom, input_var, char_input_var, pos_var, head_var, type_var,
                     mask_var,
                     X_train, POS_train, Head_train, Type_train, mask_train, X_dev, POS_dev, Head_dev, Type_dev,
                     mask_dev,
                     X_test, POS_test, Head_test, Type_test, mask_test, C_train, C_dev, C_test,
                     num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                     decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                     pos_alphabet, type_alphabet, tmp_dir, punct_set, eta, logger)
    else:
        raise ValueError('unknown mode: %s' % mode)


if __name__ == '__main__':
    main()
