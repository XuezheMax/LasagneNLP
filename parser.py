__author__ = 'max'

import time
import sys
import argparse
from collections import OrderedDict
from lasagne_nlp.utils import utils
import lasagne_nlp.utils.data_processor as data_processor
from lasagne_nlp.utils.objectives import parser_loss, crf_loss, crf_accuracy
from lasagne_nlp.networks.crf import CRFLayer
from lasagne_nlp.networks.parser import DepParserLayer
import lasagne
import theano
import theano.tensor as T
from lasagne_nlp.networks.networks import build_BiLSTM_CNN

import numpy as np


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
                  num_units, num_filters, grad_clipping, peepholes, dropout, num_pos, num_types, logger):
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
        return CRFLayer(bi_lstm_cnn, num_pos, mask_input=layer_mask), None, bi_lstm_cnn

    def build_network_for_parsing():
        return None, DepParserLayer(bi_lstm_cnn, num_types, mask_input=layer_mask), bi_lstm_cnn

    def build_network_for_both():
        layer_crf = CRFLayer(bi_lstm_cnn, num_pos, mask_input=layer_mask, name='crf')
        layer_parser = DepParserLayer(bi_lstm_cnn, num_types, mask_input=layer_mask)
        return layer_crf, layer_parser, bi_lstm_cnn

    # construct input and mask layers
    layer_incoming1 = construct_char_input_layer()
    layer_incoming2 = construct_input_layer()

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=mask_var, name='mask')

    logger.info('num_units: %d, num_filters: %d, clip: %.1f, peepholes: %s' % (
        num_units, num_filters, grad_clipping, peepholes))

    bi_lstm_cnn = build_BiLSTM_CNN(layer_incoming1, layer_incoming2, num_units, mask=layer_mask,
                                   grad_clipping=grad_clipping, peepholes=peepholes, num_filters=num_filters,
                                   dropout=dropout)
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
        return utils.create_updates(loss, params, update_algo, learning_rate_top, momentum=momentum)

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
    if update_algo == 'sgd':
        return updates
    elif update_algo == 'momentum':
        updates = lasagne.updates.apply_momentum(updates, momentum=momentum)
    elif update_algo == 'nesterov':
        updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=momentum)
    else:
        raise ValueError('unkown update algorithm: %s' % update_algo)

    return updates


def perform_pos(layer_crf, bi_lstm_cnn, input_var, char_input_var, pos_var, mask_var, X_train, POS_train, mask_train,
                X_dev, POS_dev, mask_dev, X_test, POS_test, mask_test, C_train, C_dev, C_test,
                num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, pos_alphabet, tmp_dir, logger):
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
    prediction_eval, corr_eval = crf_accuracy(energies_eval, pos_var)
    corr_eval = (corr_eval * mask_var).sum(dtype=theano.config.floatX)

    learning_rate_top = 1.0 if update_algo == 'adadelta' else learning_rate_top
    learning_rate_bottom = 1.0 if update_algo == 'adadelta' else learning_rate_bottom
    updates = create_updates(loss_train, layer_crf, bi_lstm_cnn, learning_rate_top, learning_rate_bottom, momentum,
                             grad_clipping, max_norm, update_algo)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, pos_var, mask_var, char_input_var], [loss_train, corr_train, num_tokens],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, pos_var, mask_var, char_input_var],
                              [loss_eval, corr_eval, num_tokens, prediction_eval])

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
            err, corr, num, predictions = eval_fn(inputs, pos, masks, char_inputs)
            dev_err += err * inputs.shape[0]
            dev_corr += corr
            dev_total += num
            dev_inst += inputs.shape[0]
            utils.output_predictions(predictions, pos, masks, tmp_dir + '/dev%d' % epoch, pos_alphabet,
                                     is_flattened=False)

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
                err, corr, num, predictions = eval_fn(inputs, pos, masks, char_inputs)
                test_err += err * inputs.shape[0]
                test_corr += corr
                test_total += num
                test_inst += inputs.shape[0]
                utils.output_predictions(predictions, pos, masks, tmp_dir + '/test%d' % epoch, pos_alphabet,
                                         is_flattened=False)

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
            updates = create_updates(loss_train, layer_crf, bi_lstm_cnn, lr_top, lr_bottom, momentum, grad_clipping,
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


def perform_parse(layer_parser, bi_lstm_cnn, input_var, char_input_var, head_var, type_var, mask_var,
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

    '''
    loss = parser_loss(energies_train, head_var, type_var, mask_var)
    train_fn = theano.function([input_var, head_var, type_var, mask_var, char_input_var],
                               [loss, num_tokens], on_unused_input='warn')

    ss = 30000
    nn = 10000
    X = X_train[ss:ss + nn]
    head = Head_train[ss:ss + nn]
    type = Type_train[ss:ss + nn]
    mask = mask_train[ss:ss + nn]
    c = C_train[ss:ss + nn]
    loss, num = train_fn(X, head, type, mask, c)
    print loss.shape
    print loss.min()
    print loss.max()

    '''

    loss_train = parser_loss(energies_train, head_var, type_var, mask_var).mean()
    loss_eval = parser_loss(energies_eval, head_var, type_var, mask_var).mean()
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(layer_parser, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    learning_rate_top = 1.0 if update_algo == 'adadelta' else learning_rate_top
    learning_rate_bottom = 1.0 if update_algo == 'adadelta' else learning_rate_bottom
    updates = create_updates(loss_train, layer_parser, bi_lstm_cnn, learning_rate_top, learning_rate_bottom, momentum,
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
            pars_pred, types_pred = utils.decode_MST(energies, masks)
            ucorr, lcorr, total, ucorr_nopunc, \
            lcorr_nopunc, total_nopunc = utils.eval_parsing(inputs, poss, pars_pred, types_pred, heads, types, masks,
                                                            tmp_dir + '/dev%d' % epoch, word_alphabet, pos_alphabet,
                                                            type_alphabet, punct_set=punct_set)
            dev_inst += inputs.shape[0]

            dev_ucorr += ucorr
            dev_lcorr += lcorr
            dev_total += total

            dev_ucorr_nopunc += ucorr_nopunc
            dev_lcorr_nopunc += lcorr_nopunc
            dev_total_nopunc += total_nopunc

            # np.set_printoptions(linewidth=np.nan, threshold=np.nan)
            # length = masks[0].sum()
            # energy = energies[0]
            # energy = energy[:length, :length, 1:].max(axis=2)
            # weight_pred = 0.0
            # weight_gold = 0.0
            # for ch in range(1, length):
            #     weight_pred += energy[pars_pred[0, ch], ch]
            #     weight_gold += energy[heads[0, ch], ch]
            # print length
            # print energy
            # print pars_pred[0, :length], weight_pred
            # print heads[0, :length], weight_gold
            # raw_input()

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
            updates = create_updates(loss_train, layer_parser, bi_lstm_cnn, lr_top, lr_bottom, momentum, grad_clipping,
                                     max_norm, update_algo)
            train_fn = theano.function([input_var, head_var, type_var, mask_var, char_input_var],
                                       [loss_train, num_tokens], updates=updates)


def main():
    parser = argparse.ArgumentParser(description='Tuning with neural MST parser')
    parser.add_argument('--mode', choices=['pos', 'parse', 'both'], help='mode for tasks', required=True)
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna', 'random'], help='Embedding for words',
                        required=True)
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
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true', help='Apply dropout layers')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--punctuation', default=None, help='List of punctuations separated by whitespace')
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--test')
    parser.add_argument('--tmp', default='tmp', help='Directory for temp files.')

    args = parser.parse_args()

    logger = utils.get_logger("Neural MSTParser")

    mode = args.mode
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
    num_filters = args.num_filters
    num_units = args.num_units
    gamma = args.gamma
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

    layer_crf, layer_parser, bi_lstm_cnn = build_network(mode, input_var, char_input_var, mask_var, max_length,
                                                         max_char_length, alphabet_size,
                                                         char_alphabet_size, embedd_table, embedd_dim,
                                                         char_embedd_table, char_embedd_dim, num_units,
                                                         num_filters, grad_clipping, peepholes, dropout, num_pos,
                                                         num_types, logger)

    if mode == 'pos':
        perform_pos(layer_crf, bi_lstm_cnn, input_var, char_input_var, pos_var, mask_var, X_train, POS_train, mask_train,
                    X_dev, POS_dev, mask_dev, X_test, POS_test, mask_test, C_train, C_dev, C_test,
                    num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                    decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, pos_alphabet,
                    tmp_dir, logger)
    elif mode == 'parse':
        perform_parse(layer_parser, bi_lstm_cnn, input_var, char_input_var, head_var, type_var, mask_var,
                      X_train, POS_train, Head_train, Type_train, mask_train,
                      X_dev, POS_dev, Head_dev, Type_dev, mask_dev,
                      X_test, POS_test, Head_test, Type_test, mask_test, C_train, C_dev, C_test,
                      num_data, batch_size, regular, gamma, update_algo, learning_rate_bottom, learning_rate_top,
                      decay_rate_bottom, decay_rate_top, momentum, grad_clipping, max_norm, patience, word_alphabet,
                      pos_alphabet, type_alphabet, tmp_dir, punct_set, logger)
    else:
        raise ValueError('unknown mode: %s' % mode)


if __name__ == '__main__':
    main()
