__author__ = 'max'

import time
import sys
import argparse
from lasagne_nlp.utils import utils
import lasagne_nlp.utils.data_processor as data_processor
import theano.tensor as T
import theano
import lasagne
from lasagne_nlp.networks.networks import build_BiLSTM_HighCNN
import lasagne.nonlinearities as nonlinearities


def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-HighCNN')
    parser.add_argument('--fine_tune', action='store_true', help='Fine tune the word embeddings')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna'], help='Embedding for words',
                        required=True)
    parser.add_argument('--embedding_dict', default='data/word2vec/GoogleNews-vectors-negative300.bin',
                        help='path for embedding dict')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of sentences in each batch')
    parser.add_argument('--num_units', type=int, default=100, help='Number of hidden units in LSTM')
    parser.add_argument('--num_filters', type=int, default=20, help='Number of filters in CNN')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--grad_clipping', type=float, default=0, help='Gradient clipping')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for regularization')
    parser.add_argument('--peepholes', action='store_true', help='Peepholes for LSTM')
    parser.add_argument('--oov', choices=['random', 'embedding'], help='Embedding for oov word', required=True)
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta'], help='update algorithm', default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--dropout', action='store_true', help='Apply dropout layers')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--output_prediction', action='store_true', help='Output predictions to temp files')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"

    args = parser.parse_args()

    def construct_input_layer():
        if fine_tune:
            layer_input = lasagne.layers.InputLayer(shape=(None, max_length), input_var=input_var, name='input')
            layer_embedding = lasagne.layers.EmbeddingLayer(layer_input, input_size=alphabet_size,
                                                            output_size=embedd_dim,
                                                            W=embedd_table, name='embedding')
            return layer_embedding
        else:
            layer_input = lasagne.layers.InputLayer(shape=(None, max_length, embedd_dim), input_var=input_var,
                                                    name='input')
            return layer_input

    def construct_char_input_layer():
        layer_char_input = lasagne.layers.InputLayer(shape=(None, max_sent_length, max_char_length),
                                                     input_var=char_input_var, name='char-input')
        layer_char_input = lasagne.layers.reshape(layer_char_input, (-1, [2]))
        layer_char_embedding = lasagne.layers.EmbeddingLayer(layer_char_input, input_size=char_alphabet_size,
                                                             output_size=char_embedd_dim, W=char_embedd_table,
                                                             name='char_embedding')
        layer_char_input = lasagne.layers.DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))
        return layer_char_input

    logger = utils.get_logger("BiLSTM-HighCNN")
    fine_tune = args.fine_tune
    oov = args.oov
    regular = args.regular
    embedding = args.embedding
    embedding_path = args.embedding_dict
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    update_algo = args.update
    grad_clipping = args.grad_clipping
    peepholes = args.peepholes
    num_filters = args.num_filters
    gamma = args.gamma
    output_predict = args.output_prediction
    dropout = args.dropout

    X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
    embedd_table, label_alphabet, \
    C_train, C_dev, C_test, char_embedd_table = data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                                                                                              test_path, oov=oov,
                                                                                              fine_tune=fine_tune,
                                                                                              embedding=embedding,
                                                                                              embedding_path=embedding_path,
                                                                                              use_character=True)
    num_labels = label_alphabet.size() - 1

    logger.info("constructing network...")
    # create variables
    target_var = T.imatrix(name='targets')
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    if fine_tune:
        input_var = T.imatrix(name='inputs')
        num_data, max_length = X_train.shape
        alphabet_size, embedd_dim = embedd_table.shape
    else:
        input_var = T.tensor3(name='inputs', dtype=theano.config.floatX)
        num_data, max_length, embedd_dim = X_train.shape
    char_input_var = T.itensor3(name='char-inputs')
    num_data_char, max_sent_length, max_char_length = C_train.shape
    char_alphabet_size, char_embedd_dim = char_embedd_table.shape
    assert (max_length == max_sent_length)
    assert (num_data == num_data_char)

    # construct input and mask layers
    layer_incoming1 = construct_char_input_layer()
    layer_incoming2 = construct_input_layer()

    layer_mask = lasagne.layers.InputLayer(shape=(None, max_length), input_var=mask_var, name='mask')

    # construct bi-rnn-cnn
    num_units = args.num_units
    bi_lstm_cnn = build_BiLSTM_HighCNN(layer_incoming1, layer_incoming2, num_units, mask=layer_mask,
                                   grad_clipping=grad_clipping, peepholes=peepholes, num_filters=num_filters,
                                   dropout=dropout)

    # reshape bi-rnn-cnn to [batch * max_length, num_units]
    bi_lstm_cnn = lasagne.layers.reshape(bi_lstm_cnn, (-1, [2]))

    # construct output layer (dense layer with softmax)
    layer_output = lasagne.layers.DenseLayer(bi_lstm_cnn, num_units=num_labels, nonlinearity=nonlinearities.softmax,
                                             name='softmax')

    # get output of bi-lstm-cnn shape=[batch * max_length, #label]
    prediction_train = lasagne.layers.get_output(layer_output)
    prediction_eval = lasagne.layers.get_output(layer_output, deterministic=True)
    final_prediction = T.argmax(prediction_eval, axis=1)

    # flat target_var to vector
    target_var_flatten = target_var.flatten()
    # flat mask_var to vector
    mask_var_flatten = mask_var.flatten()

    # compute loss
    num_loss = mask_var_flatten.sum(dtype=theano.config.floatX)
    # for training, we use mean of loss over number of labels
    loss_train = lasagne.objectives.categorical_crossentropy(prediction_train, target_var_flatten)
    loss_train = (loss_train * mask_var_flatten).sum(dtype=theano.config.floatX) / num_loss
    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(layer_output, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    loss_eval = lasagne.objectives.categorical_crossentropy(prediction_eval, target_var_flatten)
    loss_eval = (loss_eval * mask_var_flatten).sum(dtype=theano.config.floatX) / num_loss

    # compute number of correct labels
    corr_train = lasagne.objectives.categorical_accuracy(prediction_train, target_var_flatten)
    corr_train = (corr_train * mask_var_flatten).sum(dtype=theano.config.floatX)

    corr_eval = lasagne.objectives.categorical_accuracy(prediction_eval, target_var_flatten)
    corr_eval = (corr_eval * mask_var_flatten).sum(dtype=theano.config.floatX)

    # Create update expressions for training.
    # hyper parameters to tune: learning rate, momentum, regularization.
    batch_size = args.batch_size
    learning_rate = 1.0 if update_algo == 'adadelta' else args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(layer_output, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo, learning_rate, momentum=momentum)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var, mask_var, char_input_var], [loss_train, corr_train, num_loss],
                               updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                              [loss_eval, corr_eval, num_loss, final_prediction])

    # Finally, launch the training loop.
    logger.info(
        "Start training: %s with regularization: %s(%f), dropout: %s, fine tune: %s (#training data: %d, batch size: %d, clip: %.1f, peepholes: %s)..." \
        % (
            update_algo, regular, (0.0 if regular == 'none' else gamma), dropout, fine_tune, num_data, batch_size, grad_clipping,
            peepholes))
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
    lr = learning_rate
    patience = args.patience
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in utils.iterate_minibatches(X_train, Y_train, masks=mask_train, char_inputs=C_train,
                                               batch_size=batch_size, shuffle=True):
            inputs, targets, masks, char_inputs = batch
            err, corr, num = train_fn(inputs, targets, masks, char_inputs)
            train_err += err * num
            train_corr += corr
            train_total += num
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data,
                train_err / train_total, train_corr * 100 / train_total, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err / train_total, train_corr * 100 / train_total, time.time() - start_time)

        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        for batch in utils.iterate_minibatches(X_dev, Y_dev, masks=mask_dev, char_inputs=C_dev, batch_size=batch_size):
            inputs, targets, masks, char_inputs = batch
            err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
            dev_err += err * num
            dev_corr += corr
            dev_total += num
            if output_predict:
                utils.output_predictions(predictions, targets, masks, 'tmp/dev%d' % epoch, label_alphabet)

        print 'dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            dev_err / dev_total, dev_corr, dev_total, dev_corr * 100 / dev_total)

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
            for batch in utils.iterate_minibatches(X_test, Y_test, masks=mask_test, char_inputs=C_test,
                                                   batch_size=batch_size):
                inputs, targets, masks, char_inputs = batch
                err, corr, num, predictions = eval_fn(inputs, targets, masks, char_inputs)
                test_err += err * num
                test_corr += corr
                test_total += num
                if output_predict:
                    utils.output_predictions(predictions, targets, masks, 'tmp/test%d' % epoch, label_alphabet)

            print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
                test_err / test_total, test_corr, test_total, test_corr * 100 / test_total)

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
            lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = utils.create_updates(loss_train, params, update_algo, lr, momentum=momentum)
            train_fn = theano.function([input_var, target_var, mask_var, char_input_var],
                                       [loss_train, corr_train, num_loss],
                                       updates=updates)

    # print best performance on test data.
    logger.info("final best loss test performance (at epoch %d)" % best_epoch_loss)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_loss_test_err / test_total, best_loss_test_corr, test_total, best_loss_test_corr * 100 / test_total)
    logger.info("final best acc test performance (at epoch %d)" % best_epoch_acc)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_acc_test_err / test_total, best_acc_test_corr, test_total, best_acc_test_corr * 100 / test_total)


if __name__ == '__main__':
    main()
