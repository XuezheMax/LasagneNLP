__author__ = 'max'

import sys
import os
import time
import argparse

import numpy as np
import theano
import theano.tensor as T

import lasagne
import lasagne.nonlinearities as nonlinearities

from lasagne_nlp.utils import utils


def load_dataset_wo_val():
    def load_cifar10_images(filename):
        data = np.load(filename)
        data = data.reshape(-1, 3, 32, 32)
        return data

    def load_cifar10_labels(filename):
        labels = np.loadtxt(filename, np.int32)
        return labels

    X_train = load_cifar10_images('data/cifar10_processed/cifar_processed_train.npy')
    y_train = load_cifar10_labels('data/cifar10_processed/train.label.txt')
    X_test = load_cifar10_images('data/cifar10_processed/cifar_processed_test.npy')
    y_test = load_cifar10_labels('data/cifar10_processed/test.label.txt')

    return X_train, y_train, X_test, y_test


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


def build_network(input_var=None, architecture='dnn'):
    if architecture == 'dnn':
        return build_dnn(input_var)
    elif architecture == 'allConvB':
        return build_allConvB(input_var)
    elif architecture == 'allConvC':
        return build_allConvC(input_var)
    else:
        raise ValueError('unkown architecture: %s' % architecture)


def build_dnn(input_var=None):
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    network = lasagne.layers.dropout(network, p=0.1)
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(5, 5), stride=(1, 1), pad='same',
                                         W=lasagne.init.Uniform(), nonlinearity=nonlinearities.rectify, name='cnn1')
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2, 2))
    network = lasagne.layers.dropout(network, p=0.25)
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(5, 5), stride=(1, 1), pad='same',
                                         W=lasagne.init.Uniform(), nonlinearity=nonlinearities.rectify, name='cnn2')
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2, 2))
    network = lasagne.layers.dropout(network, p=0.25)
    # The third CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(5, 5), stride=(1, 1), pad='same',
                                         W=lasagne.init.Uniform(), nonlinearity=nonlinearities.rectify, name='cnn3')
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2, 2))
    network = lasagne.layers.dropout(network, p=0.5)
    # the first dense layer
    network = lasagne.layers.DenseLayer(network, 2048, nonlinearity=nonlinearities.rectify, name='dense1')
    network = lasagne.layers.dropout(network, p=0.5)
    # the second dense layer
    network = lasagne.layers.DenseLayer(network, 2048, nonlinearity=nonlinearities.rectify, name='dense2')
    network = lasagne.layers.dropout(network, p=0.5)
    # Output layer:
    softmax = nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax, name='output')
    return network


def build_allConvB(input_var=None):
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    network = lasagne.layers.dropout(network, p=0.2)
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(5, 5), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn1')
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(1, 1), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn2')
    # The third CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(3, 3), stride=(2, 2), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn3')
    # Dropout layer
    network = lasagne.layers.dropout(network, p=0.5)

    # ------------------------------------------------------------
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(5, 5), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn4')
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1, 1), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn5')
    # The third CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(3, 3), stride=(2, 2), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn6')
    # Dropout layer
    network = lasagne.layers.dropout(network, p=0.5)

    # ------------------------------------------------------------
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(3, 3), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn7')
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1, 1), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn8')
    # The third CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(1, 1), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn9')
    # Global pooling layer
    network = lasagne.layers.GlobalPoolLayer(network)
    # Output layer (softmax)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=nonlinearities.softmax)
    return network


def build_allConvC(input_var=None):
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32), input_var=input_var)
    network = lasagne.layers.dropout(network, p=0.2)
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(3, 3), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn1')
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=96, filter_size=(3, 3), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn2')
    # The Pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2, 2))
    # Dropout layer
    network = lasagne.layers.dropout(network, p=0.5)

    # ------------------------------------------------------------
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(3, 3), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn4')
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(3, 3), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn5')
    # The Pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(3, 3), stride=(2, 2))
    # Dropout layer
    network = lasagne.layers.dropout(network, p=0.5)

    # ------------------------------------------------------------
    # The first CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(3, 3), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn7')
    # The second CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=192, filter_size=(1, 1), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn8')
    # The third CNN layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=10, filter_size=(1, 1), stride=(1, 1), pad='same',
                                         W=lasagne.init.Normal(std=0.05), nonlinearity=nonlinearities.rectify,
                                         name='cnn9')
    # Global pooling layer
    network = lasagne.layers.GlobalPoolLayer(network)
    # Output layer (softmax)
    network = lasagne.layers.NonlinearityLayer(network, nonlinearity=nonlinearities.softmax)
    return network


def create_updates_dnn(loss, network, learning_rate_cnn, learning_rate_dense, momentum, momentum_type):
    # for dnn, only linear patten is valid
    params = lasagne.layers.get_all_params(network, trainable=True)
    params_cnn = utils.get_all_params_by_name(network, ['cnn1.W', 'cnn1.b', 'cnn2.W', 'cnn2.b', 'cnn3.W', 'cnn3.b'],
                                              trainable=True)
    params_dense = utils.get_all_params_by_name(network, ['dense1.W', 'dense1.b', 'dense2.W', 'dense2.b', 'output.W',
                                                          'output.b'], trainable=True)
    params_constraint = utils.get_all_params_by_name(network,
                                                     name=['cnn1.W', 'cnn2.W', 'cnn3.W', 'dense1.W', 'dense2.W',
                                                           'output.W'], trainable=True)
    assert len(params) == 12
    assert len(params_cnn) == 6
    assert len(params_dense) == 6
    assert len(params_constraint) == 6
    updates = lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate_cnn)
    updates_dense = lasagne.updates.sgd(loss, params=params_dense, learning_rate=learning_rate_dense)
    for param in params_dense:
        assert param in updates
        updates[param] = updates_dense[param]
    # apply momentum term
    if momentum_type == 'normal':
        updates = lasagne.updates.apply_momentum(updates, momentum=momentum)
    elif momentum_type == 'nesterov':
        updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=momentum)
    else:
        raise ValueError('unkown momentum type: %s' % momentum_type)
    # add norm constraints (should be after momentum)
    for param in params_constraint:
        assert param in updates
        updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=4.0)

    return updates


def create_updates_allConv(opt, loss, network, learning_rate_cnn, momentum, momentum_type):
    params = lasagne.layers.get_all_params(network, trainable=True)
    if opt == 'adam':
        updates = lasagne.updates.adam(loss, params=params, learning_rate=learning_rate_cnn)
    elif opt in ['linear', 'schedule']:
        updates = lasagne.updates.sgd(loss, params=params, learning_rate=learning_rate_cnn)
        # apply momentum term
        if momentum_type == 'normal':
            updates = lasagne.updates.apply_momentum(updates, momentum=momentum)
        elif momentum_type == 'nesterov':
            updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=momentum)
        else:
            raise ValueError('unkown momentum type: %s' % momentum_type)
    else:
        raise ValueError('unkown optimization algorithm: %s' % opt)
    return updates


def create_updates(architecture, opt, loss, network, learning_rate_cnn, learning_rate_dense, momentum, momentum_type):
    if architecture == 'dnn':
        return create_updates_dnn(loss, network, learning_rate_cnn, learning_rate_dense, momentum, momentum_type)
    else:
        return create_updates_allConv(opt, loss, network, learning_rate_cnn, momentum, momentum_type)


def main():
    parser = argparse.ArgumentParser(description='dropout experiments on cifar-10')
    parser.add_argument('--architecture', choices=['dnn', 'allConvB', 'allConvC'], help='network architecture',
                        required=True)
    parser.add_argument('--opt', choices=['adam', 'linear', 'schedule'], help='optimization method',
                        required=True)
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of instances in each batch')
    parser.add_argument('--learning_rate_cnn', type=float, default=0.001, help='learning rate for CNNs')
    parser.add_argument('--learning_rate_dense', type=float, default=0.1, help='learning rate for dense layers')
    parser.add_argument('--decay_rate', type=float, default=0.005, help='Decay rate of learning rate')
    parser.add_argument('--momentum0', type=float, default=0.5, help='initial momentum')
    parser.add_argument('--momentum1', type=float, default=0.95, help='final momentum')
    parser.add_argument('--momentum_type', choices=['normal', 'nesterov'], help='type of momentum', required=True)
    parser.add_argument('--gamma', type=float, default=1e-3, help='weight for L-norm regularization')
    parser.add_argument('--delta', type=float, default=0.0, help='weight for expectation-linear regularization')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--mc', type=int, default=100, help='MC sampling size')
    parser.add_argument('--batch_mc', type=int, default=50, help='Number of instances in each batch for MC')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    args = parser.parse_args()

    logger = utils.get_logger("CIFAR-10")
    regular = args.regular
    gamma = args.gamma
    delta = args.delta
    mc = args.mc
    batch_mc = args.batch_mc
    num_labels = 10
    architecture = args.architecture
    opt = args.opt

    # Load the dataset
    logger.info("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset_wo_val()

    num_data, _, _, _ = X_train.shape
    num_data_test, _, _, _ = X_test.shape

    # Prepare Theano variables for inputs and targets
    # shape = [batch, 3, 32, 32] for train, [mc * batch, 3, 32, 32] for test
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    logger.info("Building model and compiling functions (architecture: %s)..." % architecture)
    network = build_network(input_var=input_var, architecture=architecture)

    # get prediction
    prediction_train = lasagne.layers.get_output(network)
    prediction_train_det = lasagne.layers.get_output(network, deterministic=True)
    # shape = [mc * batch, num_labels]
    prediction_eval = lasagne.layers.get_output(network, deterministic=True)
    prediction_eval_mc = lasagne.layers.get_output(network)
    # reshape to [mc, batch, num_labels]
    prediction_eval = prediction_eval.reshape([mc, batch_mc, num_labels])
    prediction_eval_mc = prediction_eval_mc.reshape([mc, batch_mc, num_labels])
    # calc mean, shape = [batch, num_labels]
    prediction_eval = prediction_eval.mean(axis=0)
    prediction_eval_mc = prediction_eval_mc.mean(axis=0)

    # compute loss
    loss_train_org = lasagne.objectives.categorical_crossentropy(prediction_train, target_var)
    loss_train_org = loss_train_org.mean()

    loss_train_expect_linear = lasagne.objectives.squared_error(prediction_train, prediction_train_det)
    loss_train_expect_linear = loss_train_expect_linear.sum(axis=1)
    loss_train_expect_linear = loss_train_expect_linear.mean()

    loss_train = loss_train_org + delta * loss_train_expect_linear
    # l2 regularization?
    if regular == 'l2':
        if architecture == 'dnn':
            params_regular = utils.get_all_params_by_name(network, name=['cnn1.W', 'cnn2.W', 'cnn3.W', 'output.W'],
                                                          trainable=True, regularizable=True)
            assert len(params_regular) == 4
            l2_penalty = lasagne.regularization.apply_penalty(params_regular, lasagne.regularization.l2)
        else:
            l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    loss_eval = lasagne.objectives.categorical_crossentropy(prediction_eval, target_var)
    loss_eval = loss_eval.mean()
    loss_eval_mc = lasagne.objectives.categorical_crossentropy(prediction_eval_mc, target_var)
    loss_eval_mc = loss_eval_mc.mean()

    loss_test_expect_linear = lasagne.objectives.squared_error(prediction_eval_mc, prediction_eval)
    loss_test_expect_linear = loss_test_expect_linear.sum(axis=1)
    loss_test_expect_linear = loss_test_expect_linear.mean()

    # calculate number of correct labels
    corr_train = lasagne.objectives.categorical_accuracy(prediction_train, target_var)
    corr_train = corr_train.sum(dtype=theano.config.floatX)

    corr_eval = lasagne.objectives.categorical_accuracy(prediction_eval, target_var)
    corr_eval = corr_eval.sum(dtype=theano.config.floatX)

    corr_eval_mc = lasagne.objectives.categorical_accuracy(prediction_eval_mc, target_var)
    corr_eval_mc = corr_eval_mc.sum(dtype=theano.config.floatX)

    # Create update expressions for training.
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    # learning_rate = 1.0 if update_algo == 'adadelta' else args.learning_rate
    learning_rate_cnn = args.learning_rate_cnn
    learning_rate_dense = args.learning_rate_dense
    momentum0 = args.momentum0
    momentum1 = args.momentum1
    momentum_type = args.momentum_type
    momentum_increase_rate = 0.05
    updates = create_updates(architecture, opt, loss_train, network, learning_rate_cnn=learning_rate_cnn,
                             learning_rate_dense=learning_rate_dense, momentum=momentum0, momentum_type=momentum_type)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var],
                               [loss_train, loss_train_org, loss_train_expect_linear, corr_train], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var],
                              [loss_eval, loss_eval_mc, loss_test_expect_linear, corr_eval, corr_eval_mc])

    logger.info(
        "Start training with regularization: %s(%f), opt: %s, momentum: %s, (#epoch: %d, #training data: %d, batch size: %d, delta: %f)..." \
        % (regular, (0.0 if regular == 'none' else gamma), opt, momentum_type, num_epochs, num_data, batch_size, delta))

    num_batches = num_data / batch_size
    num_batches_test = num_data_test / batch_mc
    decay_rate = args.decay_rate
    lr_cnn = learning_rate_cnn
    lr_dense = learning_rate_dense
    momentum = momentum0
    patience = args.patience
    best_det_epoch = 0
    best_det_err = 0.
    best_det_err_mc = 0.
    best_det_err_linear = 0.
    best_det_corr = 0.
    best_det_corr_mc = 0.

    best_mc_epoch = 0
    best_mc_err = 0.
    best_mc_err_mc = 0.
    best_mc_err_linear = 0.
    best_mc_corr = 0.
    best_mc_corr_mc = 0.
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=(%.4f, %.4f), decay rate=%.4f, momentum=%.4f, increase rate=%.4f): ' % (
            epoch, lr_cnn, lr_dense, decay_rate, momentum, momentum_increase_rate)
        train_err = 0.0
        train_err_org = 0.0
        train_err_linear = 0.0
        train_corr = 0.0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            err, err_org, err_linear, corr = train_fn(inputs, targets)
            train_err += err * inputs.shape[0]
            train_err_org += err_org * inputs.shape[0]
            train_err_linear += err_linear * inputs.shape[0]
            train_corr += corr
            train_inst += inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave

            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, loss_org: %.4f, loss_linear: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                min(train_batches * batch_size, num_data), num_data,
                train_err / train_inst, train_err_org / train_inst, train_err_linear / train_inst,
                train_corr * 100 / train_inst, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == num_data
        sys.stdout.write("\b" * num_back)
        print 'train: %d/%d loss: %.4f, loss_org: %.4f, loss_linear: %.4f, acc: %.2f%%, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data,
            train_err / num_data, train_err_org / num_data, train_err_linear / num_data, train_corr * 100 / num_data,
            time.time() - start_time)

        # evaluate on test data
        test_err = 0.0
        test_err_mc = 0.0
        test_err_linear = 0.0
        test_corr = 0.0
        test_corr_mc = 0.0
        test_inst = 0
        num_back = 0
        start_time = time.time()
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_mc):
            inputs, targets = batch
            inputs_mc = np.empty((mc,) + inputs.shape, dtype=theano.config.floatX)
            inputs_mc[np.arange(mc)] = inputs
            inputs_mc = inputs_mc.reshape([mc * batch_mc, 3, 32, 32])

            err, err_mc, err_linear, corr, corr_mc = eval_fn(inputs_mc, targets)
            test_err += err * inputs.shape[0]
            test_err_mc += err_mc * inputs.shape[0]
            test_err_linear += err_linear * inputs.shape[0]
            test_corr += corr
            test_corr_mc += corr_mc
            test_inst += inputs.shape[0]

            test_batches += 1
            time_ave = (time.time() - start_time) / test_batches
            time_left = (num_batches_test - test_batches) * time_ave
            # update log
            sys.stdout.write("\b" * num_back)
            log_info = 'test: %d/%d time left (estimated): %.2fs' % (min(test_batches * batch_mc, num_data_test),
                                                                     num_data_test, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)

        if best_det_corr < test_corr:
            best_det_epoch = epoch
            best_det_corr = test_corr
            best_det_corr_mc = test_corr_mc
            best_det_err = test_err
            best_det_err_mc = test_err_mc
            best_det_err_linear = test_err_linear

        if best_mc_corr_mc < test_corr_mc:
            best_mc_epoch = epoch
            best_mc_corr = test_corr
            best_mc_corr_mc = test_corr_mc
            best_mc_err = test_err
            best_mc_err_mc = test_err_mc
            best_mc_err_linear = test_err_linear

        sys.stdout.write("\b" * num_back)
        print 'test loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%%, time: %.2fs' % (
            test_err / test_inst, test_err_mc / test_inst, test_err_linear / test_inst, test_corr, test_corr_mc,
            test_inst, test_corr * 100 / test_inst, test_corr_mc * 100 / test_inst, time.time() - start_time)
        print 'best det loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%% (epoch: %d)' % (
            best_det_err / test_inst, best_det_err_mc / test_inst, best_det_err_linear / test_inst, best_det_corr,
            best_det_corr_mc, test_inst, best_det_corr * 100 / test_inst, best_det_corr_mc * 100 / test_inst,
            best_det_epoch)
        print 'best mc  loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%% (epoch: %d)' % (
            best_mc_err / test_inst, best_mc_err_mc / test_inst, best_mc_err_linear / test_inst, best_mc_corr,
            best_mc_corr_mc, test_inst, best_mc_corr * 100 / test_inst, best_mc_corr_mc * 100 / test_inst,
            best_mc_epoch)

        if opt == 'linear':
            # re-compile a function with new learning rate for training
            lr_cnn = learning_rate_cnn / (1.0 + epoch * decay_rate)
            lr_dense = learning_rate_dense / (1.0 + epoch * decay_rate)
            f = momentum_increase_rate * epoch
            if f > 1.0:
                momentum = momentum1
            else:
                momentum = (1 - f) * momentum0 + f * momentum1

            updates = create_updates(architecture, opt, loss_train, network, learning_rate_cnn=lr_cnn,
                                     learning_rate_dense=lr_dense,
                                     momentum=momentum, momentum_type=momentum_type)

            train_fn = theano.function([input_var, target_var],
                                       [loss_train, loss_train_org, loss_train_expect_linear, corr_train],
                                       updates=updates)
        elif opt in ['adam', 'schedule']:
            if architecture == 'dnn':
                raise ValueError('value error: dnn cannot have optimization method %s' % opt)

            if opt == 'schedule' and epoch in [200, 250, 300]:
                lr_cnn = lr_cnn * decay_rate
                updates = create_updates(architecture, opt, loss_train, network, learning_rate_cnn=lr_cnn,
                                         learning_rate_dense=lr_dense,
                                         momentum=momentum, momentum_type=momentum_type)

                train_fn = theano.function([input_var, target_var],
                                           [loss_train, loss_train_org, loss_train_expect_linear, corr_train],
                                           updates=updates)
        else:
            raise ValueError('unkown optimization algorithm: %s' % opt)

    # print last and best performance on test data.
    logger.info("final test performance (at epoch %d)" % num_epochs)
    print 'test loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%%' % (
        test_err / test_inst, test_err_mc / test_inst, test_err_linear / test_inst, test_corr, test_corr_mc,
        test_inst, test_corr * 100 / test_inst, test_corr_mc * 100 / test_inst)
    logger.info("final best det acc test performance (at epoch %d)" % best_det_epoch)
    print 'test loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%%' % (
        best_det_err / test_inst, best_det_err_mc / test_inst, best_det_err_linear / test_inst, best_det_corr,
        best_det_corr_mc, test_inst, best_det_corr * 100 / test_inst, best_det_corr_mc * 100 / test_inst)
    logger.info("final best mc  acc test performance (at epoch %d)" % best_mc_epoch)
    print 'test loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%%' % (
        best_mc_err / test_inst, best_mc_err_mc / test_inst, best_mc_err_linear / test_inst, best_mc_corr,
        best_mc_corr_mc, test_inst, best_mc_corr * 100 / test_inst, best_mc_corr_mc * 100 / test_inst)


if __name__ == '__main__':
    main()
