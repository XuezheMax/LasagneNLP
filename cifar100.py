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

    X_train = load_cifar10_images('data/cifar100_processed/cifar_processed_train.npy')
    y_train = load_cifar10_labels('data/cifar100_processed/train.label.txt')
    X_test = load_cifar10_images('data/cifar100_processed/cifar_processed_test.npy')
    y_test = load_cifar10_labels('data/cifar100_processed/test.label.txt')

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
    network = lasagne.layers.DenseLayer(network, 100, nonlinearity=softmax)
    return network

def main():
    parser = argparse.ArgumentParser(description='dropout experiments on mnist')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of instances in each batch')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.01, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for L-norm regularization')
    parser.add_argument('--delta', type=float, default=0.0, help='weight for expectation-linear regularization')
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    args = parser.parse_args()

    logger = utils.get_logger("CIFAR-100")
    regular = args.regular
    update_algo = args.update
    gamma = args.gamma
    delta = args.delta

    # Load the dataset
    logger.info("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset_wo_val()
    num_data, _, _, _ = X_train.shape

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    logger.info("Building model and compiling functions...")
    network = build_dnn(input_var=input_var)

    # get prediction
    prediction_train = lasagne.layers.get_output(network)
    prediction_train_det = lasagne.layers.get_output(network, deterministic=True)
    prediction_eval = lasagne.layers.get_output(network, deterministic=True)

    # compute loss
    loss_train_org = lasagne.objectives.categorical_crossentropy(prediction_train, target_var)
    loss_train_org = loss_train_org.mean()

    loss_train_expect_linear = lasagne.objectives.squared_error(prediction_train, prediction_train_det)
    loss_train_expect_linear = loss_train_expect_linear.sum(axis=1)
    loss_train_expect_linear = loss_train_expect_linear.mean()

    loss_train = loss_train_org + delta * loss_train_expect_linear
    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty

    loss_eval = lasagne.objectives.categorical_crossentropy(prediction_eval, target_var)
    loss_eval = loss_eval.mean()

    # calculate number of correct labels
    corr_train = lasagne.objectives.categorical_accuracy(prediction_train, target_var)
    corr_train = corr_train.sum(dtype=theano.config.floatX)

    corr_eval = lasagne.objectives.categorical_accuracy(prediction_eval, target_var)
    corr_eval = corr_eval.sum(dtype=theano.config.floatX)

    # Create update expressions for training.
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = 1.0 if update_algo == 'adadelta' else args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo, learning_rate, momentum=momentum)
    params_constraint = utils.get_all_params_by_name(network,
                                                     name=['cnn1.W', 'cnn2.W', 'cnn3.W', 'dense1.W', 'dense2.W'])
    assert len(params_constraint) == 5
    for param in params_constraint:
        updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=4.0)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var],
                               [loss_train, loss_train_org, loss_train_expect_linear, corr_train], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var], [loss_eval, corr_eval])

    logger.info(
        "Start training: %s with regularization: %s(%f) (#epoch: %d, #training data: %d, batch size: %d, delta: %f)..." \
        % (update_algo, regular, (0.0 if regular == 'none' else gamma), num_epochs, num_data, batch_size, delta))

    num_batches = num_data / batch_size
    lr = learning_rate
    patience = args.patience
    best_test_epoch = 0
    best_test_err = 0.
    best_test_corr = 0.
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
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
        test_corr = 0.0
        test_inst = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size):
            inputs, targets = batch
            err, corr = eval_fn(inputs, targets)
            test_err += err * inputs.shape[0]
            test_corr += corr
            test_inst += inputs.shape[0]

        if best_test_corr < test_corr:
            best_test_epoch = epoch
            best_test_corr = test_corr
            best_test_err = test_err

        print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            test_err / test_inst, test_corr, test_inst, test_corr * 100 / test_inst)
        print 'best test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
            best_test_err / test_inst, best_test_corr, test_inst, best_test_corr * 100 / test_inst)

        # re-compile a function with new learning rate for training
        if update_algo != 'adadelta':
            lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = utils.create_updates(loss_train, params, update_algo, lr, momentum=momentum)
            train_fn = theano.function([input_var, target_var],
                                       [loss_train, loss_train_org, loss_train_expect_linear, corr_train],
                                       updates=updates)

    # print last and best performance on test data.
    logger.info("final test performance (at epoch %d)" % num_epochs)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        test_err / test_inst, test_corr, test_inst, test_corr * 100 / test_inst)
    logger.info("final best acc test performance (at epoch %d)" % best_test_epoch)
    print 'test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (
        best_test_err / test_inst, best_test_corr, test_inst, best_test_corr * 100 / test_inst)


if __name__ == '__main__':
    main()


