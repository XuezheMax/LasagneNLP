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


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        basename = filename[11:]
        print("Downloading %s" % basename)
        urlretrieve(source + basename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('data/mnist/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('data/mnist/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('data/mnist/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/mnist/t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_dataset_wo_val():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        basename = filename[11:]
        print("Downloading %s" % basename)
        urlretrieve(source + basename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('data/mnist/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('data/mnist/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('data/mnist/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('data/mnist/t10k-labels-idx1-ubyte.gz')

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


def build_dropout_dnn(input_var=None, depth=2, num_units=1024, drop_input=.2, drop_hidden=.5,
                      nonlinearity=nonlinearities.rectify):
    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    for d in range(depth):
        network = lasagne.layers.DenseLayer(network, num_units, nonlinearity=nonlinearity, name=('hidden%d' % d))
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
    return network


def main():
    parser = argparse.ArgumentParser(description='dropout experiments on mnist')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of instances in each batch')
    parser.add_argument('--depth', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num_units', type=int, default=1024, help='Number of units in hidden layers')
    parser.add_argument('--activation', choices=['rectify', 'sigmod', 'tanh'],
                        help='activation function for hidden layers', default='rectify')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for L-norm regularization')
    parser.add_argument('--delta', type=float, default=0.0, help='weight for expectation-linear regularization')
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adam'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)
    parser.add_argument('--mc', type=int, default=100, help='MC sampling size')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    args = parser.parse_args()

    logger = utils.get_logger("MNIST")
    regular = args.regular
    update_algo = args.update
    depth = args.depth
    num_units = args.num_units
    activation = args.activation
    gamma = args.gamma
    delta = args.delta
    mc = args.mc
    batch_size = args.batch_size

    # construct nonlinearity
    nonlinearity = nonlinearities.rectify
    if activation == 'rectify':
        nonlinearity = nonlinearities.rectify
    elif activation == 'sigmod':
        nonlinearity = nonlinearities.sigmoid
    elif activation == 'tanh':
        nonlinearity = nonlinearities.tanh
    else:
        raise ValueError('unkown activation function: %s' % activation)

    # Load the dataset
    logger.info("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset_wo_val()
    num_data, _, _, _ = X_train.shape

    # Prepare Theano variables for inputs and targets
    # shape = [batch, 1, 28, 28] for train, [mc * batch, 1, 28, 28] for test
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    logger.info("Building model and compiling functions...")
    network = build_dropout_dnn(input_var=input_var, depth=depth, num_units=num_units, nonlinearity=nonlinearity)

    # get prediction
    # shape = [batch, num_labels]
    prediction_train = lasagne.layers.get_output(network)
    prediction_train_det = lasagne.layers.get_output(network, deterministic=True)
    # shape = [mc * batch, num_labels]
    prediction_eval = lasagne.layers.get_output(network, deterministic=True)
    prediction_eval_mc = lasagne.layers.get_output(network)
    # reshape to [mc, batch, num_labels]
    prediction_eval = prediction_eval.reshape([mc, batch_size, 10])
    prediction_eval_mc = prediction_eval_mc.reshape([mc, batch_size, 10])
    # calc mean, shape = [batch, num_labels]
    prediction_eval = prediction_eval.mean(axis=0)
    prediction_eval_mc = prediction_eval_mc.mean(axis=0)

    logger.info("Network structure: depth=%d, hidden=%d, activation=%s" % (depth, num_units, activation))

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
    learning_rate = args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo, learning_rate, momentum=momentum)
    params_constraint = utils.get_all_params_by_name(network, name=[('hidden%d.W' % d) for d in range(depth)])
    assert len(params_constraint) == depth
    for param in params_constraint:
        updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=3.5)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([input_var, target_var],
                               [loss_train, loss_train_org, loss_train_expect_linear, corr_train], updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([input_var, target_var],
                              [loss_eval, loss_eval_mc, loss_test_expect_linear, corr_eval, corr_eval_mc])

    logger.info(
        "Start training: %s with regularization: %s(%f) (#epoch: %d, #training data: %d, batch size: %d, delta: %f)..." \
        % (update_algo, regular, (0.0 if regular == 'none' else gamma), num_epochs, num_data, batch_size, delta))

    num_batches = num_data / batch_size
    lr = learning_rate
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
        test_err_mc = 0.0
        test_err_linear = 0.0
        test_corr = 0.0
        test_corr_mc = 0.0
        test_inst = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size):
            inputs, targets = batch
            inputs_mc = np.empty((mc,) + inputs.shape, dtype=theano.config.floatX)
            inputs_mc[np.arange(mc)] = inputs
            inputs_mc = inputs_mc.reshape([mc * batch_size, 1, 28, 28])

            err, err_mc, err_linear, corr, corr_mc = eval_fn(inputs_mc, targets)
            test_err += err * inputs.shape[0]
            test_err_mc += err_mc * inputs.shape[0]
            test_err_linear += err_linear * inputs.shape[0]
            test_corr += corr
            test_corr_mc += corr_mc
            test_inst += inputs.shape[0]

        if best_det_corr < test_corr:
            best_det_epoch = epoch
            best_det_corr = test_corr
            best_det_corr_mc = test_corr_mc
            best_det_err = test_err
            best_det_err_mc = test_err_mc
            best_det_err_linear = test_err_linear

        if best_mc_corr < test_corr:
            best_mc_epoch = epoch
            best_mc_corr = test_corr
            best_mc_corr_mc = test_corr_mc
            best_mc_err = test_err
            best_mc_err_mc = test_err_mc
            best_mc_err_linear = test_err_linear

        print 'test loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%%' % (
            test_err / test_inst, test_err_mc / test_inst, test_err_linear / test_inst, test_corr, test_corr_mc,
            test_inst, test_corr * 100 / test_inst, test_corr_mc * 100 / test_inst)
        print 'best det loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%% (epoch: %d)' % (
            best_det_err / test_inst, best_det_err_mc / test_inst, best_det_err_linear / test_inst, best_det_corr,
            best_det_corr_mc, test_inst, best_det_corr * 100 / test_inst, best_det_corr_mc * 100 / test_inst, best_det_epoch)
        print 'best mc  loss: %.4f, loss_mc: %.4f, loss_linear: %.4f, corr: %d, corr_mc: %d, total: %d, acc: %.2f%%, acc_mc: %.2f%% (epoch: %d)' % (
            best_mc_err / test_inst, best_mc_err_mc / test_inst, best_mc_err_linear / test_inst, best_mc_corr,
            best_mc_corr_mc, test_inst, best_mc_corr * 100 / test_inst, best_mc_corr_mc * 100 / test_inst, best_mc_epoch)

        # re-compile a function with new learning rate for training
        if update_algo != 'adam':
            lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = utils.create_updates(loss_train, params, update_algo, lr, momentum=momentum)
            params_constraint = utils.get_all_params_by_name(network,
                                                             name=[('hidden%d.W' % d) for d in range(depth)])
            assert len(params_constraint) == depth
            for param in params_constraint:
                updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=3.5)

            train_fn = theano.function([input_var, target_var],
                                       [loss_train, loss_train_org, loss_train_expect_linear, corr_train],
                                       updates=updates)

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
