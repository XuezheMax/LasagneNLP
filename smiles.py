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

from lasagne_nlp.utils.alphabet import Alphabet
from lasagne_nlp.utils import utils
from lasagne_nlp.networks.graph import GraphConvLayer


class Molecule:
    def __init__(self, size, atoms, atom_coordinates, edges):
        self.size = size
        self.atoms = atoms
        self.atom_coordinates = atom_coordinates
        self.edges = edges


def load_smiles_data(embedd_dim):
    def load_smiles_molocule(filename):
        with open(filename, 'r') as file:
            # skip the first three lines
            file.readline()
            file.readline()
            file.readline()

            line = file.readline().strip()
            tokens = line.split()
            size = int(tokens[0])
            num_edges = int(tokens[1])
            atoms = []
            coordinates = []
            edges = []
            for i in range(size):
                line = file.readline().strip()
                tokens = line.split()
                assert len(tokens) == 16
                atom = atom_alphabet.get_index(tokens[3])
                coordinate = (np.float32(tokens[0]), np.float32(tokens[1]), np.float32(tokens[2]))
                atoms.append(atom)
                coordinates.append(coordinate)
            for i in range(num_edges):
                line = file.readline().strip()
                tokens = line.split()
                assert len(tokens) == 4
                edges.append((int(tokens[0]) - 1, int(tokens[1]) - 1))

            molecule = Molecule(size, atoms, coordinates, edges)
        return molecule

    def load_smiles_molecules(dir, num):
        molecules = []
        for i in range(num):
            molecules.append(load_smiles_molocule(dir + '/' + 'file_%d.dat' % (i + 1)))
        return molecules

    def load_smiles_values(filename):
        values = []
        with open(filename, 'r') as file:
            # skip the first line
            file.readline()
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                values.append(np.float32(tokens[1]))
        return values

    def build_atom_embedd_table():
        scale = np.sqrt(3.0 / atom_embedd_dim)
        atom_embedd_table = np.random.uniform(-scale, scale, [atom_alphabet.size(), atom_embedd_dim]).astype(
            theano.config.floatX)
        return atom_embedd_table

    def calc_distance(coordinate1, coordinate2):
        assert len(coordinate1) == len(coordinate2)
        dist = 0.0
        for d in range(len(coordinate1)):
            dist += (coordinate1[d] - coordinate2[d]) ** 2
        return np.sqrt(dist)

    def construct_tensor(molecules, values):
        num_data = len(molecules)
        X_vertex = np.empty([num_data, max_size], dtype=np.int32)
        X_edge = np.empty([num_data, max_size, max_size, 1], dtype=theano.config.floatX)
        y = np.array(values, dtype=theano.config.floatX)
        padding_atom_id = atom_alphabet.get_index(padding_atom)
        for i in range(num_data):
            molecule = molecules[i]
            size = molecule.size
            for j in range(size):
                X_vertex[i, j] = molecule.atoms[j]
            # fill index of padding atom after the end of molecule
            X_vertex[i, size:] = padding_atom_id

            edges = molecule.edges
            coordinates = molecule.atom_coordinates
            max_distance = 0.0
            for edge in edges:
                dist = calc_distance(coordinates[edge[0]], coordinates[edge[1]])
                max_distance = max(dist, max_distance)

            for j in range(max_size):
                for k in range(max_size):
                    if j == k:
                        X_edge[i, j, k, 0] = 0.0
                    elif j < size and k < size:
                        X_edge[i, j, k, 0] = max_distance + 3.0
                    elif j >= size and k >= size:
                        X_edge[i, j, k, 0] = max_distance + 1.0
                    else:
                        X_edge[i, j, k, 0] = max_distance + 2.0
            for edge in edges:
                j = edge[0]
                k = edge[1]
                dist = calc_distance(coordinates[j], coordinates[k])
                X_edge[i, j, k, 0] = dist
                X_edge[i, k, j, 0] = dist
        return X_vertex, X_edge, y

    padding_atom = '##pad##'
    atom_alphabet = Alphabet('atom')
    atom_alphabet.get_index(padding_atom)

    molecules_train = load_smiles_molecules('data/smiles/train', 1128)
    values_train = load_smiles_values('data/smiles/train_solubility.dat')
    molecules_test = load_smiles_molecules('data/smiles/test', 16)
    values_test = load_smiles_values('data/smiles/test_solubility.dat')
    assert len(molecules_train) == len(values_train)
    assert len(molecules_test) == len(values_test)
    atom_alphabet.close()

    atom_embedd_dim = embedd_dim
    atom_embedd_table = build_atom_embedd_table()

    max_size = 0
    for molecule in molecules_train:
        max_size = max(max_size, molecule.size)
    for molecule in molecules_test:
        max_size = max(max_size, molecule.size)

    X_train_vertex, X_train_edge, y_train = construct_tensor(molecules_train, values_train)
    X_test_vertex, X_test_edge, y_test = construct_tensor(molecules_test, values_test)
    return X_train_vertex, X_train_edge, y_train, X_test_vertex, X_test_edge, y_test, \
           atom_alphabet, atom_embedd_table


def iterate_minibatches(vertexs, edges, targets, batchsize, shuffle=False):
    assert len(vertexs) == len(targets)
    assert len(edges) == len(targets)
    if shuffle:
        indices = np.arange(len(vertexs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(vertexs), batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield vertexs[excerpt], edges[excerpt], targets[excerpt]


def build_nerwork(vertex_var, edge_var, max_size, d, atom_embedd_table, depth=2, num_filters=30, filter_size=2,
                  drop_input=.2, drop_hidden=.5, nonlinearity=nonlinearities.rectify):
    alphabet_size, embedd_dim = atom_embedd_table.shape
    print 'alphabet size: %d, embedding dimension: %d' % (alphabet_size, embedd_dim)
    # vertex input layer
    # output shape [batch, max_size]
    layer_vertex = lasagne.layers.InputLayer(shape=(None, max_size), input_var=vertex_var, name='vertex')
    # output shape [batch, max_size, embedd_dim]
    layer_embedding = lasagne.layers.EmbeddingLayer(layer_vertex, input_size=alphabet_size, output_size=embedd_dim,
                                                    W=atom_embedd_table, name='embedding')
    # output shape [batch, embedd_dim, max_size]
    layer_vertex = lasagne.layers.DimshuffleLayer(layer_embedding, pattern=(0, 2, 1))
    network = layer_vertex
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)

    for i in range(depth):
        if i > 0 and drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
        # edge input layer
        # output shape [batch, max_size, max_size, d]
        layer_edge = lasagne.layers.InputLayer(shape=(None, max_size, max_size, d), input_var=edge_var,
                                               name='edge%d' % i)
        # output shape [batch, d, max_size, max_size]
        layer_edge = lasagne.layers.DimshuffleLayer(layer_edge, pattern=(0, 3, 1, 2))
        # output shape [batch, num_filters, max_size]
        network = GraphConvLayer(network, layer_edge, num_filters=num_filters, filter_size=filter_size,
                                 nonlinearity=nonlinearity, name='graph%d' % i)

    _, _, pooling_size = network.output_shape
    # output shape [batch, num_filters, 1]
    network = lasagne.layers.MaxPool1DLayer(network, pool_size=pooling_size)
    if drop_hidden:
        network = lasagne.layers.dropout(network, p=drop_hidden)

    # output shape [batch, 1]
    network = lasagne.layers.DenseLayer(network, 1, nonlinearity=nonlinearities.linear, name='dense')
    # reshape to [batch,]
    network = lasagne.layers.ReshapeLayer(network, (-1,))
    return network


def main():
    parser = argparse.ArgumentParser(description='experiments on smiles')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of instances in each batch')
    parser.add_argument('--depth', type=int, default=1, help='Number of graph cnn layers')
    parser.add_argument('--embedd_dim', type=int, default=10, help='Dimension of atom embeddings')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of units in hidden layers')
    parser.add_argument('--filter_size', type=int, default=2, help='Number of units in hidden layers')
    parser.add_argument('--activation', choices=['rectify', 'sigmod', 'tanh'],
                        help='activation function for hidden layers', default='rectify')
    parser.add_argument('--drop_input', type=float, default=0., help='dropout rate of input layer')
    parser.add_argument('--drop_hidden', type=float, default=0., help='dropout rate of hidden layer')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=1e-6, help='weight for L-norm regularization')
    parser.add_argument('--update', choices=['sgd', 'momentum', 'nesterov', 'adadelta'], help='update algorithm',
                        default='sgd')
    parser.add_argument('--regular', choices=['none', 'l2'], help='regularization for training', required=True)

    args = parser.parse_args()

    logger = utils.get_logger("SMILES")
    depth = args.depth
    regular = args.regular
    update_algo = args.update
    num_filters = args.num_filters
    filter_size = args.filter_size
    activation = args.activation
    gamma = args.gamma
    embedd_dim = args.embedd_dim
    drop_input = args.drop_input
    drop_hidden = args.drop_hidden

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
    X_train_vertex, X_train_edge, y_train, X_test_vertex, X_test_edge, y_test, \
    atom_alphabet, atom_embedd_table = load_smiles_data(embedd_dim)
    num_data, max_size, _, d = X_train_edge.shape
    logger.info("constructing network...")
    # create variables
    vertex_var = T.imatrix(name='vertex')
    edge_var = T.tensor4(name='edge', dtype=theano.config.floatX)
    target_var = T.vector(name='targets', dtype=theano.config.floatX)
    logger.info("Network structure: depth=%d, filters=%d, neighbor=%d, activation=%s" % (
        depth, num_filters, filter_size, activation))

    # construct network
    network = build_nerwork(vertex_var, edge_var, max_size, d, atom_embedd_table, depth=depth, num_filters=num_filters,
                            filter_size=filter_size, drop_input=drop_input, drop_hidden=drop_hidden,
                            nonlinearity=nonlinearity)

    prediction_train = lasagne.layers.get_output(network)
    prediction_eval = lasagne.layers.get_output(network, deterministic=True)

    loss_train = lasagne.objectives.squared_error(prediction_train, target_var)
    loss_train = loss_train.mean()
    # l2 regularization?
    if regular == 'l2':
        l2_penalty = lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)
        loss_train = loss_train + gamma * l2_penalty
    loss_eval = lasagne.objectives.squared_error(prediction_eval, target_var)
    loss_eval = loss_eval.mean()

    # Create update expressions for training.
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = 1.0 if update_algo == 'adadelta' else args.learning_rate
    decay_rate = args.decay_rate
    momentum = 0.9
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = utils.create_updates(loss_train, params, update_algo, learning_rate, momentum=momentum)
    names = [('graph%d.W' % i) for i in range(depth)]
    names.append('dense.W')
    params_constraint = utils.get_all_params_by_name(network, name=names)
    assert len(params_constraint) == depth + 1
    for param in params_constraint:
        updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=3.0)

    # Compile a function performing a training step on a mini-batch
    train_fn = theano.function([vertex_var, edge_var, target_var], loss_train, updates=updates)
    # Compile a second function evaluating the loss and accuracy of network
    eval_fn = theano.function([vertex_var, edge_var, target_var], loss_eval)

    logger.info(
        "Start training: %s with regularization: %s(%f) (#epoch: %d, #training data: %d, batch size: %d)..." \
        % (update_algo, regular, (0.0 if regular == 'none' else gamma), num_epochs, num_data, batch_size))

    num_batches = num_data / batch_size
    lr = learning_rate
    best_test_epoch = 0
    best_test_err = 1e10
    for epoch in range(1, num_epochs + 1):
        print 'Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, decay_rate)
        train_err = 0.0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in iterate_minibatches(X_train_vertex, X_train_edge, y_train, batch_size, shuffle=True):
            vertexs, edges, targets = batch
            err = train_fn(vertexs, edges, targets)
            train_err += err * vertexs.shape[0]
            train_inst += vertexs.shape[0]
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
        print 'train: %d/%d loss: %.4f, time: %.2fs' % (
            min(train_batches * batch_size, num_data), num_data, train_err / num_data, time.time() - start_time)

        # evaluate on test data
        test_err = 0.0
        test_inst = 0
        for batch in iterate_minibatches(X_test_vertex, X_test_edge, y_test, batch_size):
            vertexs, edges, targets = batch
            err = eval_fn(vertexs, edges, targets)
            test_err += err * vertexs.shape[0]
            test_inst += vertexs.shape[0]

        if best_test_err > test_err:
            best_test_epoch = epoch
            best_test_err = test_err

        print 'test loss: %.4f, total: %d' % (test_err / test_inst, test_inst)
        print 'best test loss: %.4f, total: %d' % (best_test_err / test_inst, test_inst)

        # re-compile a function with new learning rate for training
        if update_algo != 'adadelta':
            lr = learning_rate / (1.0 + epoch * decay_rate)
            updates = utils.create_updates(loss_train, params, update_algo, lr, momentum=momentum)
            params_constraint = utils.get_all_params_by_name(network, name=names)
            assert len(params_constraint) == depth + 1
            for param in params_constraint:
                updates[param] = lasagne.updates.norm_constraint(updates[param], max_norm=3.0)
            train_fn = theano.function([vertex_var, edge_var, target_var], loss_train, updates=updates)

    # print last and best performance on test data.
    logger.info("final test performance (at epoch %d)" % num_epochs)
    print 'test loss: %.4f, total: %d' % (test_err / test_inst, test_inst)
    logger.info("final best acc test performance (at epoch %d)" % best_test_epoch)
    print 'test loss: %.4f, total: %d' % (best_test_err / test_inst, test_inst)


if __name__ == '__main__':
    main()
