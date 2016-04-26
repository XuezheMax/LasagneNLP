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


class Molecule:
    def __init__(self, size, atoms, atom_coordinates, edges):
        self.size = size
        self.atoms = atoms
        self.atom_coordinates = atom_coordinates
        self.edges = edges


def load_smiles_data():
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
        X_edge = np.empty([num_data, max_size, max_size], dtype=theano.config.floatX)
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
                        X_edge[i, j, k] = 0.0
                    elif j < size and k < size:
                        X_edge[i, j, k] = max_distance + 3.0
                    elif j >= size and k >= size:
                        X_edge[i, j, k] = max_distance + 1.0
                    else:
                        X_edge[i, j, k] = max_distance + 2.0
            for edge in edges:
                j = edge[0]
                k = edge[1]
                dist = calc_distance(coordinates[j], coordinates[k])
                X_edge[i, j, k] = dist
                X_edge[i, k, j] = dist
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

    atom_embedd_dim = 30
    atom_embedd_table = build_atom_embedd_table()

    max_size = 0
    for molecule in molecules_train:
        max_size = max(max_size, molecule.size)
    for molecule in molecules_test:
        max_size = max(max_size, molecule.size)

    X_train_vertex, X_train_edge, y_train = construct_tensor(molecules_train, values_train)
    X_test_vertex, X_test_edge, y_test = construct_tensor(molecules_test, values_test)
    return X_train_vertex, X_train_edge, y_train, X_test_vertex, X_test_edge, y_test, atom_alphabet, atom_embedd_table


def main():
    load_smiles_data()


if __name__ == '__main__':
    main()
