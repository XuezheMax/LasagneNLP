"""
objectives for some loss functions
"""
__author__ = 'max'

import numpy as np
import theano
import theano.tensor as T
import lasagne_nlp.theano.nlinalg as nlinalg


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """

    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def parser_loss(energies, heads, types, masks):
    """
    compute minus log likelihood of parser as parser loss.
    :param energies: Theano 4D tensor
        energies of each edge. the shape is [batch_size, n_steps, n_steps, num_labels],
        where the summy root is at index 0.
    :param heads: Theano 2D tensor
        heads in the shape [batch_size, n_steps].
    :param types: Theano 2D tensor
        types in the shape [batch_size, n_steps].
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_steps].
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """
    input_shape = energies.shape
    batch_size = input_shape[0]
    length = input_shape[1]
    # get the exp of energies, and add along the label axis.
    # the shape is [batch_size, n, n].
    E = T.exp(energies).sum(axis=3)

    # zero out the elements out the length of each sentence.
    if masks is not None:
        masks_shuffled = masks.dimshuffle(0, 1, 'x')
        E = E * masks_shuffled
        masks_shuffled = masks.dimshuffle(0, 'x', 1)
        E = E * masks_shuffled

    # compute the D tensor.
    # the shape is [batch_size, n, n]
    D = E.sum(axis=1)
    D = T.zeros_like(E) + D.dimshuffle(0, 1, 'x')
    # zeros out all elements except diagonal.
    D = D * T.eye(length, length, 0).dimshuffle('x', 0, 1)

    # compute lengths
    lengths = T.cast(masks, dtype='int32').sum(axis=1)
    # compute laplacian matrix
    L = D - E

    # compute partition Z(x)
    partitions, _ = theano.scan(fn=lambda laps, length: nlinalg.logabsdet(laps[1:length, 1:length]), outputs_info=None,
                                sequences=[L, lengths])

    # compute targets energy
    # first create indice matrix
    indices = T.zeros_like(heads) + T.arange(length).dimshuffle('x', 0)
    # compute loss matrix shape = [n_steps, batch_size]
    target_energy = energies[T.arange(batch_size), heads.T, indices.T, types.T]
    # shuffle loss to [batch_size, n_steps]
    target_energy = target_energy.dimshuffle(1, 0)
    # remove the first element [batch, n_steps -1]
    target_energy = target_energy[:, 1:]
    # sum over n_step shape = [batch_size]
    target_energy = target_energy.sum(axis=1)

    return partitions - target_energy#, E, D, L, partitions, target_energy


def crf_loss(energies, targets, masks):
    """
    compute minus log likelihood of crf as crf loss.
    :param energies: Theano 4D tensor
        energies of each step. the shape is [batch_size, n_time_steps, num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert energies.ndim == 4
    assert targets.ndim == 2
    assert masks.ndim == 2

    def inner_function(energies_one_step, targets_one_step, mask_one_step, prior_partition, prev_label, tg_energy):
        """

        :param energies_one_step: [batch_size, t, t]
        :param targets_one_step: [batch_size]
        :param prior_partition: [batch_size, t]
        :param prev_label: [batch_size]
        :param tg_energy: [batch_size]
        :return:
        """

        partition_shuffled = prior_partition.dimshuffle(0, 1, 'x')
        partition_t = T.switch(mask_one_step.dimshuffle(0, 'x'),
                               theano_logsumexp(energies_one_step + partition_shuffled, axis=1),
                               prior_partition)

        return [partition_t, targets_one_step,
                tg_energy + energies_one_step[T.arange(energies_one_step.shape[0]), prev_label, targets_one_step]]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    targets_shuffled = targets.dimshuffle(1, 0)
    masks_shuffled = masks.dimshuffle(1, 0)

    # initials should be energies_shuffles[0, :, -1, :]
    init_label = T.cast(T.fill(energies[:, 0, 0, 0], -1), 'int32')
    energy_time0 = energies_shuffled[0]
    target_time0 = targets_shuffled[0]
    initials = [energies_shuffled[0, :, -1, :], target_time0,
                energy_time0[T.arange(energy_time0.shape[0]), init_label, target_time0]]
    [partitions, _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials,
                                                      sequences=[energies_shuffled[1:], targets_shuffled[1:],
                                                                 masks_shuffled[1:]])
    partition = partitions[-1]
    target_energy = target_energies[-1]
    loss = theano_logsumexp(partition, axis=1) - target_energy
    return loss


def crf_accuracy(energies, targets):
    """
    decode crf and compute accuracy
    :param energies: Theano 4D tensor
        energies of each step. the shape is [batch_size, n_time_steps, num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    assert energies.ndim == 4
    assert targets.ndim == 2

    def inner_function(energies_one_step, prior_pi, prior_pointer):
        """

        :param energies_one_step: [batch_size, t, t]
        :param prior_pi: [batch_size, t]
        :param prior_pointer: [batch_size, t]
        :return:
        """
        prior_pi_shuffled = prior_pi.dimshuffle(0, 1, 'x')
        pi_t = T.max(prior_pi_shuffled + energies_one_step, axis=1)
        pointer_t = T.argmax(prior_pi_shuffled + energies_one_step, axis=1)

        return [pi_t, pointer_t]

    def back_pointer(pointer, pointer_tp1):
        """

        :param pointer: [batch, t]
        :param point_tp1: [batch,]
        :return:
        """
        return pointer[T.arange(pointer.shape[0]), pointer_tp1]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    # pi at time 0 is the last rwo at time 0. but we need to remove the last column which is the pad symbol.
    pi_time0 = energies_shuffled[0, :, -1, :-1]

    # the last row and column is the tag for pad symbol. reduce these two dimensions by 1 to remove that.
    # now the shape of energies_shuffled is [n_time_steps, b_batch, t, t] where t = num_labels - 1.
    energies_shuffled = energies_shuffled[:, :, :-1, :-1]

    initials = [pi_time0, T.cast(T.fill(pi_time0, -1), 'int64')]

    [pis, pointers], _ = theano.scan(fn=inner_function, outputs_info=initials, sequences=[energies_shuffled[1:]])
    pi_n = pis[-1]
    pointer_n = T.argmax(pi_n, axis=1)

    back_pointers, _ = theano.scan(fn=back_pointer, outputs_info=pointer_n, sequences=[pointers], go_backwards=True)

    # prediction shape [batch_size, length]
    prediction_revered = T.concatenate([pointer_n.dimshuffle(0, 'x'), back_pointers.dimshuffle(1, 0)], axis=1)
    prediction = prediction_revered[:, T.arange(prediction_revered.shape[1] - 1, -1, -1)]
    return prediction, T.eq(prediction, targets)
