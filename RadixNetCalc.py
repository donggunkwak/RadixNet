from __future__ import division
import numpy as np
import time
import fractions

def emr_net(radix_lists):
    """ Simplified version of EMR_Net, creates an X-net sparse network topology from the input parameters
    using Ryan's Extended Mixed-Radix method.

    Inputs:

    radix_lists - A list of lists of radices to be used for the [possibly
    extended] mixed radix system. The number of entries is the number of layers
    in the network W that is output.

    Returns: 

    layers - A list of numpy arrays where the i'th entry is the weight matrix
    W_i for the i'th layer.

    """

    # print('radix lists: ' + str(radix_lists))
    num_layers = sum(len(radix_list) for radix_list in radix_lists)
    # this is the number of neurons per layer.
    num_neurons = np.prod(radix_lists[0])

    # for all but last radix list, product of radices must equal num_neurons
    if not np.all( [num_neurons == np.prod(radix_list)
            for radix_list in radix_lists[:-1]]):
        raise ValueError('Product of radices for each radix list must equal'
            + 'number of neurons from first radix list for all but last radix'
            + 'list')
    # for last radix list, product of radices must divide num_neurons
    if num_neurons % np.prod(radix_lists[-1]) != 0:
        raise ValueError('Product of radices for last radix list must divide'
             + 'number of neurons from first radix list')

    # the actual math part
    # The N x N identity matrix, used for constructing permutation matrices
    I = np.identity(num_neurons)

    layers = [] # the output list containing W_i's
    # make layers
    for radix_list in radix_lists:
        place_value = 1
        for radix in radix_list:
            layer = np.sum( [np.roll(I, -j * place_value, axis=1)
                for j in range(radix)], axis=0)
            layers.append(layer)
            place_value *= radix

    return layers


def kemr_net(radix_lists, B):
    """ Simplified version of kemr_net. Creates a sparse network topology using the Kronecker/EMR method. First
    calls extended_mixed_radix_network(radix_lists). This network is then
    expanded via kronecker product to fill the fully connected structure
    defined by B.

    Inputs:

    radix_lists - A list of lists of radices to be used for the [possibly
    extended] mixed radix system. The number of entries is the number of layers
    in the network W that is output.

    B - a list of integers giving the number of neurons per layer of the
    superstructure into which the EMR network is being Kroneckered.

    Returns:

    layers - A list of numpy arrays where the i'th entry is the weight matrix
    W_i for the i'th layer.
    """
    # print('radix lists: ' + str(radix_lists))
    # print('B: ' + str(B))

    emr_layers = emr_net(radix_lists)
    num_layers = len(emr_layers)

    # check valid input for B
    if len(B) - 1 != num_layers:
        raise  ValueError('Incorrect lengths of N, B parameters; len(B) should'
            + 'be one more than num_layers')

    # make the B graph to kronecker with emr_layers
    B_layers = [np.ones((B[i], B[i+1])) for i in range(len(B)-1)]

    expanded_layers = [np.kron(B_layer, emr_layer)
        for (B_layer, emr_layer) in zip(B_layers, emr_layers)]

    return expanded_layers