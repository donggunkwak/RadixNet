from __future__ import division
import numpy as np
import time
import math
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import fractions
import itertools

class Mask(tf.Module):
    """
    A class used to create tensorflow masks that represent the RadiXNet weight masks. 
    They are used to be applied at every call of a RadixLayer, specifically at every step of training.
    ...
    Attributes
    ----------
    val : None or tf.Variable
        keeps track of what the mask is in a tensorflow format, kept as None to start with 
        but is changed the first time it's called
    layerval : np.array
        input of what the mask should be in a numpy array format.
    """
    def __init__(self,layerval:np.array):
        """
        Used to construct a Mask, but doesn't have the correct shape yet.
        Inputs:
        layerval : np.array
            input of what the mask should be in a numpy array format
        """
        self.val=None
        self.layerval = layerval
    @tf.function
    def __call__(self):
        """
        Used whenever you want to actually use the mask
        Example of usage:
            mask = Mask(layerval) #create the mask
            tf.math.multiply(kernel, mask()) #calling the mask
        """
        if self.val is None:
            #arr = np.zeros(self.input_shape[1:]+(self.num_outputs,))
            arr = self.layerval
            tensor = tf.constant(arr, dtype= tf.float32)
            self.val = tf.Variable(tensor)
        return self.val

class RadixLayer(layers.Layer):
    """
    A class used to create layers of a RadiXNet, made for an easier implementation of RadiXNets
    ...
    Attributes
    ----------
    num_outputs : int
        the number of outputs that the layer should have, or how many neurons are in the next layer
    layerval : numpy array
        input of what the mask should be in a numpy array format
    activation : function
        activation function used
    kernel : tensorflow Variable
        the values of the layer that will be trained
    """
    def __init__(self, num_outputs:int, layerval:np.array, activation = tf.nn.relu):
        """
        Used to construct a RadixLayer
        Inputs:
        num_outputs : int
            number of outputs that layer should have
        layerval : numpy array
            input of what the mask should be
        activation : function
            activation type, defaults to relu function
        """
        super(RadixLayer,self).__init__()
        self.num_outputs = num_outputs
        self.layerval = layerval
        self.activation = activation
    def build(self, input_shape:tuple):
        """
        Used when first creating and calling a RadixLayer, creates the kernel variable
        Inputs:
        input_shape : tuple
            Size of inputs, this is needed in the build function but is currently not used because our kernel 
            shapes should take on the shape of the radix layer
        """
        self.kernel = self.add_weight("kernel", shape = self.layerval.shape, initializer='random_normal', trainable=True)
    def call(self, inputs:tf.Tensor):
        """
        Used whenever the RadixLayer is called during iterations
        Inputs:
        inputs: tensor
            What will be going into the RadiXLayer
        Outputs:
            returns the inputs matrix multiplied with a masked version of the kernel (element wise multiplication between kernel and mask)
        """
        mask = Mask(self.layerval)
        masked = tf.math.multiply(self.kernel,mask())
        #extrazero =  tf.zeros(shape=(self.layerval.shape[0]-inputs.shape[1]), dtype=tf.int32)
        #inputs = tf.concat(inputs,extrazero)
        try:
            return self.activation(tf.matmul(inputs,masked))
        except:
            return None

class CustomModel(tf.keras.Model):
    """
    Class used to customize whatever kind of model the user would like
    ...
    Attributes
    ----------
    layers : RadixLayer
        However many RadixLayers the user would like, take in from rlayers, which is done with kemr_net and trimming to the desired_layer_sizes size
    """
    def __init__(self, topology:list):
        """
        Initialization of layers (choose how many layers you want)
        Inputs:
        topology : list of numpy arrays
            fully created topology from kemr_net and trimmed to fit desired_layer_sizes
        """
        super(CustomModel,self).__init__()
        self.layer1 = RadixLayer(topology[0].shape[-1], topology[0],tf.nn.relu)
        self.layer2 = RadixLayer(topology[1].shape[-1], topology[1],tf.nn.relu)
        self.layer3 = RadixLayer(topology[2].shape[-1], topology[2],tf.nn.softmax)
    def call(self, input_tensor:tf.Tensor):
        """
        Actually does the math and calculates the result of an input through the network.
        Customize what activation functions are used
        Inputs:
        input_tensor : tensor
            the input
        Outputs:
            The result of the calculation
        """
        x = self.layer1(input_tensor)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

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

#def ones(shape):
    #return [[1 for i in range(int(shape[0]))] for j in range(int(shape[1]))]

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

    expanded_layers = [np.kron(B_layer, emr_layer) for (B_layer, emr_layer) in zip(B_layers, emr_layers)]

    return expanded_layers

def generate_network_structure(model):
    """Generates the desired network structure based on model layers."""

    desired_network_structure = []

    for layer in model.layers:
        # Check if the layer is a Dense layer
        if layer.__class__.__name__ == "Dense":
            for weights in layer.get_weights():
                # Extract dimensions from weights shape
                for dim in weights.shape:
                    # Append dimension if not already in the structure
                    if dim not in desired_network_structure:
                        desired_network_structure.append(dim)

    return desired_network_structure

def calculate_sparsity(topology):
    """
    Calculate the sparsity of a topology list

    Parameters:
        topology: a list of 2D numpy arrays.

    Returns:
        float: The sparsity value of the topology ranging from 0 to 1.
    """
    if isinstance(topology, list):
        return 1 - sum(np.count_nonzero(layer) for layer in topology) / sum(np.prod(layer.shape) for layer in topology)
    elif isinstance(topology, tf.keras.Model):
         return 1 - sum(np.count_nonzero(layer) for layer in topology.layers) / sum(np.prod(layer.shape) for layer in topology.layers)

def permutations_with_replacement(iterable, r):
    """
    Generate permutations with replacement from the given iterable.

    Parameters:
        iterable (iterable): The input iterable to generate permutations from.
        r (int): The length of each permutation.

    Returns:
        list: List of permutations as lists.
    """
    return [list(perm) for perm in itertools.product(iterable, repeat=r)]

def generate_sublists(input_list):
    """
    Generate all sublist permutations of a given input list.

    Parameters:
        input_list (list): The input list.

    Returns:
        list: List containing all sublist permutations of the input list.
    """
    if not input_list:
        return [[]]

    new_sublists = []
    
    rest_sublists = generate_sublists(input_list[1:])
    for sublist in rest_sublists:
        if sublist and isinstance(sublist[0], list):
            new_sublists.append([[input_list[0]] + sublist[0]] + sublist[1:])
        new_sublists.append([[input_list[0]]] + sublist)
    return new_sublists

def generate_permutations(desired_network_structure, factors=None):
    """
    Generate valid permutations for the desired network structure.

    Parameters:
        desired_network_structure (list): The desired network structure.

    Returns:
        list: List of valid permutations.
    """
    factors = list(range(1,11)) if not factors else factors #defaults to [1,...,10] if not specified
    all_permutations = permutations_with_replacement(factors, len(desired_network_structure) - 1)
    all_sublist_permutations = []
    for permutation in all_permutations:
        all_sublist_permutations.extend(generate_sublists(permutation))
    #print(all_sublist_permutations)
    valid_radix_list_permutations = []
    for permutation in all_sublist_permutations:
        first_prod = np.prod(permutation[0])
        if all(np.prod(sub) == first_prod for sub in permutation[:-1]) and first_prod % np.prod(permutation[-1]) == 0:
            valid_radix_list_permutations.append(permutation)
    return valid_radix_list_permutations

def build_radix_nets(permutations, desired_network_structure):
    """
    Build radix networks from the given permutations.

    Parameters:
        permutations (list): List of valid permutations.
        desired_network_structure (list): The desired network structure.

    Returns:
        list: List of tuples, each containing a radix list permutation, list of kronecker factors "B", and the corresponding
        topology to be used as a sparse mask.
    """
    masks = []
    for permutation in permutations:
        scale = np.prod(np.array(permutation[0]))
        B = [math.ceil(n / scale) for n in desired_network_structure]
        radix_net = kemr_net(permutation, B)
        for i in range(len(radix_net)):
            if radix_net[i].shape[0]>desired_network_structure[i] and radix_net[i].shape[1]>desired_network_structure[i+1]:
                radix_net[i] = radix_net[i][:desired_network_structure[i],:desired_network_structure[i+1]]
            if radix_net[i].shape[0]>desired_network_structure[i]:
                radix_net[i] = radix_net[i][:desired_network_structure[i],:]
            if radix_net[i].shape[1]>desired_network_structure[i+1]:
                radix_net[i] = radix_net[i][:,:desired_network_structure[i+1]]
        masks.append(radix_net)
    # ret = list(zip(permutations, Bs, masks))
    if len(permutations) == 1:
        masks = masks[0] #assuming input is a radix list, returns the kroneckered radix net topology for that
    return masks

def categorize_by_sparsity(radix_perms):
    """
    Categorize a list of radix networks by their sparsity.

    Parameters:
        lst (list): List of tuples, each containing a permutation, B, and the corresponding network.

    Returns:
        dict: Dictionary containing lists of networks categorized by sparsity.
    """
    sparsity_dict = {i/10: [] for i in range(10)}
    for radix_list, B, topology in radix_perms[::2]: #output of build_radix_nets has copies of each radix_net.
        sparsity = calculate_sparsity(topology)
        category = round(sparsity // 0.1 / 10, 1)
        sparsity_dict[category].append((radix_list, B, sparsity))
    return sparsity_dict

"""def sort_by_sparsity(radix_perms):
    
    Categorize a list of radix networks by their sparsity.

    Parameters:
        lst (list): List of tuples, each containing a permutation, B, and the corresponding network.

    Returns:
        dict: List of tuples containing a radix list permutation, B, and sparsity, with the tuples sorted
        by their sparsity.
    
    for radix_list, B, topology in radix_perms[::2]:
        sparsity = calculate_sparsity(topology)
    return sorted(radix_perms, key = lambda x: x[2])"""

def findSimpleModels(desired:list):
    """
    Finds a bunch of simple models with structure [[a,b],[x,x,...]]
    Inputs:
    desired: list
        The desired network structures
    Returns:
    tuple - (sparsemodels,greater90), dictionaries that map sparsity to models, with the second one only having models with sparsity >=0.9
    """
    sparsemodels = dict()
    greater90 = dict()
    N = [[],[]]
    for i in range(1,16):
        N[0].append(i)
        for j in range(1,16):
            N[0].append(j)
            B= []
            for num in desired:
                B.append(math.ceil(num/(i*j)))
            for k in range(1,i*j+1):
                for l in range(len(desired)-3):
                    N[1].append(k)
                #print(N,B)
                try:
                    curlayers = kemr_net(N,B)
                    model = ((tuple(N[0]),tuple(N[1])),tuple(B))
                    spars = calculate_sparsity(curlayers)
                    sparsemodels[spars] = model
                    if spars>=0.9:
                        greater90[spars] = model
                except:
                    pass
                N[1].clear()
            N[0].pop()
        N[0].pop()
    return (sparsemodels, greater90)

def findModelWithSparsity(sparsity:float,sparsemodels:dict):
    """
    Finds a model structure closest to the desired sparsity
    Inputs:
    sparsity:float
        the sparsity of the model you want
    sparsemodels:dict
        all sparse models (found in findSimpleModels)
    Returns:
        curmodel - tuple, input to be fed into createModel, tuple of lists, with the first element being the list of radices and the second being the Kronecker numbers
    """
    temp =min(sparsemodels.keys(), key=lambda x: abs(x - sparsity))
    curmodel = sparsemodels[temp]
    return curmodel

def createModel(curmodel:tuple):
    """
    Given a model with a list of radices and Kronecker numbers, creates a CustomModel Object
    Inputs:
    curmodel:tuple
        tuple of lists, with the first element being the list of radices and the second being the Kronecker numbers
    Returns:
        CustomModel object of a RadixNet, the radix-net made by the model radices and kronecker
    """
    N = curmodel[0]
    B = curmodel[1]
    #print(N, B)
    desired = [784,300,100,10]
    rlayers = kemr_net(N,B)
    for i in range(len(rlayers)):
        if rlayers[i].shape[0]>desired[i]:
            rlayers[i] = rlayers[i][:desired[i],:]
        if rlayers[i].shape[1]>desired[i+1]:
            rlayers[i] = rlayers[i][:,:desired[i+1]]
    return CustomModel(rlayers)
