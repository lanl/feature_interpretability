# TENSORFLOW CALICO MODEL FUNCTIONS
"""
Contains functions to create a calico network and do prints/checks on the calcio network inputs
"""

#############################
## Packages
#############################
import os
import sys
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, models, layers, activations

sys.path.insert(0, os.path.abspath('../../'))
import fns

#############################
## Calico Checks
#############################
def check_calico_layer(model, lay: str, branch: str='None', catlay: str='None'):
	""" Function that checks if the layer is a valid selection for the split layer

		Args:
			model (loaded keras model)
			lay (str): name of layer to test
			branch (str): key used to identify which layers are on the secondary branch; 
							use 'None' if the model only has one branch
			catlay (str): name of layer where the branches of the model are concatenated
							use 'None' if the model only has one branch

		Returns:
			No Return Objects
	"""
	layer_names=[layer.name for layer in model.layers]

	## Check if the layer name is valid
	if lay=='None': ## Check that layer name was given
		raise ValueError('Layer name is required as input; use -L or --LAYER_NAME.')
	elif lay not in layer_names: ## Check that the layer name is in the model
		raise ValueError('Layer name given is not present in the model. Use -PL or --PRINT_LAYERS to show list of valid layer names.')
	
	## Check if layer is on the branch
	if branch != 'None':
		## Check if branch is in the model
		if fns.misc.search_list(layer_names, branch) == []:
			raise ValueError('Branch '+branch+' is not present in the model. Use -PL or --PRINT_LAYERS to show list of valid layer names.')

		## Check if catlay is in the model
		if catlay == 'None':
			raise ValueError('If branch is given, concatenation layer cannot be "None"; Use -PL or --PRINT_LAYERS to show list of valid layer names.')
		elif catlay not in layer_names:
			raise ValueError('Concatenation layer name given is not present in the model. Use -PL or --PRINT_LAYERS to show list of valid layer names.')

		## Check split layer is not on the branch
		if branch in lay:
			raise ValueError('The split layer cannot be on the '+branch+' branch. Use -PL or --PRINT_LAYERS to show list of layer names.')

		## Check if the layer is before the concatenate layer
		layerIDX = layer_names.index(lay)
		catIDX = layer_names.index(catlay)
		if layerIDX >= catIDX:
			raise ValueError('The split layer must be above the concatenation layer. Use -PL or --PRINT_LAYERS to show list of layer names in order.')

	## Check that if there is no branch, there is no concatenate layer
	if branch =='None':
		if catlay != 'None':
			raise ValueError('If no branch is given (branch = "None"), there cannot be a concatenation layer. Use catlay="None", or specify branch.')


def check_calico_features(model, lay: str, features: str):
	""" Function that checks if number of features requested are available from a layer

        Args:
            model (loaded keras model)
            lay (str): name of layer where features will be extracted from
            features (str): an integer; features the calico model scales
        
        Returns:
			No Return Objects
    """
	if np.size(features) != 1:
		raise ValueError('The calico network requires that only one feature be selected.\nUse -PT or --PRINT_FEATURES to show how mnay features are extracted from the given layer.')
	else:
		ft = features[0]
		n_fts = model.get_layer(name=lay).output_shape[-1]
		if not ft.isdigit():
			raise ValueError('Feature given is not an integer.')
		elif int(ft) > n_fts:
			raise ValueError('Feature number given exceeds features from layer in model. \nUse -PT or --PRINT_FEATURES to show how mnay features are extracted from the given layer.')

#############################
## Calico Layer Functions
#############################
def layer_copy(cnnlayer, tag: str):
	""" Function to create a copy of a keras layer object

		Based on the layer type and configuration

		Does NOT copy trained layer weights

		Args: 
			cnnlayer (keras.layers): layer to make a copy of
			tag (str): string "tag" to be appended to the name of the copied layer

		Returns:
			cnnlayer2 (keras.layer): copy of cnnlayer with same type and configuration
	"""
	layer_type = cnnlayer.__class__.__name__
	layer_config = cnnlayer.get_config()
	cnnlayer2 = eval('layers.'+layer_type+'.from_config(layer_config)')
	cnnlayer2._name = cnnlayer.name + tag 
	return cnnlayer2

def idy(x): return x
"""This is just the identity function. 
You need it to make an identity layer in Keras that does not require installing the nightly build"""

