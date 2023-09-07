# PYTORCH CALICO MODEL FUNCTIONS
"""
Contains functions to create a calico network and do prints/checks on the calcio network inputs
"""

#############################
## Packages
#############################
import os
import sys
import typing
import copy
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision.models.feature_extraction as ftex

sys.path.insert(0, os.path.abspath('../../'))
import fns

#############################
## Calico Checks
#############################
def check_calico_layer(model, lay: str, branch: str='None', catlay: str='None'):
	""" Function that checks if the layer is a valid selection for the split layer

		Args:
			model (loaded pytorch model)
			lay (str): name of layer to test
			branch (str): key used to identify which layers are on the secondary branch; 
							use 'None' if the model only has one branch
			catlay (str): name of layer where the branches of the model are concatenated
							use 'None' if the model only has one branch

		Returns:
			No Return Objects
	"""
	layer_names=ftex.get_graph_node_names(model)[0][1:]

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


def check_calico_features(model, features: str):
	""" Function that checks if number of features requested are available from a layer

        Args:
            model (loaded pytorch model)
            features (str): an integer; features the calico model scales
        
        Returns:
			No Return Objects
    """
	if np.size(features) != 1:
		raise ValueError('The calico network requires that only one feature be selected.\nUse -PT or --PRINT_FEATURES to show how mnay features are extracted from the given layer.')
	else:
		ft = features[0]
		n_fts = model.Nfilters
		if not ft.isdigit():
			raise ValueError('Feature given is not an integer.')
		elif int(ft) > n_fts:
			raise ValueError('Feature number given exceeds features from layer in model. \nUse -PT or --PRINT_FEATURES to show how mnay features are extracted from the given layer.')

#############################
## Calico Layer Functions
#############################
def eval_layer(model, lay: str, x: typing.Union[torch.FloatTensor, torch.cuda.FloatTensor]):
	""" Function to evaluate a pytorch layer given it's name

		Args:
			model (loaded pytorch model)
			lay (str): name of layer to evaluate
			x (typing.Union[torch.FloatTensor, torch.cuda.FloatTensor]): input to layer

		Returns:
			x (torch.Tensor): output of layer
	"""
	## Evaluate layer in a ModuleList
	if '.' in lay:
		name = lay.split('.')[0]
		number = int(lay.split('.')[-1])
		x = getattr(model, name)[number](x)

	## Evaluate predefined torch layer
	elif lay == 'flatten':
		x = x.flatten()

	## Evaluate custom layer
	else:
		x = getattr(model, lay)(x)

	return x

