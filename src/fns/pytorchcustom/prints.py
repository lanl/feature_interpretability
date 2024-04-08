# PYTORCH PRINTS SCRIPT
"""
Contains functions to print out lists of options for model-related input arguments
"""

#############################
## Packages
#############################
import glob
import numpy as np
import torch
import torchvision.models.feature_extraction as ftex

import fns.pytorchcustom as pytc

#############################
## Print Functions
#############################
def print_layers(model):
	""" Function that prints a list of layer names in a model

		Args:
			model (loaded pytorch model)
		
		Returns:
			No Return Objects
	"""
	layers = ftex.get_graph_node_names(model)[0][1:]
	print("Layer names: \n")
	for l in layers:
		print(l)
	print("Layer names printed. \n")

def print_features(model):
	""" Function that prints how many features are extracted from a layer of a model

        Args:
            model (loaded pytorch model)
        
        Returns:
			No Return Objects
	"""
	n_fts = model.features
	print("Model has "+str(n_fts)+" features. \n")

