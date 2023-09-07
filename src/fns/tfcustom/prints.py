# TENSORFLOW PRINTS SCRIPT
"""
Contains functions to print out lists of options for model-related input arguments
"""

#############################
## Packages
#############################
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import fns.tfcustom as tfc

#############################
## Print Functions
#############################
def print_layers(model):
	""" Function that prints a list of layer names in a model

        Args:
            model (loaded keras model)

        Returns:
			No Return Objects
    """
	print("Layer names: \n")
	for layer in model.layers:
		print(layer.name)
	print("Layer names printed. \n")

def print_features(model, lay: str):
	""" Function that prints how many features are extracted from a layer of a model

        Args:
            model (loaded keras model)
            lay (str): name of layer to get features from
            
        Returns:
			No Return Objects
	"""
	quit = tfc.checks.check_layer(model, lay)
	if quit:
		print('Number of features cannot be printed because layer name is not in the model.\n')
	else:
		n_fts = model.get_layer(name=lay).output_shape[-1]
		print("Layer "+lay+" has "+str(n_fts)+" features. \n")

