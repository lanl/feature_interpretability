# TENSORFLOW CHECKS SCRIPT
"""
Contains functions to check that the model-related input arguments passed are valid
"""


#############################
## Packages
#############################
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras


#############################
## Check Functions
#############################
def check_layer(model, lay: str):
	""" Function that checks if a layer name is in the model

		Args:
			model (loaded keras model)
			lay (str): name of layer to test

		Returns:
			No Return Objects
	"""
	quit = False
	layer_names=[layer.name for layer in model.layers]
	if lay=='None':
		raise ValueError('Layer name is required as input; use -L or --LAYER_NAME.')
	elif lay not in layer_names:
		raise ValueError('Layer name given is not present in the model. Use -PL or --PRINT_LAYERS to show list of valid layer names.')

def check_features(model, lay: str, features: list[str]):
	""" Function that checks if number of features requested are available from a layer

        Args:
            model (loaded keras model)
            lay (str): name of layer where features will be extracted from
            features (list[str]): should be ['Grid'], ['All'], or a list of integers;
            				features the script plans on extracting
        
        Returns:
			No Return Objects
    """
	quit = False
	if features == ['Grid']:
		pass
	elif features == ['All']:
		pass
	else:
		check_layer(model, lay)
		n_fts = model.get_layer(name=lay).output_shape[-1]
		for ft in features:
			if not ft.isdigit():
				raise ValueError('Feature given is not an integer.')
			elif int(ft) > n_fts:
				raise ValueError('Feature number given exceeds features from layer in model. \nUse -PT or --PRINT_FEATURES to show how mnay features are extracted from the given layer.')
