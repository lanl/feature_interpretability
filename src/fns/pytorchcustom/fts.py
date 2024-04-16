# PYTORCH FEATURES SCRIPT
"""
Contains functions to extract features from a pytorch neural network
"""

#############################
## Packages
#############################
import os
import sys
import numpy as np
import torch
import torchvision.models.feature_extraction as ftex

import fns

#############################
## Extract Features
#############################
def feature_extractor(model, lay: str, model_in, norm: str, device: torch.device):
	""" Function to extract the features from a given layer of a model

		Args:
			model (loaded pytorch model) 
			lay (str): name of a layer in model
			model_in (varies): correctly formatted model input
			norm (str): {'ft01', 'all01', 'none'} 
						a string to indicate which normalization methodology to use;
			device (torch.device): device index to select

		Returns:
			ft_mat (np.ndarray[(any, any, any), float]): an array of all features extracted from a given layer; 
								the first two dimensions are the size of the feature; 
								the last dimension is the number of features in a layer

		See Also:
			:func:`fns.mat.normalize_mat` for information about choices for *norm*
    """
    ## Check Norm is Valid
	assert norm in ['ft01', 'all01', 'none'], "norm must be one of {'ft01', 'all01', 'none'}."

	#Extract Features
	in_y, in_x = model_in.shape
	model_in = torch.tensor(model_in.reshape((1, 1, in_y, in_x))).to(torch.float32).to(device)
	extractor = ftex.create_feature_extractor(model, return_nodes={lay: lay})
	fts = extractor(model_in)[lay].detach().numpy()[0, :, :, :]

	#Fix dimensions convention
	n_fts, ft_y, ft_x = fts.shape
	features = np.zeros((ft_y, ft_x, n_fts))
	for i in range(n_fts):
		features[:, :, i] = fts[i, :, :]

	ft_mat = fns.mat.normalize_mat(features, norm)

	return ft_mat


#############################
## Parse Features
#############################
def parse_features(model, features: list[str]):
	""" Function to make a list of the features to plot

		Prints error message and exits program for features = ['Grid']

	    Args:
	        model (loaded pytorch model) 
			features (list[str]): list of features to plot, starting at feature 1
			
	    Returns:
			n_features (int): how may features to plot
			features (list[int]): list of features to plot, starting at feature 0
    """
	if features == ['Grid']:
		print("Features=Grid is not a valid selection for this program; Use 'All' or a list of integers.\n")
		sys.exit()
	elif features == ['All']:
		n_features = model.features
		features = np.arange(n_features)
	else:
		n_features = np.size(features)
		features = [int(i)-1 for i in features]
	return n_features, features