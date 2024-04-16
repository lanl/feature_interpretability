# CALICO MODEL DEFINITION FOR SINGLE BRANCH PYTORCH NESTED CYLINDER MODELS
"""
Defines the calico model for the single branch pytorch nested cylinder models

Execution will print unit test information, perform unit tests, and print the results to the terminal.

Input Line:
``python pyt_nestedcyl_calico_model.py -M ../../../examples/pyt_nestedcyl/trained_rho2PTW_model.pth -IF rho -IN ../../../examples/pyt_nestedcyl/data/nc231213_Sn_id0643_pvi_idx00112.npz -DF ../../../examples/pyt_nestedcyl/nestedcyl_design_file.csv -L interp_module.interpActivations.10``
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
import torch.nn as nn
import torchvision.models.feature_extraction as ftex
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, os.path.abspath('../../'))
import fns
import fns.setup as setup
import fns.nestedcylinderdata as nc
import fns.pytorchcustom as pytc
import fns.pytorchcustom.calico as pytcalico

#############################
## Recurrsive Get Attribute
#############################
def recurr_getattr(obj, attr: str, default=None):
	""" Recursive getattr function; allowes 'attr' to contain '.'

		| Taken from: https://programanddesign.com/python-2/recursive-getsethas-attr/

		Args:
			obj (object)
			attr (str): attribute to obtain; can contain '.', which would be a recurrsive attribute
			default: value that is returned when the named attribute is not found

		Returns:
			Attribute of Object

	"""
	try: left, right = attr.split('.', 1)
	except: return getattr(obj, attr, default)
	return recurr_getattr(getattr(obj, left), right, default)

#############################
## Create Calico Model
#############################
class make_calico(nn.Module):
	""" Pytorch model class that creates a "calico network" from an existing nested cylinder neural net
		
		Args:
			model(loaded pytorch model): model to copy into calico
			lay (str): the name of the layer in model that will become the split layer
			ftIDX (int): index of the feature to scale; feature w.r.t. the derivative is taken
			dScale (float): derivative scaling factor
	"""

	def __init__(self, model,
						lay: str,
						ftIDX: int=0,
						dScale: float=0.01):

		## Call the parent constructor
		super(make_calico, self).__init__()

		## Define Properties
		self.oldlayers = ftex.get_graph_node_names(model)[1][1:]
		self.lay = lay
		self.layIDX = self.oldlayers.index(lay)
		self.ftIDX = ftIDX
		self.dScale = dScale

		## Copy layers from original model to calico model
		for layer in self.oldlayers:
			## Copy layers in ModuleLists
			if '.' in layer:
				split_name = layer.split('.')
				if split_name[-1].isdigit():
					number = int(split_name[-1])
					old_layer_name = '.'.join(split_name[0:-1])
					new_layer_name = '_'.join(split_name[0:-1])
					if number == 0: #If it's the first layer in a ModuleList, define a ModuleList
						setattr(self, new_layer_name, nn.ModuleList())
					getattr(self, new_layer_name).append(copy.deepcopy(recurr_getattr(model, old_layer_name)[number]))
				else:
					new_layer_name = '_'.join(split_name)
					setattr(self, new_layer_name, copy.deepcopy(recurr_getattr(model, layer)))

			## Skip layers that are predefined in pytorch
			elif layer == 'flatten': #skip flatten layer in the init
				pass

			## Copy custom layers
			else:
				setattr(self, layer, copy.deepcopy(recurr_getattr(model, layer)))
				
			
	def forward(self, x: typing.Union[torch.FloatTensor, torch.cuda.FloatTensor]):
		""" Forward pass of pytorch neural network class

			Args:
				x (typing.Union[torch.FloatTensor, torch.cuda.FloatTensor]): input to layer

			Returns:
				splitx (torch.tensor[float]): prediction from original model
				diff (torch.tensor[float]): difference in prediction between original model and calico model
		"""

		for k, layer in enumerate(self.oldlayers):
			## Before the split layer
			if k < self.layIDX:
				x = pytcalico.eval_layer(self, layer, x)

			## At the split layer
			elif k == self.layIDX: 
				## Original branch
				splitx = pytcalico.eval_layer(self, layer, x)

				## Multiply Branch
				y = pytcalico.eval_layer(self, layer, x)
				y[:, self.ftIDX, :, :] = torch.mul(y[:, self.ftIDX, :, :], 1+self.dScale) 

			else: #after the split layer
				splitx = pytcalico.eval_layer(self, layer, splitx)
				y = pytcalico.eval_layer(self, layer, y)

		diff = torch.sub(y, splitx)

		return(splitx, diff)

def load_calico(model,
				checkpoint: str, 
				device: torch.device,
				lay: str,
				ftIDX: int=0,
				dScale: float=0.01,):
	""" Function to create a pytorch nested cylinder calico model and load in the correct weights

		Args:
			model (loaded pytorch model): model to copy into calico
			checkpoint(str): path to model checkpoint with orignal model weights
			device (torch.device): device index to select
			lay (str): the name of the layer in model that will become the split layer
			ftIDX (int): index of the feature to scale; feature w.r.t. the derivative is taken
			dScale (float): derivative scaling factor

		Returns:
			calico (pytorch model): calico network
	"""
	## Make calico model
	calico = make_calico(model, lay, ftIDX, dScale).to(device)

	## Load in Weights
	loadedCheckpoint = torch.load(checkpoint, map_location=device)
	calico.load_state_dict(loadedCheckpoint['modelState'], strict=False)

	return calico


#############################
## Set Up Parser
#############################
def calico_model_parser():
	descript_str = 'Creates and tests a calcio network given an input model'

	parser = argparse.ArgumentParser(prog='Calico Functions',
									 description=descript_str,
									 fromfile_prefix_chars='@')
	## Model Imports
	setup.args.model(parser) # -M
	setup.args.input_field(parser) # -IF
	setup.args.input_npz(parser) # -IN
	setup.args.design_file(parser) # -DF

	## Print Toggles
	setup.args.print_layers(parser) # -PL
	setup.args.print_fields(parser) # -PF

	## Layer Properties
	setup.args.layer(parser) # -L

	return parser

parser = calico_model_parser()

#############################
## Unit Tests for Calico Network
#############################
if __name__ == '__main__':

	## Parse Args
	args = parser.parse_args()

	## Model Imports
	model_path = args.MODEL
	in_field = args.INPUT_FIELD
	input_path = args.INPUT_NPZ
	design_path = args.DESIGN_FILE

	## Print Toggles
	PRINT_LAYERS = args.PRINT_LAYERS
	PRINT_FIELDS = args.PRINT_FIELDS

	## Layer Properties
	lay = args.LAYER

	#############################
	## Data Processing
	#############################
	## Load npz file
	npz = np.load(input_path)

	## Data Prints
	if PRINT_FIELDS: 
		setup.data_prints.print_fields(npz)
		sys.exit()

	## Data Checks
	setup.data_checks.check_fields(npz, [in_field])

	#############################
	## Model Processing
	#############################
	## Load Model
	import fns.pytorchcustom.field2PTW_model_definition as field2PTW_model
	model = field2PTW_model.field2PTW(img_size = (1, 1700, 500),
											size_threshold = (8, 8),
											kernel = 5,
											features = 12, 
											interp_depth = 12,
											conv_onlyweights = True,
											batchnorm_onlybias = False,
											act_layer = torch.nn.GELU,
											hidden_features = 20)
	checkpoint = torch.load(model_path, map_location=device)
	model.load_state_dict(checkpoint["modelState"])

	## Model Prints
	if PRINT_LAYERS: 
		pytc.prints.print_layers(model)
		sys.exit()

	## Model Checks
	pytcalico.check_calico_layer(model, lay, branch='None', catlay='None')

	#############################
	## Print Unit Test Information 
	#############################
	print('This script runs unit tests for the Calico model.\n')
	print('-----\n')
	print('Unit Test for the Difference Output:\n')
	print('The Calico difference output is the difference between the original branch and the multiply branch.\n')
	print('This test sets the dScale value to zero, meaning the multiply branch is scaled by 1. Therefore, the difference should be zero.\n')
	print('-----\n')
	print('Unit Test for the Prediction Output:\n')
	print('The Calico prediction output is the output from the origial branch.\n')
	print('This test compares the Calico prediction output to the original model prediction output. The difference should be zero.\n')
	print('-----\n')

	#############################
	## Make & Test Calico Model
	#############################
	## Make Calico Model
	calico = load_calico(model = model, 
						checkpoint = model_path, 
						device = device,
						lay = lay,
						ftIDX = 0,
						dScale = 0.0,)

	## Making the Model Inputs
	img_input = nc.process.get_field(npz, in_field)
	in_y, in_x = img_input.shape
	img_input = img_input.reshape((1, 1, in_y, in_x))
	img_input = torch.tensor(img_input).to(torch.float32).to(device)

	## Making Predictions
	model.eval()
	model_pred = model.forward(img_input).item()

	calico.eval()
	calico_pred = calico.forward(img_input)
	print('.\n.\n.\n')

	## Test Predictions
	pred_output = calico_pred[0].item()
	diff_output = calico_pred[1].item()
	pred_diff = pred_output - model_pred

	#############################
	## Print Unit Test Results
	#############################
	print('Calico Network Unit Test Results\n')
	print('-----\n')
	print('Checking Calico Difference Output:\n')
	print(diff_output, '\n')
	if diff_output==0:
		print('Difference output is zero as expected.\n')
	else:
		print('Difference output is NOT zero; error present.\n')
	print('\n')

	print('Checking Calico Prediction Output:\n')
	print('Calico prediction output: ', pred_output, '\n')
	print('Original model prediction output: ', model_pred, '\n')
	print('Difference between Calcio prediction and Original prediction: ', pred_diff, '\n')
	if pred_diff==0:
		print('Difference between Calico prediction and Original prediction is zero as expected.\n')
	else:
		print('Difference between Calico prediction and Original prediction is NOT zero; error present.\n')
	print('\n')





