# CALICO MODEL DEFINITION FOR BRANCHED TENSORFLOW COUPON MODELS
"""
Defines the calico model for the branched tensorflow coupon models

Execution will print unit test information, perform unit tests, and print the results to the terminal.

Input Line:
``python tf_coupon_calico_model.py -M ../../../../network_files/tfmodels/study_02_221216_prad2tepla_deterministic_model_0100.h5 -IF pRad -IN ../../../../network_files/coupondata/r60um_tpl210_complete_idx00095.npz -DF ../../../../network_files/coupondata/design_res60um_tepla_study220620_MASTER.csv -L activation_15``
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
import fns.setup as setup
import fns.coupondata as cp
import fns.tfcustom as tfc
import fns.tfcustom.calico as tfcalico

#############################
## Create Calico Model
#############################
def make_calico(model, lay: str):
	""" Function that creates a tensorflow "calcio network" from an existing coupon neural net

		Args: 
			model (loaded keras model): model to copy into calico
			lay (str): the name of the layer in model that will become the split layer

		Returns:
			calico (keras.model): calico network
	"""
	## Getting the split layer info
	layer_names=[layer.name for layer in model.layers]
	layerIDX = layer_names.index(lay)
	split_lay = model.layers[layerIDX]

	## Creating the Image Input Layer
	img_input = layers.Input(batch_shape=model.layers[0].get_input_shape_at(0), name='img_input')
	prev_layer = img_input

	## Creating the Constant Tensor Input Layer (to be used in the scaling layer)
	const_input = layers.Input(batch_shape=split_lay.output_shape,
	                           name='const_input')

	## Creating the Input of Model "Ground Truth" and passing it through an identity layer
	truth_input = layers.Input(batch_shape=(None, 4),
                           name='truth_input')

	truth_output = layers.Lambda(tfcalico.idy)(truth_input)

	## Making calico using the keras functional API
	for k, cnnlayer in enumerate(model.layers):
	    if k < layerIDX and k != 0 and "SIM_PRMS" not in cnnlayer.name:
	        prev_layer = cnnlayer(prev_layer)
	    elif k == layerIDX: #does not need SIM_PRMS check because CANNOT be on the SIM_PRMS branch
	        prev_layer1 = cnnlayer(prev_layer)
	        
	        cnnlayer2 = tfcalico.layer_copy(cnnlayer, '_M' ) #tag=M for multiply
	        
	        prev_layer2 = cnnlayer2(prev_layer)
	        prev_layer2 = layers.multiply([const_input, prev_layer2], name='multiply')
	        
	    elif k > layerIDX and "SIM_PRMS" not in cnnlayer.name:
	        prev_layer1 = cnnlayer(prev_layer1)
	        
	        cnnlayer2 = tfcalico.layer_copy(cnnlayer, '_M' ) #tag=M for multiply
	        
	        prev_layer2 = cnnlayer2(prev_layer2)
	        
	    elif k > layerIDX and "SIM_PRMS_IMG" in cnnlayer.name:
	        #constructing the SIM_PRMS branch
	        for i, prms_layer in enumerate(model.layers):
	            if "SIM_PRMS_input"==prms_layer.name:
	                prms_input = layers.Input(batch_shape=model.layers[i].get_input_shape_at(0), name="SIM_PRMS_input")
	                prev_layer_prms = prms_input
	            elif "SIM_PRMS" in prms_layer.name and "SIM_PRMS_IMG" not in prms_layer.name:
	                SIM_PRMS_layer = tfcalico.layer_copy(prms_layer, "") #no tag since they are already labeled
	                prev_layer_prms = SIM_PRMS_layer(prev_layer_prms)
	                
	        #use the SIM_PRMS branch in the concat layer
	        prev_layer1 = cnnlayer([prev_layer1, prev_layer_prms])
	        
	        cnnlayer2 = tfcalico.layer_copy(cnnlayer, '_M' ) #tag=M for multiply
	        
	        prev_layer2 = cnnlayer2([prev_layer2, prev_layer_prms])
	        
	diff_output = layers.subtract([prev_layer1, prev_layer2])

	pred_output = prev_layer1

	calico = models.Model([img_input, const_input, prms_input, truth_input], [diff_output, pred_output, truth_output])

	## Setting the calico weights
	for k, cnnlayer in enumerate(model.layers):
	    layer_weights = cnnlayer.get_weights()
	    layer_name = cnnlayer.name
	    #Copy over all the one for one layers
	    get_layer = calico.get_layer(layer_name)
	    get_layer.set_weights(layer_weights)
	    #Try each layer with a _M tag
	    try:
	        get_layer_M = calico.get_layer(layer_name+"_M") 
	        get_layer_M.set_weights(layer_weights)
	    except:
	        pass

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
	model = keras.models.load_model(model_path)

	## Try to Rename the SIM_PRMS Input Layer (not all models require this)
	try:
		prm_in_lay =  model.get_layer('param_input')
		prm_in_lay._name = 'SIM_PRMS_input'
		print('\nLayer "param_input" has been renamed to "SIM_PRMS_input".\n')
	except:
		pass
	print('\n') #Seperate other output from Tensorflow warnings

	## Model Prints
	if PRINT_LAYERS: 
		tfc.prints.print_layers(model)
		sys.exit()

	## Model Checks
	tfcalico.check_calico_layer(model, lay, branch='SIM_PRMS', catlay='SIM_PRMS_IMG_concat')

	#############################
    ## Print Unit Test Information 
    #############################
	print('This script runs unit tests for the Calico model. Since the Calico network has three outputs, there are three unit tests.\n')
	print('-----\n')
	print('Unit Test for the Difference Output:\n')
	print('The Calico difference output is the difference between the original branch and the multiple branch.\n')
	print('This test sets the tensor multiplier to one. Therefore, the difference should be zero.\n')
	print('-----\n')
	print('Unit Test for the Prediction Output:\n')
	print('The Calico prediction output is the output from the origial branch.\n')
	print('This test compares the Calico prediction output to the original model prediction output. The difference should be zero.\n')
	print('-----\n')
	print('Unit Test for the Truth Output:\n')
	print('The Calico truth output is the the same as the ground truth passed into Calico.\n')
	print('This test compares the truth inputed to Calcio with the truth output. The difference should be zero.\n')
	print('.\n.\n.\n')

	#############################
    ## Make & Test Calico Model
    #############################
	## Make Calico Model
	calico = make_calico(model, lay)

	## Making the Model Inputs
	img_input = cp.process.get_field(npz, in_field)
	key = cp.process.npz2key(input_path)
	dcj = cp.process.csv2dcj(design_path, key)
	img_y, img_x = model.input_shape[0][1:3]
	cont_y, cont_x, cont_ft = calico.input_shape[1][1:4]
	cont_ones= np.ones((cont_y, cont_x, cont_ft))
	design_df = pd.read_csv(design_path,
		                    sep=', ',
		                    header=0,
		                    index_col=0,
		                    engine='python')
	sim_time = npz['sim_time']
	truth = np.array([sim_time, design_df.at[key, 'yspall'], design_df.at[key, 'log10_phi0'], design_df.at[key, 'eta']])

	## Making Predictions
	model_pred = model.predict([img_input.reshape((1,img_y,img_x,1)), dcj])
	calico_pred = calico.predict([img_input.reshape((1,img_y,img_x,1)), cont_ones.reshape((1,cont_y, cont_x, cont_ft)), dcj, truth.reshape((1,4))])
	print('.\n.\n.\n')

	## Test Predictions
	diff_output = calico_pred[0]
	pred_output = calico_pred[1]
	pred_diff = pred_output - model_pred
	truth_output = calico_pred[2]
	truth_diff = truth - truth_output

	#############################
    ## Print Unit Test Results
    #############################
	print('Calico Network Unit Test Results\n')
	print('-----\n')
	print('Checking Calico Difference Output:\n')
	print(diff_output, '\n')
	if np.all(diff_output==0):
		print('Difference output is all zeros as expected.\n')
	else:
		print('Difference output is NOT all zeros; error present.\n')
	print('\n')

	print('Checking Calico Prediction Output:\n')
	print('Calico prediction output: ', pred_output, '\n')
	print('Original model prediction output: ', model_pred, '\n')
	print('Difference between Calcio prediction and Original prediction: ', pred_diff, '\n')
	if np.all(pred_diff==0):
		print('Difference between Calico prediction and Original prediction is all zeros as expected.\n')
	else:
		print('Difference between Calico prediction and Original prediction is NOT all zeros; error present.\n')
	print('\n')

	print('Checking Calico Truth Output:\n')
	print('Ground Truth input to Calico: ', truth, '\n')
	print('Calico truth output: ', truth_output, '\n')
	print('Difference between input truth and output truth: ', truth_diff, '\n')
	if np.all(truth_diff==0):
		print('Difference between input truth and output truth is all zeros as expected.\n')
	else:
		print('Difference between input truth and output truth is NOT all zeros; error present.\n')
	print('\n')




