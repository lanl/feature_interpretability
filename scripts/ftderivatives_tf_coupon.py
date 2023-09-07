# TENSORFLOW COUPON FEATURE DERIVATIVES SCRIPT
"""
Calculates the derivatives of model outputs with respect to scaling an internal feature using a calico network.

Only operational on tensorflow models trainined on the coupon experiment.

Exports derivatives and other sample information as a pandas-readable .csv, including:
 - Difference (diff\_) between the scaled and unscaled outputs
 - Derivative (derv\_) of the outputs; equivalent to the *difference* divided by the *dScale*
 - Prediction (pred\_) of the unscaled network
 - Truth (true\_) TePla parameters corresponding to the input sample, and identifying sample information

Input Line:
``python ftderivatives_tf_coupon.py -M ../examples/tf_coupon/trained_pRad2TePla_model.h5 -IF pRad -ID ../examples/tf_coupon/data/ -DF ../examples/tf_coupon/coupon_design_file.csv -NF ../examples/tf_coupon/coupon_normalization.npz -L activation_15 -T 1 -S ../examples/tf_coupon/figures/``
"""

#############################
## Packages
#############################
import os
import sys
import argparse
import math
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, models, layers, activations


#############################
## Custom Functions Import Code
#############################
sys.path.insert(0, os.path.abspath('../src/'))
import fns
import fns.setup as setup
import fns.coupondata as cp
import fns.tfcustom.fts as tfc
import fns.tfcustom.calico as tfcalico
from fns.derivatives.tf_coupon_calico_model import make_calico
from fns.derivatives.tf_coupon_calico_seq import calicoSEQ


#############################
## Set Up Parser
#############################
def feature_derivatives_parser():
	descript_str = 'Calculates the derivatives of model outputs with respect to scaling an internal feature using a calico network.'

	parser = argparse.ArgumentParser(prog='Feature Derivatives',
	                             description=descript_str,
	                             fromfile_prefix_chars='@')
	## Model Imports
	setup.args.model(parser) # -M
	setup.args.input_field(parser) # -IF
	setup.args.input_dir(parser) # -ID
	setup.args.file_list(parser) # -FL
	setup.args.design_file(parser) # -DF
	setup.args.norm_file(parser) # -NF

	## Print Toggles
	setup.args.print_layers(parser) # -PL
	setup.args.print_features(parser) # -PT
	setup.args.print_fields(parser) # -PF
	setup.args.print_keys(parser) # -PK
	setup.args.print_samples(parser) # -PS

	## Layer Properties
	setup.args.layer(parser) # -L
	setup.args.features(parser, dft=['1']) # -T
	setup.args.dScale(parser) #-DS

	## Data Properties
	setup.args.fixed_key(parser) # -XK
	setup.args.num_samples(parser) # -NS

	## Save Properties
	setup.args.save_fig(parser) # -S

	return parser

parser = feature_derivatives_parser()

#############################
## Executable
#############################
if __name__ == '__main__':

	args = parser.parse_args()

	## Model Imports
	model_path = args.MODEL
	in_field = args.INPUT_FIELD
	input_dir = args.INPUT_DIR
	file_list_path = args.FILE_LIST
	design_path = args.DESIGN_FILE
	norm_path = args.NORM_FILE

	## Print Toggles
	PRINT_LAYERS = args.PRINT_LAYERS
	PRINT_FEATURES = args.PRINT_FEATURES
	PRINT_FIELDS = args.PRINT_FIELDS
	PRINT_KEYS = args.PRINT_KEYS
	PRINT_SAMPLES = args.PRINT_SAMPLES

	## Layer Properties
	lay = args.LAYER
	features = args.FEATURES
	dScale = args.D_SCALE

	## Data Properties
	fixed_key = args.FIXED_KEY
	num_samples = args.NUM_SAMPLES

	## Save Properties
	fig_path = args.SAVE_FIG

	#############################
    ## File List Path Processing
    #############################
	## Import Custom Packages
	import fns.coupondata as cp
	import fns.coupondata.prints as cpprints
	search_dir = input_dir+'r*tpl*idx*.npz'

	## Create the File list 
	if file_list_path != 'MAKE':
		## Import given file list
		num_samples, sample_list = fns.misc.load_filelist(file_list_path)
		fixed_key = cp.process.findkey(sample_list)

		## Prints
		if PRINT_KEYS: print('Key present in '+file_list_path+' is '+fixed_key)
		if PRINT_SAMPLES: print(str(num_samples)+' samples found in '+file_list_path+'.\n')
		if PRINT_KEYS or PRINT_SAMPLES: sys.exit()

		if fixed_key=='None': fixed_key=''

	elif file_list_path == 'MAKE':
		## Print Keys
		if PRINT_KEYS: 
			cpprints.print_keys(search_dir)

			if PRINT_SAMPLES: 
				print('Print Samples -PS command is not compatible with Print Keys -PK.')
				print('Specify a fixed key -XK to use -PS.')

			sys.exit()

		## Set Fixed Key
		if fixed_key != 'None':
			search_dir = input_dir+'r*'+fixed_key+'*.npz'

		## Check Fixed Key
		setup.data_checks.check_key(fixed_key, search_dir)

		## Print Samples
		if PRINT_SAMPLES:
			setup.data_prints.print_samples(search_dir)
			sys.exit()

		## Check Samples
		setup.data_checks.check_samples(num_samples, search_dir)

		## Make File List
		file_path = fig_path+'feature_derivatives'
		num_samples, sample_list = fns.save.makefilelist(search_dir, num_samples=num_samples, save_path=file_path, save=True)
		file_list_path = file_path + '_SAMPLES.txt'

	print('File list loaded and processed sucessfully.')

    #############################
    ## Data Processing
    #############################
	## Getting a Test Data File
	test_path = os.path.join(input_dir, sample_list[0])
	test_npz =  np.load(test_path)

	## Data Prints
	if PRINT_FIELDS: 
		setup.data_prints.print_fields(test_npz)
		sys.exit()

	## Data Checks
	setup.data_checks.check_fields(test_npz, [in_field])

	## Data Formatting
	test_input = cp.process.get_field(test_npz, in_field)
	in_y, in_x = test_input.shape

    #############################
    ## Model Processing
    #############################
	## Load Model
	model = keras.models.load_model(model_path)
	print('\n') #Seperate other output from Tensorflow 

	## Try to Rename the SIM_PRMS Input Layer (not all models require this)
	try:
	    prm_in_lay =  model.get_layer('param_input')
	    prm_in_lay._name = 'SIM_PRMS_input'
	    print('\nLayer "param_input" has been renamed to "SIM_PRMS_input".\n')
	except:
	    pass
	print('\n') #Seperate other output from Tensorflow warnings

	## Prints
	if PRINT_LAYERS: tfc.prints.print_layers(model)
	if PRINT_FEATURES: tfc.prints.print_features(model, lay)
	if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

	## Checks
	tfcalico.check_calico_layer(model, lay, branch='SIM_PRMS', catlay='SIM_PRMS_IMG_concat')
	tfcalico.check_calico_features(model, lay, features)

    #############################
    ## Creating the Sequence
    #############################
	## Creating Sequence Inputs
	ft = int(features[0])-1
	layshape = model.get_layer(name=lay).output_shape[1:]
	epochs = math.ceil(num_samples / 8)

	## Initiate the Sequence
	seq = calicoSEQ(#Model Arguments
	                 input_field=in_field,
	                 input_dir=input_dir,
	                 filelist=file_list_path,
	                 design_file=design_path,
	                 normalization_file=norm_path,
	                 batch_size=8,
	                 epoch_length=epochs,
	                 #Derivative Argumets
	                 layshape=layshape,
	                 ftIDX=ft,
	                 dScale=dScale)
	print('Sequence Created Sucessfully\n')

    #############################
    ## Creating the Calico Network
    #############################
	calico = make_calico(model, lay)
	print('Calico Model Created Sucessfully\n')

	#############################
    ## Predicting with the Calico Network
    #############################
	predictions = calico.predict(seq,
							    max_queue_size=10,
							    workers=1,
							    use_multiprocessing=False
							    )

	print('Predictions Created Sucessfully\n')


	#############################
    ## Saving the Output
    #############################
	## Seperating the Output
	diff_out = predictions[0]
	derv_out = diff_out / dScale
	pred_out = predictions[1]
	truth_out = predictions[2]
	out_concat = np.append(diff_out, derv_out, axis=1)
	out_concat = np.append(out_concat, pred_out, axis=1)
	out_concat = np.append(out_concat, truth_out, axis=1)

	## Making Headers
	diff_label = ['diff_'] * diff_out.shape[-1]
	derv_label = ['derv_'] * derv_out.shape[-1]
	pred_label = ['pred_'] * pred_out.shape[-1]
	truth_label = ['true_'] * truth_out.shape[-1]
	all_labels = [diff_label, derv_label, pred_label, truth_label]
	header1 = [item for sublist in all_labels for item in sublist]
	header1 = np.array(header1)

	TePla = np.array(['sim_time', 'yspall', 'log10_phi0', 'eta'] * 4)
	header2 = np.append(TePla, ['tpl', 'idx'])

	header = np.core.defchararray.add(header1, header2)

	## Save Results to a DataFrame
	fileprefix = fig_path+lay+'_ft'+str(ft+1)+'_dScale'+str(dScale).replace('.', '_')
	df = pd.DataFrame(out_concat, columns = header)
	df.to_csv(fileprefix+'_derivative_DF_long.csv')
	cleandf = df.drop_duplicates().reset_index().iloc[:,1:]
	cleandf.to_csv(fileprefix+'_derivative_DF.csv')
	print('Predictions Saved to DataFrame Sucessfully\n')
