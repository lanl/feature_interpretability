# PYTORCH NESTED CYLINDER FEATURE DERIVATIVES SCRIPT
"""
Calculates the derivatives of model outputs with respect to scaling an internal feature using a calico network.

Only operational on pytorch models trainined on the nested cylinder experiment.

Exports derivatives and other sample information as a pandas-readable .csv, including:
 - Difference between the scaled and unscaled outputs
 - Derivative of the outputs; equivalent to the *difference* divided by the *dScale*
 - Prediction of the unscaled network
 - Truth PTW scaling corresponding to the input sample, and identifying sample information

Input Line:
``COMING SOON``
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
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################
## Custom Functions Import Code
#############################
sys.path.insert(0, os.path.abspath('../src/'))
import fns
import fns.setup as setup
import fns.nestedcylinderdata as nc
import fns.pytorchcustom as pytc
import fns.pytorchcustom.calico as pytcalico
from fns.derivatives.pyt_nestedcyl_calico_model import load_calico
from fns.derivatives.pyt_nestedcyl_calico_dataloader import calico_dataloader



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

	raise NotImplementedError('Nested cylinder examples not included in open source.')
	
	args = parser.parse_args()

	## Model Imports
	model_path = args.MODEL
	in_field = args.INPUT_FIELD
	input_dir = args.INPUT_DIR
	file_list_path = args.FILE_LIST
	design_path = args.DESIGN_FILE

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
	import fns.nestedcylinderdata as nc
	import fns.nestedcylinderdata.prints as ncprints
	search_dir = input_dir+'ncyl_sclPTW*idx*.npz'

	## Create the File list 
	if file_list_path != 'MAKE':
		## Import given file list
		num_samples, sample_list = fns.misc.load_filelist(file_list_path)
		fixed_key = nc.process.findkey(sample_list)

		## Prints
		if PRINT_KEYS: print('Key present in '+file_list_path+' is '+fixed_key)
		if PRINT_SAMPLES: print(str(num_samples)+' samples found in '+file_list_path+'.\n')
		if PRINT_KEYS or PRINT_SAMPLES: sys.exit()

		if fixed_key=='None': fixed_key=''

	elif file_list_path == 'MAKE':
		## Print Keys
		if PRINT_KEYS: 
			ncprints.print_keys(search_dir)

			if PRINT_SAMPLES: 
				print('Print Samples -PS command is not compatible with Print Keys -PK.')
				print('Specify a fixed key -XK to use -PS.')

			sys.exit()

		## Set Fixed Key
		if fixed_key != 'None':
			search_dir = input_dir+'ncyl*'+fixed_key+'*.npz'

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
	test_input = nc.process.get_field(test_npz, in_field)
	in_y, in_x = test_input.shape

    #############################
    ## Model Processing
    #############################
	## Load Model
	model_class = 'NCylANN_V1'
	model = pytc.fts.load_model(model_path, model_class, device)

	## Prints
	if PRINT_LAYERS: pytc.prints.print_layers(model)
	if PRINT_FEATURES: pytc.prints.print_features(model)
	if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

	## Checks
	pytcalico.check_calico_layer(model, lay, branch='None', catlay='None')
	pytcalico.check_calico_features(model, features)

	#############################
	## Creating the Dataloader
	#############################
	dataloader = calico_dataloader(input_field = in_field,
									input_dir = input_dir,
									filelist = file_list_path,
									design_file = design_path,
									batch_size = 8)

	print('DataLoader Created Sucessfully\n')

	#############################
	## Creating the Calico Network
	#############################
	ft = int(features[0])-1
	calico = load_calico(model = model, 
					checkpoint = model_path, 
					device = device,
					lay = lay,
					ftIDX = ft,
					dScale = dScale,)
	print('Calico Model Created Sucessfully\n')

	#############################
	## Predicting with the Calico Network
	#############################
	pred_out = np.zeros((num_samples, 1))
	diff_out = np.zeros((num_samples, 1))
	truth_out = np.zeros((num_samples, 1))
	sample_out = np.zeros((num_samples, 2))

	calico.eval()
	with torch.no_grad():
		for i, data in enumerate(dataloader):
			img_input, truth, sampleID = data
			truth_out[i*8 : (i+1)*8, 0] = truth
			sample_out[i*8 : (i+1)*8, :] = np.transpose(sampleID)

			(splitx, diff) = calico(img_input)
			pred_out[i*8 : (i+1)*8, 0] = splitx
			diff_out[i*8 : (i+1)*8, 0] = diff

	print('Predictions Created Sucessfully\n')

	#############################
    ## Saving the Output
    #############################
	## Seperating the Output
	derv_out = diff_out / dScale
	out_concat = np.append(diff_out, derv_out, axis=1)
	out_concat = np.append(out_concat, pred_out, axis=1)
	out_concat = np.append(out_concat, truth_out, axis=1)
	out_concat = np.append(out_concat, sample_out, axis=1)

	## Making Headers
	header = ['difference', 'derivative', 'prediction', 'truth', 'ptw', 'idx']

	## Save Results to a DataFrame
	fileprefix = fig_path+lay+'_ft'+str(ft+1)+'_dScale'+str(dScale).replace('.', '_')
	df = pd.DataFrame(out_concat, columns = header)
	df.to_csv(fileprefix+'_derivative_DF_long.csv')
	cleandf = df.drop_duplicates().reset_index().iloc[:,1:]
	cleandf.to_csv(fileprefix+'_derivative_DF.csv')
	print('Predictions Saved to DataFrame Sucessfully\n')
