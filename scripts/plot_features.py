# PLOT FEATURES SCRIPT
"""
Plots one or multiple features extracted from a given layer over the model's input field
 - Can plot all features from a layer on the same plot (``-T Grid``)
 - Can plot all features from a layer on their own plots (``-T All``)
 - Can plot some features from a layer on their own plots (``-T # #``)

Saves all features to a .npz file.

Input Line for TF Coupon Models: 
``python plot_features.py -P tensorflow -E coupon -M ../examples/tf_coupon/trained_pRad2TePla_model.h5 -IF pRad -IN ../examples/tf_coupon/data/r60um_tpl112_complete_idx00110.npz -DF ../examples/tf_coupon/coupon_design_file.csv -L activation_15 -T Grid -NM ft01 -S ../examples/tf_coupon/figures/``

Input Line for PYT Nested Cylinder Models:
``COMING SOON``

"""

#############################
## Packages
#############################
import os
import sys
import argparse
import numpy as np
import pandas as pd

## Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('pdf') ## To change which backend is used

## Custom Functions
sys.path.insert(0, os.path.abspath('../src/'))
import fns
import fns.setup as setup
import fns.plot

#############################
## Set Up Parser
#############################
def plot_features_parser():
	descript_str = "Plots one or multiple features extracted from a given layer over the model's input field"

	parser = argparse.ArgumentParser(prog='Feature Ploting',
									 description=descript_str,
									 fromfile_prefix_chars='@')
	## Package & Experiment
	setup.args.package(parser) # -P
	setup.args.experiment(parser) # -E

	## Model Imports
	setup.args.model(parser) # -M
	setup.args.input_field(parser) # -IF
	setup.args.input_npz(parser) # -IN
	setup.args.design_file(parser) # -DF

	## Print Toggles
	setup.args.print_layers(parser) # -PL
	setup.args.print_features(parser) # -PT
	setup.args.print_fields(parser) # -PF

	## Layer Properties
	setup.args.layer(parser) # -L
	setup.args.features(parser, dft=['Grid']) # -T
	setup.args.mat_norm(parser) # -NM

	## Color Properties
	setup.args.alpha1(parser, dft=0.25) # -A1
	setup.args.alpha2(parser, dft=1.00) # -A2
	setup.args.color1(parser, dft='yellow') # -C1
	setup.args.color2(parser, dft='red') # -C2

	## Save Properties
	setup.args.save_fig(parser) # -S

	return parser

parser = plot_features_parser()

#############################
## Executable
#############################
if __name__ == '__main__':

	args = parser.parse_args()

	## Package & Experiment
	PACKAGE = args.PACKAGE
	EXP = args.EXPERIMENT

	## Model Imports
	model_path = args.MODEL
	in_field = args.INPUT_FIELD
	input_path = args.INPUT_NPZ
	design_path = args.DESIGN_FILE

	## Print Toggles
	PRINT_LAYERS = args.PRINT_LAYERS
	PRINT_FEATURES = args.PRINT_FEATURES
	PRINT_FIELDS = args.PRINT_FIELDS

	## Layer Properties
	lay = args.LAYER
	features = args.FEATURES
	norm = args.MAT_NORM

	## Color Properties
	alpha1 = args.ALPHA1
	alpha2 = args.ALPHA2
	color1 = args.COLOR1
	color2 = args.COLOR2

	## Save Properties
	fig_path = args.SAVE_FIG

	#############################
	## Data Processing
	#############################

	npz = np.load(input_path)

	## Data Prints
	if PRINT_FIELDS: 
		setup.data_prints.print_fields(npz)
		sys.exit()

	## Data Checks
	setup.data_checks.check_fields(npz, [in_field])

	if EXP == 'coupon':
		## Package Imports
		import fns.coupondata as cp

		## Input Processing
		img_in = cp.process.get_field(npz, in_field)
		Rlabels, Rticks, Zlabels, Zticks = cp.process.get_ticks(npz)
		key = cp.process.npz2key(input_path)
		dcj = cp.process.csv2dcj(design_path, key)
		model_in = [img_in, dcj]

		in_y, in_x = img_in.shape
		model_in = [img_in.reshape((1,in_y,in_x,1)),dcj]

		## Plotting Settings
		bigfigsize = (20,10)
		smallfigsize = (10,5)
		RMinorTicks = False
		ZMinorTicks = False

	elif EXP == 'nestedcylinder':
		## Package Imports
		import fns.nestedcylinderdata as nc

		## Input Processing
		img_in = nc.process.get_field(npz, in_field)
		Rlabels, Rticks, Zlabels, Zticks = nc.process.get_ticks(npz)
		model_in = img_in

		## Plotting Settings
		bigfigsize = (10,20)
		smallfigsize = (5,10)
		RMinorTicks = True
		ZMinorTicks = False

	print('Data loaded and processed sucessfully.')

	#############################
	## Model Processing
	#############################

	if PACKAGE == 'tensorflow':
		## Package Imports
		import tensorflow as tf
		from tensorflow import keras
		import fns.tfcustom as tfc

		## Load Model
		model = keras.models.load_model(model_path)
		print('\n') #Seperate other output from Tensorflow warnings

		## Prints
		if PRINT_LAYERS: tfc.prints.print_layers(model)
		if PRINT_FEATURES: tfc.prints.print_features(model, lay)
		if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

		## Checks
		tfc.checks.check_layer(model, lay)
		tfc.checks.check_features(model, lay, features)

		## Extract Features
		ft_mat = tfc.fts.feature_extractor(model, lay, model_in, norm)

	elif PACKAGE == 'pytorch':
		## Package Imports
		import torch
		import fns.pytorchcustom as pytc
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		## Load Model
		import fns.pytorchcustom.field2PTW_model_definition as field2PTW_model
		model = field2PTW_model.field2PTW(img_size = (1, 1700, 500),
											size_threshold = (8, 8),
											kernel = 5,
											features = 12, 
											interp_depth = 12,
											conv_onlyweights = False,
											batchnorm_onlybias = False,
											act_layer = torch.nn.GELU,
											hidden_features = 20)

		## Prints
		if PRINT_LAYERS: pytc.prints.print_layers(model)
		if PRINT_FEATURES: pytc.prints.print_features(model)
		if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

		## Checks
		pytc.checks.check_layer(model, lay)
		pytc.checks.check_features(model, features)

		## Extract Features
		ft_mat = pytc.fts.feature_extractor(model, lay, model_in, norm, device)

	## Save Extracted Features
	fns.save.features2npz(ft_mat, save_path = fig_path+lay+'_all_features' )

	print('Model loaded sucessfully; features extracted and saved.')

	#############################
	## Plot Features
	#############################
	maxiter = np.shape(ft_mat)[-1]
	fns.plot.custom_colormap(color1, color2, alpha1, alpha2)
	lims = [0,1]

	if features == ['Grid']: #Plot all features in a grid
		plt.rcParams.update({'font.size': 22})
		fig, axs = plt.subplots(4, 3, figsize=bigfigsize, layout="constrained")
		for i in range(maxiter):
			plt.subplot(4,3,i+1)
			im = fns.plot.feature_plot(img_in, ft_mat[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, lims)
			plt.title('Feature '+str(i+1), fontsize=16)

		fig.colorbar(im, ax=axs, label='Activation Intensity')
		plt.clim(lims)
		fig.suptitle('Features from '+lay)

		fig.savefig(fig_path+lay+'_all_features.png')
		plt.close()

	elif features == ['All']: #Plot all features in thier own figures
		plt.rcParams.update({'font.size': 14})
		for i in range(maxiter):
			fig = plt.figure(figsize=smallfigsize, layout="constrained")
			im = fns.plot.feature_plot(img_in, ft_mat[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, lims)
			plt.suptitle('Feature '+str(i+1)+' from '+lay)
			fig.colorbar(im, label='Activation Intensity')
			plt.clim(lims)
			
			fig.savefig(fig_path+lay+'_feature_'+str(i+1)+'.png')
			plt.close()

	else: #Plotting a specific list of features
		plt.rcParams.update({'font.size': 14})
		features = [int(i)-1 for i in features]
		for i in features:
			fig = plt.figure(figsize=smallfigsize, layout="constrained")
			im = fns.plot.feature_plot(img_in, ft_mat[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, lims)
			plt.suptitle('Feature '+str(i+1)+' from '+lay)
			fig.colorbar(im, label='Activation Intensity')
			plt.clim(lims)

			fig.savefig(fig_path+lay+'_feature_'+str(i+1)+'.png')
			plt.close()

	print('Plots generated and saved.')


