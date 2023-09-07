# ARGPARSE FUNCTIONS
"""
Contains functions for creating a consistent set of argparse arguments for the entire couponmlactivation suite.

Default file paths execute the example code for the tensorflow coupon models
"""

#############################
## Packages
#############################
import argparse
import matplotlib as mpl

#################################
## Operating System & Data Types
#################################
def package(parser):
	parser.add_argument('--PACKAGE', '-P',
	                action='store',
	                type=str,
	                default='tensorflow',
	                choices=['tensorflow', 'pytorch'],
	                help='Which python package was used to create the model',
	                metavar='')

def experiment(parser):
	parser.add_argument('--EXPERIMENT', '-E',
	                action='store',
	                type=str,
	                default='coupon',
	                choices=['coupon', 'nestedcylinder'],
	                help='Which experiment the model was trained on',
	                metavar='')


#############################
## Model Imports
#############################
def model(parser):
	parser.add_argument('--MODEL', '-M',
	                    action='store',
	                    type=str,
	                    default='../examples/tf_coupon/trained_pRad2TePla_model.h5',
	                    help='Model file',
	                    metavar='')

def input_field(parser):
	parser.add_argument('--INPUT_FIELD', '-IF',
	                    action='store',
	                    type=str,
	                    default='pRad',
	                    help='The radiographic/hydrodynamic field the model is trained on',
	                    metavar='')
def input_npz(parser):
	parser.add_argument('--INPUT_NPZ', '-IN',
	                    action='store',
	                    type=str,
	                    default='../examples/tf_coupon/data/r60um_tpl112_complete_idx00110.npz',
	                    help='The .npz file with an input image to the model',
	                    metavar='')
def input_dir(parser):
	parser.add_argument('--INPUT_DIR', '-ID',
	                    action='store',
	                    type=str,
	                    default='../examples/tf_coupon/data/',
	                    help='Directory path where all of the .npz files are stored',
	                    metavar='')
def file_list(parser):
		parser.add_argument('--FILE_LIST', '-FL',
	                    action='store',
	                    type=str,
	                    default='MAKE',
	                    help='The .txt file containing a list of .npz file paths; use "MAKE" to generate a file list given an input directory (passed with -ID) and a number of samples (passed with -NS).',
	                    metavar='')

def design_file(parser):
	parser.add_argument('--DESIGN_FILE', '-DF',
	                    action='store',
	                    type=str,
	                    default='../examples/tf_coupon/coupon_design_file.csv',
	                    help='The .csv file with master design study parameters',
	                    metavar='') 

def norm_file(parser):
	parser.add_argument('--NORM_FILE', '-NF',
	                    action='store',
	                    type=str,
	                    default='../examples/tf_coupon/coupon_normalization.npz',
	                    help='The .npz file normalization values',
	                    metavar='') 

#############################
## Print Toggles
#############################
def print_layers(parser):
	parser.add_argument('--PRINT_LAYERS', '-PL',
	                    action='store_true',
	                    help='Prints list of layer names in a model (passed with -M) and quits program')

def print_features(parser):
	parser.add_argument('--PRINT_FEATURES', '-PT',
	                    action='store_true',
	                    help='Prints number of features extracted by a layer (passed with -L) and quits program')

def print_fields(parser):
	parser.add_argument('--PRINT_FIELDS', '-PF',
	                    action='store_true',
	                    help='Prints list of hydrodynamic/radiographic fields present in a given .npz file (passed with -IN) and quits program')

def print_keys(parser):
	parser.add_argument('--PRINT_KEYS', '-PK',
	                    action='store_true',
	                    help='Prints list of choices for the fixed key avialable in a given input dirrectory (passed with -ID) and quits program')

def print_samples(parser):
	parser.add_argument('--PRINT_SAMPLES', '-PS',
	                    action='store_true',
	                    help='Prints number of samples in a directory (passed with -ID) matching a fixed key (passed with -XK) and quits program')

#############################
## Layer Properties
#############################
def layer(parser):
	parser.add_argument('--LAYER', '-L',
	                    type=str,
	                    action='store',
	                    default='None',
	                    help='Name of model layer that features will be extracted from',
	                    metavar='')

def features(parser, dft=['Grid']):
	parser.add_argument('--FEATURES', '-T',
						nargs='+',
	                    action='store',
	                    default=dft,
	                    help='List of features to include; "Grid" plots all features in one figure using subplots; "All" plots all features each in a new figure; A list of integers can be passed to plot those features each in a new figure. Integer convention starts at 1.',
	                    metavar='')
def dScale(parser):
	parser.add_argument('--D_SCALE', '-DS',
	                    type=float,
	                    action='store',
	                    default=0.001,
	                    help='Scaling factor used in feature derivatives.',
	                    metavar='')

def mat_norm(parser):
	parser.add_argument('--MAT_NORM', '-NM',
	                    type=str,
	                    action='store',
	                    default='ft01',
	                    choices=['ft01', 'all01', 'none'],
	                    help='How the extracted features will be normalized, resulting in a scaled matrix; "ft01" normalizes by the min and max of each feature separately; "all01" normalizes by the min and max of all extracted features; "none" does not normalize features.',
	                    metavar='')

def sclr_norm(parser):
	parser.add_argument('--SCLR_NORM', '-NR',
	                    type=str,
	                    action='store',
	                    default='2',
	                    choices=['fro', 'nuc', 'inf', '-inf', '0', '1', '-1', '2', '-2'],
	                    help='How the extracted features will be normalized, resulting in a scalar value; for choices, see numpy.linalg.norm documentation.',
	                    metavar='')

#############################
## Data Properties
#############################
def fields(parser):
	parser.add_argument('--FIELDS', '-F',
	                    nargs='+',
	                    action='store',
	                    default=['rho', 'eqps', 'eqps_rate', 'eff_stress', 'porosity'],
	                    help='List of fields to be included; pass "none" to use an all-zero field; pass "All" to use all valid fields.',
	                    metavar='')

def fixed_key(parser):
	parser.add_argument('--FIXED_KEY', '-XK',
	                    action='store',
	                    type=str,
	                    default='None',
	                    help='The identifying string for some subset of all data samples; pass "None" to consider all samples',
	                    metavar='')

def num_samples(parser):
	parser.add_argument('--NUM_SAMPLES', '-NS',
	                    action='store',
	                    type=str,
	                    default='All',
	                    help='Number of samples to use; pass "All" to use all samples in a given input dirrectory (passed with -ID)',
	                    metavar='')

#############################
## Color Properties
#############################
def alpha1(parser, dft=0.25):
	parser.add_argument('--ALPHA1', '-A1',
	                    type=float,
	                    action='store',
	                    default=dft,
	                    help='Opacity of colormap at value 0',
	                    metavar='')

def alpha2(parser, dft=1.00):
	parser.add_argument('--ALPHA2', '-A2',
	                    type=float,
	                    action='store',
	                    default=dft,
	                    help='Opacity of colormap at value 1',
	                    metavar='')

def color1(parser, dft='yellow'):
	parser.add_argument('--COLOR1', '-C1',
	                    type=str,
	                    action='store',
	                    choices=mpl.colors.CSS4_COLORS.keys(),
	                    default=dft,
	                    help='Color of colormap at value 0; Choose from matplotlib CSS4 color list.',
	                    metavar='') #prevents printing the long list of color choices

def color2(parser, dft='red'):
	parser.add_argument('--COLOR2', '-C2',
	                    type=str,
	                    action='store',
	                    choices=mpl.colors.CSS4_COLORS.keys(),
	                    default='red',
	                    help='Color of colormap at value 1; Choose from matplotlib CSS4 color list.',
	                    metavar='') #prevents printing the long list of color choices

#############################
## Save Output
#############################
def save_fig(parser):
	parser.add_argument('--SAVE_FIG', '-S',
	                    action='store',
	                    type=str,
	                    default='../examples/tf_coupon/figures/',
	                    help='Directory to save the outputs to.',
	                    metavar='') 

###################################
## Functions Uesd to Autogenerate Documentation
###################################
def argument(parser: argparse.ArgumentParser, dft):
	"""Example function to create the argument "--ARGUMENT"
	
		Args:
			parser (argparse.ArgumentParser): pre-existing parser
			dft (type varies, optional): default value, should be same type as the argument being defined

		Returns:
			No Return Objects
	"""

def all_args():
	parser = argparse.ArgumentParser()

	## Operating System & Data Types
	package(parser)
	experiment(parser)

	## Model Imports
	model(parser)
	input_field(parser)
	input_npz(parser)
	input_dir(parser)
	file_list(parser)
	design_file(parser)
	norm_file(parser)

	## Print Toggles
	print_layers(parser)
	print_features(parser)
	print_fields(parser)
	print_keys(parser)
	print_samples(parser)

	## Layer Properties
	layer(parser)
	features(parser)
	dScale(parser)
	mat_norm(parser)
	sclr_norm(parser)

	## Data Properties
	fields(parser)
	fixed_key(parser)
	num_samples(parser)

	## Color Properties
	alpha1(parser)
	alpha2(parser)
	color1(parser)
	color2(parser)

	## Save Outputs
	save_fig(parser)

	return parser


