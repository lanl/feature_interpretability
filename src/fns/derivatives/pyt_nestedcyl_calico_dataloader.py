# CALICO DATALOADER DEFINITION FOR SINGLE BRANCH PYTORCH NESTED CYLINDER MODELS
"""
Defines the pytorch dataset class for the single branch pytorch nested cylinder models

Execution will print test information, perform tests, and print the results to the terminal.

Input Line:
``python pyt_nestedcyl_calico_dataloader.py -M ../../../../network_files/pytmodels/checkpoint_V0.3_Epoch1000.pth -IF hr_MOICyl -ID /data/nested_cyl_230428/ -DF ../../../../network_files/nestedcyldata/runsKey.csv``
"""

#############################
## Packages
#############################
import os
import sys
import glob
import re
import typing
import copy
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.insert(0, os.path.abspath('../../'))
import fns
import fns.setup as setup
import fns.nestedcylinderdata as nc
import fns.pytorchcustom as pytc
import fns.pytorchcustom.calico as pytcalico

#############################
## Calico Dataset Definition
############################# 
class calico_DataSet(Dataset):
	""" The definition of a dataset object used as input to the pytorch nested cylinder calico neural networks.

		Args:
			input_field (str): The radiographic/hydrodynamic field the model is trained on
			input_dir (str): The directory path where all of the .npz files are stored 
			filelist (str): Text file listing file names to read.
			design_file (str): .csv file with master design study parameters
	"""

	def __init__(self,
					input_field: str='hr_MOICyl',
					input_dir: str='/data/nested_cyl_230428/',
					filelist: str='../../coupon_ml/yellow_r60um_tpl_testing.txt',
			        design_file: str='/data/nested_cyl_230428//runsKey.csv'):

		## Model Arguments 
		self.input_field = input_field
		self.input_dir = input_dir
		self.filelist = filelist
		self.design_file = design_file

		## Create filelist
		with open(filelist, 'r') as f:
			self.filelist = [line.rstrip() for line in f]
		self.Nsamples = len(self.filelist)

	def __len__(self):
		"""
		Return number of samples in dataset.
		"""
		return self.Nsamples

	def __getitem__(self, index):
		"""
		Return a tuple of a batch's input and output data for training at a given index.
		"""

		filepath = self.filelist[index]

		## Get index to design file from NPZ filename
		key = nc.process.npz2key(filepath)
		ptw = float(key.split('_')[-1])
		idx_long = nc.process.npz2idx(filepath)
		idX = float(re.split('idx', idx_long)[-1])
		sampleID = (ptw, idX)

		## Get the input image
		npz = np.load(os.path.join(self.input_dir, filepath))
		img_input = nc.process.get_field(npz, self.input_field)
		in_y, in_x = img_input.shape
		img_input = img_input.reshape((1, in_y, in_x))
		img_input = torch.tensor(img_input).to(torch.float32)

		## Get the ground truth
		key = nc.process.npz2key(filepath)
		truth = nc.process.csv2scalePTW(self.design_file, key)

		return (img_input, truth, sampleID)

def calico_dataloader(input_field: str='hr_MOICyl',
					input_dir: str='/data/nested_cyl_230428/',
					filelist: str='../../coupon_ml/yellow_r60um_tpl_testing.txt',
					design_file: str='/data/nested_cyl_230428//runsKey.csv',
					batch_size: int=8):
		""" Function to create a pytorch dataloader from the pytorch nested cylinder calico model dataset

			Args:
				input field (str): The radiographic/hydrodynamic field the model is trained on
				input_dir (str): The directory path where all of the .npz files are stored 
				filelist (str): Text file listing file names to read.
				design_file (str): .csv file with master design study parameters

			Returns:
				dataloader (torch.utils.data.DataLoader): pytorch dataloader made from calico model dataset
		"""

		dataset = calico_DataSet(input_field, input_dir, filelist, design_file)
		dataloader = DataLoader(dataset, batch_size=batch_size)

		return dataloader

#############################
## Set Up Parser
#############################
def calico_dataloader_parser():
    descript_str = 'Creates and tests a calcio dataloader (for input to a calico model) given an input model, layer, and feature'

    parser = argparse.ArgumentParser(prog='Calico DataLoader',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')

    ## Model Imports
    setup.args.model(parser) # -M
    setup.args.input_field(parser) # -IF
    setup.args.input_dir(parser) # -ID
    setup.args.file_list(parser) # -FL
    setup.args.design_file(parser) # -DF

    ## Print Toggles
    setup.args.print_samples(parser) # -PS

    ## Layer Properties
    setup.args.dScale(parser) #-DS

    ## Data Properties
    setup.args.num_samples(parser) # -NS

    return parser

parser = calico_dataloader_parser()

#############################
## Unit Tests for Calico Sequence
#############################
if __name__ == '__main__':

	## Parse Args
	args = parser.parse_args()

	## Model Imports
	model_path = args.MODEL
	in_field = args.INPUT_FIELD
	input_dir = args.INPUT_DIR
	file_list_path = args.FILE_LIST
	design_path = args.DESIGN_FILE

	## Print Toggles
	PRINT_SAMPLES = args.PRINT_SAMPLES

	## Layer PropertiesS
	dScale = args.D_SCALE

	## Data Properties
	num_samples = args.NUM_SAMPLES

	## Load Model
	model_class = 'NCylANN_V1'
	sys.path.append(os.path.abspath('../../../../../syntheticnestedcyldata/'))
	model = pytc.fts.load_model(model_path, model_class, device)

	## Get a Test File
	search_dir = input_dir+'ncyl_sclPTW*idx*.npz'
	test_path = glob.glob(search_dir)[0]
	test_npz = np.load(test_path)

	## Prints
	if PRINT_SAMPLES: 
	        setup.data_prints.print_samples(search_dir)
	        sys.exit()

	## Checks
	setup.data_checks.check_fields(test_npz, [in_field])
	if file_list_path == 'MAKE': 
	    setup.data_checks.check_samples(num_samples, search_dir)

	## Printing Unit Test Information
	print('This script runs unit tests for the Calico dataloader.\n')
	print('The calcio dataloader creates a tuple of an image input, the ground truth PTW scaling constant, and the sample information.')
	print('The unit tests print the length of the dataset to confirm that is is the same length as the number of samples provided.\n')
	print('The unit tests print the shapes of the batched input and ground truth. The user must check that these sizes are correct. Batch size 8 is used.\n')
	print('-----\n')

	## Processing the File List
	if file_list_path == 'MAKE':
		num_samples, sample_list = fns.save.makefilelist(search_dir, num_samples, save_path='./calico_dataloader_test', save=True)
		file_list_path = './calico_dataloader_test_SAMPLES.txt'
	else:
		num_samples, sample_list = fns.misc.load_filelist(file_list_path)

	dataloader = calico_dataloader(input_field = in_field,
									input_dir = input_dir,
									filelist = file_list_path,
									design_file = design_path,
									batch_size = 8)


	## Evaluate the Sequence
	imgs, ptws = next(iter(dataloader))
	data_length = len(dataloader.dataset)

	## Delete the Samples .txt file
	if file_list_path == './calico_dataloader_test_SAMPLES.txt': os.remove('./calico_dataloader_test_SAMPLES.txt')


	## Print Unit Tests to Terminal
	print('Calico DataLoader Length Method Results:\n')
	print('Number of samples submitted:', num_samples)
	print('Number of samples in dataloader:', data_length)
	if num_samples == data_length:
		print('Dataset length method returned expected value.')
	else:
		print('Dataset length method did NOT return expected value; error present.')
	print('\n')

	print('Calico DataLoader Get Item Results:\n')
	print('Image input array shape:', imgs.shape)
	print('Ground truth array shape:', ptws.shape)
