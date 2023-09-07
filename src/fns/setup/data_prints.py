# DATA PRINT FUNCTIONS
"""
Contains functions to print out lists of options for data-related input arguments
"""

#############################
## Packages
#############################
import glob
import numpy as np

import fns

#############################
## Print Functions
#############################
def print_fields(npz: np.lib.npyio.NpzFile):
	""" Function that prints a list of radiographic/hydrodynamic fields in an npz file
		
		Args:
			npz (np.lib.npyio.NpzFile): a loaded .npz file
		
		Returns:
			No Return Objects
	"""
	print('List of fields: \n')
	print(npz.files)
	print('List of fields printed. \n')

def print_samples(search_dir: str):
	""" Function that prints how many samples (npz files) in a directory

		Args:
			search_dir (str): file path to directory to search for samples in; include any restrictions for file name
		
		Returns:
			No Return Objects
	"""
	n_samples, _ = fns.save.makefilelist(search_dir, num_samples='All', save=False)
	print(str(n_samples)+' samples found matching '+search_dir+'.\n')