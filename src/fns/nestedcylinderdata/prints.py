# NESTED*CYINDER*DATA PRINTS SCRIPT
"""
Contains functions to print out lists of options for input arguments
"""

#############################
## Packages
#############################
import glob
import numpy as np

import fns
import fns.nestedcylinderdata as nc

#############################
## Print Functions
#############################
def print_keys(search_dir: str):
	""" Function that prints a list of unique variable indexes for a given fixed variable

		Args:
			search_dir (str): file path to directory to search for samples in
		
		Returns:
			No Return Objects
	"""

	_, all_files = fns.save.makefilelist(search_dir, num_samples='All', save=False)
	unq_ptws, unq_idxs = nc.process.listkeys(all_files)

	print('Searching in', search_dir)
	print('List of unique keys that fix the strength model scaling factor:', unq_ptws)
	print('List of unique keys that fix a time stamp:', unq_idxs)