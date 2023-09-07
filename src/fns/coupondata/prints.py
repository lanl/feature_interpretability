# COUPON*DATA PRINTS SCRIPT
"""
Contains functions to print out lists of options for data-related input arguments
"""

#############################
## Packages
#############################
import glob
import numpy as np

import fns
import fns.coupondata as cp

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
	unq_tpls, unq_idxs = cp.process.listkeys(all_files)

	print('Searching in', search_dir)
	print('List of unique keys that fix experimental parameters:', unq_tpls)
	print('List of unique keys that fix a time stamp:', unq_idxs)