# DATA CHECKS FUNCTIONS
"""
Contains functions to check that the data-related input arguments passed are valid
"""


#############################
## Packages
#############################
import glob
import numpy as np

import fns

#############################
## Check Functions
#############################
def check_fields(npz: np.lib.npyio.NpzFile, fields: list[str]):
	""" Function that checks if all fields in a list exist in a npz file

        Args:
            npz (np.lib.npyio.NpzFile): a loaded .npz file
            fields (list[str]): list of the fields to test

        Returns:
			No Return Objects
    """
	all_fields = npz.files
	quit = False
	for field in fields:
		if field == 'none':
			pass
		elif field == 'All':
			pass
		elif field not in all_fields:
			raise ValueError('Field "'+field+'" is not a valid field. \nUse -PF or -PRINT_FIELDS to show a list of valid fields.')
		elif field == 'sim_time':
			raise ValueError('Sim_time is a scalar, and cannot be plotted. \nUse -PF or -PRINT_FIELDS to show a list of valid fields.')

def check_key(fixed_key: str, search_dir: str):
	""" Function that prints a list of unique variable indexes for a given fixed variable

		Args:
			fixed_key (str): the key for samples that is being checked
			search_dir (str): file path to directory to search for samples in

		Returns:
			No Return Objects
	"""
	quit = False
	if 'None' == fixed_key:
		pass
	else:
		n, _ = fns.save.makefilelist(search_dir, num_samples='All', save=False)
		## If no samples are found
		if n == 0:
			raise ValueError('Provided fixed key ('+fixed_key+') is not present in the directory.\nUse --PRINT_KEYS or -PK to show a list of valid fixed keys.')

def check_samples(num_samples: str, search_dir: str):
	""" Function that checks if the number of samples requested are aviailable in a directory

		Args:
			num_samples (str): 	should be 'All', or an integer; number of samples the script will use
			search_dir (str): path to directory of .npz files
		    
		Returns:
			No Return Objects
	"""
	quit = False
	if num_samples == 'All':
		pass
	elif not num_samples.isdigit():
		raise ValueError('Number of samples must be an integer or "All".')
	else:
		num_samples = int(num_samples)
		n, _ = fns.save.makefilelist(search_dir, num_samples='All', save=False)
		if num_samples > n:
			raise ValueError('Number of samples entered is larger than number of samples available in the input directory.')
