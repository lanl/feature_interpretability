# SAVE FUNCTIONS
"""
Contains functions to save neural network outputs, such as features.
"""

#############################
## Packages
#############################
import glob
import numpy as np
import random
import pandas as pd
import math

############################
## Make File List
#############################
def makefilelist(search_dir: str, num_samples: str, save_path: str='None', save: bool=True):
	""" Function to make list of .npz files to sample from

        Args:
            search_dir (str): 	file path to directory to search for samples in; include any restrictions for file name
            num_samples (str): integer number of samples to include, or 'All' for all samples found
            save_path (str): path to save .txt file contianing list of samples
            save (bool): boolean for if the sample list is saved to a .txt file

        Returns:
			num_samples (int): number of samples in the sample list
			sample_list (np.ndarray[str]): array of file paths to .npz samples; if save=True,sample_list is also saved to a .txt file
    """
	assert num_samples=='All' or num_samples.isdigit(), "Number of samples must be 'All' or a digit."

	sample_list = glob.glob(search_dir)
	sample_list = [s.split('/')[-1] for s in sample_list]
	sample_list = np.unique(sample_list).tolist()
	if num_samples == 'All':
		num_samples = np.size(sample_list)
	else:
		num_samples = int(num_samples)
		sample_list = random.sample(sample_list, k=num_samples)

	if save:
		if save_path == 'None':
			print('None is not a valid save path for makefilelist.')
			print('Either provide a valid save path or use save=False.')
			sys.exit()
		else:
			sample_file = open(save_path+'_SAMPLES.txt', 'w')
			np.savetxt(sample_file, sample_list, fmt='%s')
			sample_file.close()

	return num_samples, sample_list


#############################
## Save to .npz File
#############################
def features2npz(ft_mat: np.ndarray[(any, any, any), float], save_path: str, ft_suffix: str=''):
	""" Function to save extracted features (or other simialr objects) to an .npz file

	    Args:
			ft_mat (np.ndarray[(any, any, any), float]): tensor of all features
			save_path (str): path to save .npz file
			ft_suffix (str): suffix to append to the .npz internal file names (deafult file names are "feature#")
	    
	    Returns:
			No Return Objects
    """
	n_fts = ft_mat.shape[-1]
	save_dict = dict()

	for i in range(n_fts):
		key = 'feature'+str(i+1)+ft_suffix
		save_dict[key] = ft_mat[:,:,i]

	np.savez(save_path+'.npz', **save_dict)

def fields2npz(fld_mat: np.ndarray[(any, any, any), float], fields: list[str], save_path: str, suffix: str=''):
	""" Function to save extracted features (or other simialr objects) to an .npz file

	    Args:
			fld_mat (np.ndarray[(any, any, any), float]): tensor of all fields
			fields (list[str]): names of fields
			save_path (str): path to save .npz file
			suffix (str): suffix to append to the .npz internal file names (deafult file names are "field")
	    
	    Returns:
			No Return Objects
    """
	n_fields = fld_mat.shape[-1]
	save_dict = dict()

	for i in range(n_fields):
		key = fields[i]+suffix
		save_dict[key] = fld_mat[:,:,i]

	np.savez(save_path+'.npz', **save_dict)
