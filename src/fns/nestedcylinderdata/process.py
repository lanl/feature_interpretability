# NESTED*CYINDER*DATA PROCESS SCRIPT
"""
Contains functions to process the data from the nested cylinder experiments.
"""

#############################
## Packages
#############################
import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import math

import fns

#####################################
## Retrieve Info from a Directory of Input Files
#####################################
def listkeys(sample_list: list[str]):
	""" Function that makes lists of unique variable indexes in a given directory

		Args:
			sample_list (list[str]): list of file paths to .npz samples

		Returns:
			unq_tpls (list[str]): list of unique keys that fix the strength model scaling factor (ptw)
			unq_idxs (list[str]): list of unique keys that fix a time stamp (idx)
	"""
	## Seperate out keys
	all_ptws = [i.split('_')[2] for i in sample_list]
	unq_ptws = np.unique(all_ptws)

	all_idxs = [i.split('_')[-1].split('.')[0] for i in sample_list]
	unq_idxs = np.unique(all_idxs)

	return unq_ptws, unq_idxs

def findkey(sample_list: list[str]):
	""" Function that finds the fixed key given a list of samples

		Args:
			sample_list (list[str]): list of file paths to .npz samples

		Returns:
			fixed_key (str): the fixed key found in the samples
	"""
	unq_ptws, unq_idxs = listkeys(sample_list)

	if np.size(unq_ptws) == 1:
		fixed_key = unq_ptws[0]
	elif np.size(unq_idxs) ==  1:
		fixed_key = unq_idxs[0]
	else:
		fixed_key = 'None'

	return fixed_key

#####################################
## Retrieve Info from .npz file name
#####################################
def npz2key(npz_file: str):
	""" Function to extract study information from the name of an .npz file

	    Args:
	        npz_file (str): file path from working directory to .npz file
	    
	    Returns:
			key (str): 	the study information for the simulation that generated the .npz file; of the form "# ncyl_sclPTW_###"
    """
	key = npz_file.split('/')[-1].split('_')
	key = '_'.join(key[0:3])
	return key

def npz2idx(npz_file: str):
	""" Function to extract the time stamp from the name of an .npz file

	    Args:
	        npz_file (str): file path from working directory to .npz file
	    
	    Returns:
			idx (str): 	the idx component of the file name; of the form "idx#####"
	"""
	idx = npz_file.split('/')[-1].split('_')[-1]
	idx = idx.split('.')[0]
	return idx

############################################
## Retrive Info from Master Study .csv file
############################################
def csv2scalePTW(csv_file: str, key:str):
	""" Function to extract the dcj value from the design .csv file given the study key
	    
	    Args:
	        csv_file (str): file path from working directory to the .csv design file
	        key (str): 	the study information for a given simulation; of the form "# ncyl_sclPTW_###"
	    
	    Returns:
			ptw (float): the PTW scaling value used to a given simulation
    """
	design_df = pd.read_csv(csv_file,
								sep=', ',
                            	header=0,
                            	index_col=0,
                            	engine='python')
	ptw = design_df.at[key, 'ptw_scale']
	return ptw

#######################################
## Retrieve Data Values from .npz file
#######################################
def remove_fields(npz: np.lib.npyio.NpzFile, fields: list[str], NoneValid: bool=False):
	""" Function to extract a normalized field "picture" array from an .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file
	        fields (list[str]): list of fields supplied by user
	        NoneValid (bool): indicates if the blank field "none" is a valid selection for the program

	    Returns:
			file_fields (list[str]): list of fields that are valid for correlation purposes
			n_fields (int): number of valid fields
	"""
	## Make list of Fields
	if fields == ['All']:
		file_fields = npz.files

		# Remove non-matrix fields
		file_fields.remove('sim_time')
		file_fields.remove('porosity')
		file_fields.remove('melt_state')
		file_fields.remove('rVel')
		file_fields.remove('zVel')
		file_fields.remove('Rcoord')
		file_fields.remove('Zcoord')

		# Remove partial density fields
		indexing_fields = file_fields.copy()
		for f in indexing_fields:
			if ('hr_' in f) and (f != 'hr_MOICyl'): file_fields.remove(f)

		if NoneValid:
			file_fields.append('none')

	else:
		file_fields = fields

	## Deal with 'none' field
	if not NoneValid:
		try: 
			fields.remove('none')
		except:
			pass
		else:
			print('Field="none" is not a valid selection for this program; "none" has been removed from the list of fields.')
	
	n_fields = np.size(file_fields)

	return file_fields, n_fields


def get_field(npz: np.lib.npyio.NpzFile, field: str):
	""" Function to extract a field "picture" array from an .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file
	        field (str): name of field to extract

	    Returns:
			pic (np.ndarray[(1700, 250), float]): field 
    """
	pic = npz[field]
	pic = pic[800:, :250]
	pic = np.concatenate((np.fliplr(pic), pic), axis=1)
	return pic

def get_ticks(npz: np.lib.npyio.NpzFile):
	""" Function to generate axis tick markers and labels from the .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file

	    Returns:
	        Rlabels (np.ndarray[(any), floar]): Array of labels for radial axis; 
	        									looks like [max --> 0 --> max], in increments of 0.5cm
	        Rticks (np.ndarray[(any), float]):  Array of pixel locations where tick marks go, corresponding to Rlabels
	        Zlabels (np.ndarray[(any), float]): Array of labels for vertical axis; 
	        									looks like [0 --> max], in increments of 1cm
	        Zticks (np.ndarray[(any), float]): Array of pixel locations where tick marks go, corresponding to Zlabels
	"""

	Rcoord = npz['Rcoord']
	Rcoord = Rcoord[0:250]#crop, only half
	R_max = round(max(Rcoord), 1)
	Rlabel = np.arange(0,R_max+1,0.5)[:-1]
	Rlabels = np.concatenate((np.flip(Rlabel), Rlabel[1:]))
	Raxis = np.shape(Rlabels)[0]
	Rticks = np.linspace(start=0, stop=500, num=Raxis)

	Zcoord = npz['Zcoord']
	Zcoord = Zcoord[(2500-1700):]
	Z_min = round(min(Zcoord), 2)
	Z_max = round(max(Zcoord), 2)
	Zlabels = np.arange(Z_min,Z_max+1,1)
	Zaxis = np.shape(Zlabels)[0]
	Zticks = np.linspace(start=0, stop=1700, num=Zaxis)

	return Rlabels, Rticks, Zlabels, Zticks
