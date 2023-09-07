# COUPON*DATA PROCESS FUNCTIONS
"""
Contains functions to process the data from the HE driven coupon experiments.
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
			unq_tpls (list[str]): list of unique keys that fix experimental parameters (tpl)
			unq_idxs (list[str]): list of unique keys that fix a time stamp (idx)
	"""
	## Seperate out keys
	all_tpls = [i.split('_')[1] for i in sample_list]
	unq_tpls = np.unique(all_tpls)
	all_idxs = [i.split('_')[-1].split('.')[0] for i in sample_list]
	unq_idxs = np.unique(all_idxs)

	return unq_tpls, unq_idxs

def findkey(sample_list: list[str]):
	""" Function that finds the fixed key given a list of samples

		Args:
			sample_list (list[str]): list of file paths to .npz samples
		
		Returns:
			fixed_key (str): the fixed key found in the samples; returns 'None' if no fixed key is found
	"""
	unq_tpls, unq_idxs = listkeys(sample_list)

	if np.size(unq_tpls) == 1:
		fixed_key = unq_tpls[0]
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
			key (str): 	the study information for the simulation that generated the .npz file; of the form "r60um_tpl###"
    """
	key = npz_file.split('/')[-1].split('_')
	key = '_'.join(key[0:2])
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
def csv2dcj(csv_file: str, key:str):
	""" Function to extract the dcj value from the design .csv file given the study key
	    
	    Args:
	        csv_file (str): file path from working directory to the .csv design file
	        key (str): 	the study information for a given simulation; of the form "r60um_tpl###"
	    
	    Returns:
			dcj (np.ndarray[(1), float]): an array of size one with the dcj (explosive velocity) value for a given simulation
    """
	design_file = pd.read_csv(csv_file,
								sep=',',
                            	header=0,
                            	index_col=0,
                            	engine='python')
	Dcj = design_file.loc[key, 'Dcj']
	Dcj = np.array([float(Dcj.split(' ')[3])])
	return Dcj

#######################################
## Retrieve Data Values from .npz file
#######################################
def remove_fields(npz: np.lib.npyio.NpzFile, fields: list[str], NoneValid: bool=False):
	""" Function to extract a normalized field "picture" array from an .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file
	        fields (list[str]): array of fields supplied by user
	        NoneValid (bool): indicates if the blank field "none" is a valid selection for the program
	    
	    Returns:
			fields (list[str]): list of fields that are valid for correlation purposes
			n_fields (int): number of valid fields
	"""
	if fields == ['All']:
		fields = npz.files
		fields.remove('sim_time')
		fields.remove('melt_state')
		fields.remove('hr_coupon')
		fields.remove('hr_maincharge')
		fields.remove('rVel')
		fields.remove('zVel')
		fields.remove('Rcoord')
		fields.remove('Zcoord')

		if NoneValid:
			fields.append('none')

	if not NoneValid:
		try: 
			fields.remove('none')
		except:
			pass
		else:
			print('Field="none" is not a valid selection for this program; "none" has been removed from the list of fields.')
	
	n_fields = np.size(fields)

	return fields, n_fields

def get_field(npz: np.lib.npyio.NpzFile, field: str):
	""" Function to extract a normalized field "picture" array from an .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file
	        field (str): name of field to extract
	    
	    Returns:
			pic (np.ndarray[(300, 1000), float]): normalized field 
    """
	pic = npz[field]
	# Flip pRad to be hte right side up
	if field == 'pRad':
		pic = np.flipud(pic)
	# Create reflection if needed
	if field != 'pRad':
		pic = np.concatenate((np.fliplr(pic), pic), axis=1)
	# Crop
	pic = pic[0:300, 100:1100]
	# Normalize
	if field != 'melt_state':
		pic = fns.mat.normalize01(pic)
	return pic

def get_ticks(npz: np.lib.npyio.NpzFile):
	""" Function to generate axis tick markers and labels from the .npz file

	    Args:
	        npz (np.lib.npyio.NpzFile): a loaded .npz file
	    
	    Returns:
	        Rlabels (np.ndarray[(any), float]): Array of labels for radial axis; 
	        									looks like [max --> 0 --> max], in increments of 1cm
	        Rticks (np.ndarray[(any), float]):  Array of pixel locations where tick marks go, corresponding to Rlabels
	        Zlabels (np.ndarray[(any), float]): Array of labels for vertical axis; 
	        									looks like [0 --> max], in increments of 1cm
	        Zticks (np.ndarray[(any), float]): Array of pixel locations where tick marks go, corresponding to Zlabels
	"""

	Rcoord = npz['Rcoord']
	Rcoord = Rcoord[0:500] #crop, only half
	R_max = round(max(Rcoord), 2)
	Rlabel = np.arange(0,R_max+1,1)
	Rlabels = np.concatenate((np.flip(Rlabel), Rlabel[1:]))
	Raxis = np.shape(Rlabels)[0]
	Rticks = np.linspace(start=0, stop=1000, num=Raxis)

	Zcoord = npz['Zcoord']
	Zcoord = Zcoord[0:300]
	Z_max = round(max(Zcoord), 2)
	Zlabels = np.arange(0,Z_max+1,1)
	Zaxis = np.shape(Zlabels)[0]
	Zticks = np.linspace(start=0, stop=300, num=Zaxis)

	return Rlabels, Rticks, Zlabels, Zticks
