# MATRIX OPERATIONS FUNCTIONS
"""
Contains functions that can be applied to any matrix. Includes functions that normalize, resize, and concatenate matricies.
"""

#############################
## Packages
#############################
import numpy as np
import numpy.typing as npt
import pandas as pd
import math
import skimage


#############################
## Normalize Matricies
#############################
def normalize01(item: npt.ArrayLike):
	""" Function to normalize an array between 0 and 1

	    Args:
	        item (npt.ArrayLike): array to be normalized (typically rho, activation features, etc)

	    Returns:
			item (npt.ArrayLike): normalized array
    """
	item = (item - np.amin(item)) / (np.amax(item) - np.amin(item))
	return item

def normalize_mat(mat: npt.ArrayLike, norm: str):
	""" Function to extract the features from a given layer of a model

		Args:
			mat (npt.ArrayLike): 3D tensor to be normalized
			norm (str): {'ft01', 'all01', 'none'} 
				a string to indicate which normalization methodology to use; 
				*'ft01'* normalizes each tensor element between 0 and 1 (using min/max of that element);
				*'all01'* normalizes entire tensor between 0 and 1 (using min/max of entire tensor);
				*'none'* does not normalize the tensor

		Returns:
			new_mat (np.array): normalized tensor

		See Also:
			:func:`fns.mat.normalize01`
    """
	## Check Norm is Valid
	assert norm in ['ft01', 'all01', 'none'], "norm must be one of {'ft01', 'all01', 'none'}."

	new_mat = np.empty(np.shape(mat))
	if norm == 'ft01':
		d = np.shape(mat)[-1]
		for i in range(d):
			new_mat[:,:,i] = normalize01(mat[:,:,i])
	elif norm == 'all01':
		new_mat = normalize01(mat)
	elif norm == 'none':
		new_mat = mat

	return new_mat

#############################
## Resize Matricies
#############################
def matrix_padding(mat: npt.ArrayLike, x: int, y: int):
	""" Function to add zero padding to a matrix to match some size

		Dimensions of mat must be smaller than (y, x)

	    Args:
	        mat (npt.ArrayLike): 2D array
	        x (int): desired width (no. of columns)
	        y (int): desited height (no. of rows)

	    Returns:
			mat_pad (np.ndarray): an array with new size (y, x)
    """
    ## Check Desired Size
	mat_y, mat_x = np.shape(mat)
	assert x >= mat_x, 'Desired size x must be larger than the dimension x of the input matrix.'
	assert y >= mat_y, 'Desired size y must be larger than the dimension y of the input matrix.'

	## Check that Padding is Symmetric
	pad_x = (x - mat_x)/2
	pad_y = (y - mat_y)/2
	assert pad_x%1 == 0, 'Desired size does not allow for symmetric padding along x axis: change x.'
	assert pad_y%1 == 0, 'Desired size does not allow for symmetric padding along y axis: change y.'

	## Add Padding
	pad_x = int(pad_x)
	pad_y = int(pad_y)
	mat_pad = np.pad(mat, ((pad_y, pad_y), (pad_x, pad_x)), 'constant', constant_values=0)
		
	return mat_pad

def matrix_cutting(mat: npt.ArrayLike, x: int, y: int):
	""" Function to cut a matrix down to match some size

		Dimensions of mat must be larger than (y, x)

	    Args:
	        mat (npt.ArrayLike): 2D array
	        x (int): desired width (no. of columns)
	        y (int): desired height (no. of rows)

	    Returns:
			mat_cut (npt.ArrayLike): an array with new size (y, x)
    """
	## Check Desired Size
	mat_y, mat_x = np.shape(mat)
	assert x <= mat_x, 'Desired size x must be smaller than the dimension x of the input matrix.'
	assert y <= mat_y, 'Desired size y must be smaller than the dimension y of the input matrix.'
	
	## Check that Cutting is Symmetric
	pad_x = (mat_x - x)/2
	pad_y = (mat_y - y)/2
	assert pad_x%1 == 0, 'Desired size does not allow for symmetric cutting along x axis: change x.'
	assert pad_y%1 == 0, 'Desired size does not allow for symmetric cutting along y axis: change y.'
	
	## Cut Matrix
	pad_x = int(pad_x)
	pad_y = int(pad_y)
	mat_cut = mat[pad_y:mat_y-pad_y, pad_x:mat_x-pad_x]

	return mat_cut

def matrix_scale(mat: np.ndarray, x: int, y: int):
	""" Function to scale a matrix to match some size

	    Args:
	        mat (np.ndarray): single array (2D)
	        x (int): desired width (no. of columns)
	        y (int): desired height (no. of rows)

	    Returns:
			mat_scaled (np.ndarray): an array with new size (x, y)
    """
	mat_scaled = skimage.transform.resize(mat, (y, x))
	return mat_scaled

#############################
## Add to Matricies
#############################
def concat_avg(matrix: npt.ArrayLike, axis: int = 0, spacer: bool = True):
	"""	Function to expand an array to include the absolute value averages of it's rows or columns, with or without a spacer of zeros between original matrix and averages
		
		Args:
			matrix (npt.ArrayLike): starting array
			axis (int): {0, 1}: axis to take average along,  =0 for columns, =1 for rows
			spacer (bool): whether or not to include a spacer of zeros between original matrix and averages
		
		Returns:
			avg_matrix (np.ndarray): a copy of the starting matrix with the average concatenated
	"""
	## Check Valid Axis
	assert axis in [0, 1], "Axis must be 0 or 1."

	y, x = matrix.shape

	## Make Averages
	if axis==0: avg = np.reshape(np.mean(np.abs(matrix), axis=axis), (1, x))
	elif axis==1: avg = np.reshape(np.mean(np.abs(matrix), axis=axis), (y, 1))

	## Concatenate Averages & Spacer
	spcr = np.zeros_like(avg)
	if spacer:
		if axis==0: avg_matrix = np.concatenate((avg, spcr, matrix), axis=axis)
		elif axis==1: avg_matrix = np.concatenate((matrix, spcr, avg), axis=axis)
	else:
		if axis==0: avg_matrix = np.concatenate((avg, matrix), axis=axis)
		elif axis==1: avg_matrix = np.concatenate((matrix, avg), axis=axis)

	return avg_matrix

#############################
## Matrix Correlation
#############################
def scalar_2Dcorr(f: npt.ArrayLike, g: npt.ArrayLike):
	""" Function to compute the sclar cross-correlation of two 2D arrays

		| Uses first formula from https://en.wikipedia.org/wiki/Digital_image_correlation_and_tracking considering i = j = 0
		| Equivalent to pearson correlation coefficient

        Args:
            f (npt.ArrayLike): 2D array
            g (npt.ArrayLike): 2D array

        Returns:
			r (float): 2D discrete cross correlation 

    """
	f_bar = np.mean(f)
	g_bar = np.mean(g)

	num = np.sum(np.multiply(f-f_bar, g-g_bar))
	den = math.sqrt(np.sum(np.square(f-f_bar)) * np.sum(np.square(g-g_bar)))

	r = num / den
	return r

#############################
## Extracting Statistically Significant Values
#############################
def get_statsig(corr: npt.ArrayLike, pvals: npt.ArrayLike, threshold: float=0.05):
	"""	Function to extract the statistically significant values from a matrix of correlation coefficients
		
		Args:
			corr (npt.ArrayLike): matrix of correlation coefficients
			pvals (npt.ArrayLike): matrix of p-values corresponding to coorrelation coefficients
			threshold (float): p-value cuttoff for significance 
								(if p-val > threshold, correlation coefficient is discarded)
		
		Returns:
			statsig (npt.ArrayLike): a matrix of correlation coefficients with non-stattistically-significant values set to zero
	"""
	statsig = corr.copy()
	for i in range(pvals.shape[0]):
		for j in range(pvals.shape[1]):
			if pvals[i, j] > threshold:
				statsig[i, j] = 0.0

	return statsig