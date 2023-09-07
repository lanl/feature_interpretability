# PLOT FUNCTIONS
"""
Contains functions relate to or make matplotlib.pyplot plots. 
Includes functions to create custom colormaps, plot features, and plot heatmaps.
"""

#############################
## Packages
#############################
import os
import sys
import glob
import random
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib as mpl
import matplotlib.image
import matplotlib.pyplot as plt

import fns


#############################
## Process Ticks
#############################
def split_ticks(labels: npt.ArrayLike, ticks: npt.ArrayLike):
	""" Function split a set of axis ticks and labels into major and minor ticks and labels

        Args:
	        labels (npt.ArrayLike): Array of labels for radial axis
	        ticks (npt.ArrayLike):  Array of pixel locations where tick marks go, corresponding to labels
        
        Returns:
			MajorLabels (npt.ArrayLike): Array of labels for major radial axis
			MajorTicks (npt.ArrayLike):  Array of pixel locations where major axis tick marks go, corresponding to MajorLabels
			MinorTicks (npt.ArrayLike):  Array of pixel locations where minor tick marks go, corresponding to labels not included in MajorLabels
    """
    #Major Axis
	MajorLabels = labels[1:-1:2]
	MajorTicks = ticks[1:-1:2]

	#Minor Axis
	MinorTicks = ticks[0:-1:2]

	return MajorLabels, MajorTicks, MinorTicks

#############################
## Colormap
#############################
def custom_colormap(color1: str, color2: str, alpha1: float, alpha2: float):
    """ Function to make a custom colormap for values [0,1]

        Function creates two global callable mpl colormaps:

        - Colormap Object c_map with name "my_cmap" is a two color gradient from color1 --> color2, with an alpha1 --> alpha2 opacity gradient
        - Colormap Object new_cmap with name "new_cmap" is a two color gradient from color1 --> color2, with an alpha1 --> alpha2 opacity gradient, with the zero value replaced with 100% transparent white
        
        Args:
            color1 (str): name of the first color in the color gradient, must be in the matplotlib CSS4 color list
            color2 (str): name of the last color in the color gradient, must be in the matplotlib CSS4 color list
            alpha1 (float): [0,1] value of first opacity in the gradient (0=full transparency, 1=full opacity)
            alpha2 (float): [0,1] value of last opacity in the gradient (0=full transparency, 1=full opacity)
        
        Returns:
            No Return Objects
    """
    ## Check that inputs are valid
    assert color1 in mpl.colors.CSS4_COLORS.keys(), "Color arguments must be in the matplotlib CSS4 color list."
    assert color2 in mpl.colors.CSS4_COLORS.keys(), "Color arguments must be in the matplotlib CSS4 color list."
    assert alpha1 <= 1 and alpha1 >= 0, "Alpha arguments must be in [0, 1]."
    assert alpha2 <= 1 and alpha1 >= 0, "Alpha arguments must be in [0, 1]."

    #Make a color map from color1 at opactiy1 to color2 at opacity2
    mpl.colormaps.unregister('my_cmap')
    mpl.colormaps.unregister('new_cmap')
    c_color1 = mpl.colors.colorConverter.to_rgba(color1, alpha = alpha1)
    c_color2 = mpl.colors.colorConverter.to_rgba(color2, alpha = alpha2)
    c_map = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',[c_color1,c_color2])
    mpl.colormaps.register(cmap=c_map)

    #Make feature==0 completely transparent
    c_map = mpl.colormaps['my_cmap'].resampled(256)
    c_mat = c_map(np.linspace(0, 1, 256))
    c_mat[0,:] = (1.0, 1.0, 1.0, 0.0)
    new_cmap = mpl.colors.ListedColormap(c_mat, name="new_cmap")
    mpl.colormaps.register(cmap=new_cmap)


#############################
## Feature Over Field
#############################
def feature_plot(field: np.ndarray[(any, any), float], 
				ft: np.ndarray[(any, any), float],
				Rlabels: np.ndarray[str], Rticks: np.ndarray[int], 
				Zlabels: np.ndarray[str], Zticks: np.ndarray[int], 
				RMinorTicks: bool=False, ZMinorTicks: bool=False, 
				lims: list[float, float]=[0,1], 
				c_map: str='new_cmap', 
				resize: str='scale'):
    """ Function to plot one extracted feature over a given hydrodynamic field
        
        Function will plot feature over field image on existing mpl axes and will adjust axis ticks/labels appropriately

        Args:
			field (np.ndarray[(any, any), float]): picture to underlay the feature (typically radiographic fields)
			ft (np.ndarray[(any, any), float]):  single extracted feature array; assumed the ft array is the same size or smaller than the field array
			Rlabels (np.ndarray[str]): Array of labels for radial axis
			Rticks (np.ndarray[int]): Array of pixel locations where tick marks go, corresponding to Rlabels
			Zlabels (np.ndarray[str]): Array of labels for vertical axis
			Zticks (np.ndarray[int]): Array of pixel locations where tick marks go, corresponding to Zlabels
			lims (list[float, float]): array of colorbar limits
			c_map (str): name of color map to plot the activation intensity with; default "new_cmap" is a custom colormap creaated by the custom_colormap function
			resize (str):	{'cut', 'pad', 'scale', 'None'}
							method of resizing feature and field arrays to be the same size;
							*'pad'* adds zeros to border of feature;
							*'cut'* removes borders of field;
							*'scale'* upsamples feature to be same size as field;
							*'None'* implies the feature and field are already the same size
        
        Returns:
            im (mpl.image.AxesImage): the mpl image object of just the feature plot (does NOT contain field_pic image)

    	See Also:
    		:func:`fns.coupondata.get_ticks` and :func:`fns.nestedcylinderdata.get_ticks` to generate Rlabels, Rticks, Zlabels, Zticks
    		
    		:func:`fns.plot.custom_colormap` to generate colormap "new_cmap"
    """
    ## Get Sizes
    fd_y, fd_x = np.shape(field)
    ft_y, ft_x = np.shape(ft)
    lim_low = lims[0]
    lim_high = lims[1]

    ## Check Arguments
    assert resize in ['cut', 'pad', 'scale', 'None'], "resize must be one of {'cut', 'pad', 'scale', 'None'}."
    assert c_map in plt.colormaps(), "c_map passed must be a valid matplotlib colormap."
    assert (fd_x >= ft_x) and (fd_y >= ft_y), "Dimensions of field must be larger or equal to the size of feature."

    ## Resize Images
    if resize == 'pad':
    	ft = fns.mat.matrix_padding(ft, fd_x, fd_y)
    	plot_x, plot_y = fd_x, fd_y
    elif resize == 'cut':
    	field = fns.mat.matrix_cutting(field, ft_x, ft_y)
    	plot_x, plot_y = ft_x, ft_y
    elif resize == 'scale':
    	ft = fns.mat.matrix_scale(ft, fd_x, fd_y)
    	plot_x, plot_y = fd_x, fd_y
    elif resize == 'None':
    	assert (fd_x==ft_x) and (fd_y==ft_y), "When resize='None', feature and field must be the same size.\nSelect differnt rezsize method."
    	plot_x, plot_y = fd_x, fd_y

    ## Make Plot
    #X-Axis
    plt.xlim([0, plot_x])
    if RMinorTicks:
    	RMajorLabels, RMajorTicks, RMinorTicks = split_ticks(Rlabels, Rticks)
    	plt.xticks(ticks=RMajorTicks,labels=RMajorLabels, fontsize=12, rotation=45, minor=False)
    	plt.xticks(ticks=RMinorTicks, fontsize=12, rotation=45, minor=True)
    else:
    	plt.xticks(ticks=Rticks,labels=Rlabels, fontsize=12, rotation=45)
    plt.xlabel('cm', fontsize=12,x=1,y=1)
    plt.axvline(500, color = 'k',linestyle=':', alpha=0.5)

    #Y-Axis
    plt.ylim([0, plot_y])
    if ZMinorTicks:
    	ZMajorLabels, ZMajorTicks, ZMinorTicks = split_ticks(Zlabels, Zticks)
    	plt.yticks(ticks=ZMajorTicks,labels=ZMajorLabels, fontsize=12, minor=False)
    	plt.yticks(ticks=ZMinorTicks, fontsize=12, minor=True)
    else:
    	plt.yticks(ticks=Zticks,labels=Zlabels, fontsize=12)
    plt.ylabel('cm', fontsize=12,y=1.05,rotation=0)

    #Images
    plt.grid(False)
    plt.imshow(field, cmap='gray_r', origin='lower')
    im = plt.imshow(ft, cmap=c_map, origin='lower', vmin=lim_low, vmax=lim_high)

    return im

#############################
## Heatmap Plot
#############################
def heatmap_plot(matrix:np.ndarray[(any, any), float],
				xticklabels: np.ndarray[str], yticklabels: np.ndarray[str],
				xlabel: str, ylabel: str, 
				title:str, 
				save_path: str, 
				grid: bool=False):
	"""	Function to plot a matrix heatmap with labeled cells

		Args:
			matrix (np.ndarray[(any, any), float]): matrix to plot
			xticklabels (np.ndarray[str]): array of tick labels for the x-axis
			yticklabels (np.ndarray[str]): array of tick labels for the y-axis
			xlabel (str): label for the x-axis
			ylabel (str): label for the y-axis
			title (str): title for plot
			save_path (str): path to save figure to
			grid (bool): determines if the matrix grid is visible or not
		
		Returns:
			No Return Objects
	"""

	## Set the Upper Colorbar Limit
	if max(np.abs(matrix.flatten())) <= 0.5:
		up_clim = 0.5
	else:
		up_clim = 1.0
	
	## Set the Lower Colorbar Limit
	if min(matrix.flatten()) < 0:
		low_clim = -up_clim
		c_map = 'RdBu'
	else:
		low_clim = 0
		c_map = 'Reds'

	## Making the Plot
	y, x = matrix.shape
	plt.rcParams.update({'font.size': 30})
	fig, ax = plt.subplots(figsize=(4*x+3,2*y),layout="constrained")
	im = ax.imshow(matrix, cmap=c_map)
	ax.set_xticks(ticks=np.arange(x),labels=xticklabels, rotation=45, ha='right');
	ax.set_yticks(ticks=np.arange(y),labels=yticklabels);
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	im.set_clim([low_clim, up_clim])
	plt.colorbar(im, label='Correlation Coefficient', fraction=0.07, pad=0.04)
	ax.set_title(title)

	if grid:
		plt.vlines(np.arange(0.5, x-0.5, 1), -0.5, y-0.5, colors='#DDDDDD')
		plt.hlines(np.arange(0.5, y-0.5, 1), -0.5, x-0.5, colors='#DDDDDD')

    ## Adding Value Labels
	for i in range(y):
		for j in range(x):
			if matrix[i,j]==0: #Don't label if it's zero
				pass
			elif np.abs(matrix[i,j]) > (up_clim/2):
				ax.text(j, i, round(matrix[i, j], 2), ha='center', va='center', color='w', fontsize=20)
			else:
				ax.text(j, i, round(matrix[i, j], 2), ha='center', va='center', color='k', fontsize=20)

	fig.savefig(save_path+'.png')
	plt.close()

