# PLOT OPERATIONS FUNCTIONS UNIT TESTS
"""
Contains tests for fns.plot functions
"""

#############################
## Packages
#############################
import pytest
import sys
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('../src/'))
import fns.plot as plot

#############################
## Define Fixtures
#############################
@pytest.fixture
def bigmatrix():
	return np.random.rand(10,7)

@pytest.fixture
def smallmatrix():
	return np.random.rand(6,3)

#############################
## Process Ticks
#############################
class Test_split_ticks():
	"""tests :func:`fns.plot.split_ticks`"""

	def test_runs_ZMinorTicks(self):
		"""tests that code runs to completion"""
		try:
			plot.split_ticks(labels=['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7'], ticks=[1, 2, 3, 4, 5, 6, 7])
		except:
			assert False, "Exception raised for plot.split_ticks(labels: npt.ArrayLike, ticks: npt.ArrayLike)."
		else:
			assert True

#############################
## Colormap
#############################
class Test_custom_colormap():
	"""tests :func:`fns.plot.custom_colormap`"""

	def test_invalid_color1(self):
		"""tests that assertion error is raised when color1 is invalid"""
		with pytest.raises(AssertionError):
			plot.custom_colormap(color1='invalid', color2='red', alpha1=0.0, alpha2=1.0)

	def test_invalid_color2(self):
		"""tests that assertion error is raised when color2 is invalid"""
		with pytest.raises(AssertionError):
			plot.custom_colormap(color1='red', color2='invalid', alpha1=0.0, alpha2=1.0)

	def test_invalid_alpha1(self):
		"""tests that assertion error is raised when alpha1 is invalid"""
		with pytest.raises(AssertionError):
			plot.custom_colormap(color1='yellow', color2='red', alpha1=-1.0, alpha2=1.0)

	def test_invalid_alpha2(self):
		"""tests that assertion error is raised when alpha2 is invalid"""
		with pytest.raises(AssertionError):
			plot.custom_colormap(color1='yellow', color2='red', alpha1=0.0, alpha2=8.0)

	def test_registed_my_cmap(self):
		"""tests that the function registers the 'my_cmap' colormap"""
		plot.custom_colormap(color1='yellow', color2='red', alpha1=0.0, alpha2=1.0)
		assert 'my_cmap' in plt.colormaps()

	def test_registed_new_cmap(self):
		"""tests that the function registers the 'new_cmap' colormap"""
		plot.custom_colormap(color1='yellow', color2='red', alpha1=0.0, alpha2=1.0)
		assert 'new_cmap' in plt.colormaps()

#############################
## Feature Over Field
#############################
class Test_feature_plot():
	"""tests :func:`fns.plot.feature_plot`"""

	def test_invalid_resize(self, bigmatrix, smallmatrix):
		"""tests that an assertion error is raised when an invalid resize option is passed"""
		with pytest.raises(AssertionError):
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='invalid')

	def test_invalid_cmap(self, bigmatrix, smallmatrix):
		"""tests that an assertion error is raised when an invalid c_map option is passed"""
		with pytest.raises(AssertionError):
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='invalid',
				resize='scale')

	def test_feature_larger_than_field(self, bigmatrix, smallmatrix):
		"""tests that an assertion error is raised when the feature is larger than the field"""
		with pytest.raises(AssertionError):
			plot.feature_plot(field=smallmatrix, ft=bigmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='scale')

	def test_fail_resize_none(self, bigmatrix, smallmatrix):
		"""tests that an assertion error is raised when resize='None' and feature and field do not have identical dimensions"""
		with pytest.raises(AssertionError):
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='None')

	def test_runs_resize_pad(self, bigmatrix, smallmatrix):
		"""tests that code runs to completion with resize='pad' """
		try:
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='pad')
		except:
			assert False, "Exception raised for plot.feature_plot(...)."
		else:
			assert True

	def test_runs_resize_cut(self, bigmatrix, smallmatrix):
		"""tests that code runs to completion with resize='cut' """
		try:
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='cut')
		except:
			assert False, "Exception raised for plot.feature_plot(...)."
		else:
			assert True

	def test_runs_resize_scale(self, bigmatrix, smallmatrix):
		"""tests that code runs to completion with resize='scale' """
		try:
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='scale')
		except:
			assert False, "Exception raised for plot.feature_plot(...)."
		else:
			assert True

	def test_runs_resize_none(self, bigmatrix):
		"""tests that code runs to completion with resize='None' """
		try:
			plot.feature_plot(field=bigmatrix, ft=bigmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=False, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='None')
		except:
			assert False, "Exception raised for plot.feature_plot(...)."
		else:
			assert True

	def test_runs_RMinorTicks(self, bigmatrix, smallmatrix):
		"""tests that code runs to completion with RMinorTicks=True """
		try:
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label1', 'label2', 'label3', 'label4', 'label5'], Rticks=[1, 2, 3, 4, 5],
				Zlabels=['label'], Zticks=[1],
				RMinorTicks=True, ZMinorTicks=False,
				lims=[0,1],
				c_map='RdBu',
				resize='pad')
		except:
			assert False, "Exception raised for plot.feature_plot(...)."
		else:
			assert True

	def test_runs_ZMinorTicks(self, bigmatrix, smallmatrix):
		"""tests that code runs to completion with RMinorTicks=True """
		try:
			plot.feature_plot(field=bigmatrix, ft=smallmatrix, 
				Rlabels=['label'], Rticks=[1],
				Zlabels=['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7'], Zticks=[1, 2, 3, 4, 5, 6, 7],
				RMinorTicks=False, ZMinorTicks=True,
				lims=[0,1],
				c_map='RdBu',
				resize='pad')
		except:
			assert False, "Exception raised for plot.feature_plot(...)."
		else:
			assert True

#############################
## Heatmap Plot
#############################
class Test_heatmap_plot():
	"""tests :func:`fns.plot.heatmap_plot`"""

	def test_runs_upclim05(self):
		"""tests that code runs to completion when upper color limit is 0.5 """
		try:
			plot.heatmap_plot(matrix=0.25*np.ones((3,2)),
				xticklabels=['label1', 'label2'], yticklabels=['label1', 'label2', 'label3'],
				xlabel='xlabel', ylabel='ylabel',
				title='title',
				save_path='./test',
				grid=False)
			os.remove('./test.png')
		except:
			assert False, "Exception raised for plot.heatmap_plot(...)."
		else:
			assert True

	def test_runs_upclim10(self):
		"""tests that code runs to completion when upper color limit is 1.0 """
		try:
			plot.heatmap_plot(matrix=0.75*np.ones((3,2)),
				xticklabels=['label1', 'label2'], yticklabels=['label1', 'label2', 'label3'],
				xlabel='xlabel', ylabel='ylabel',
				title='title',
				save_path='./test',
				grid=False)
			os.remove('./test.png')
		except:
			assert False, "Exception raised for plot.heatmap_plot(...)."
		else:
			assert True

	def test_runs_lowclim_neg(self):
		"""tests that code runs to completion when lower color limit is negative """
		try:
			plot.heatmap_plot(matrix=-0.75*np.ones((3,2)),
				xticklabels=['label1', 'label2'], yticklabels=['label1', 'label2', 'label3'],
				xlabel='xlabel', ylabel='ylabel',
				title='title',
				save_path='./test',
				grid=False)
			os.remove('./test.png')
		except:
			assert False, "Exception raised for plot.heatmap_plot(...)."
		else:
			assert True

	def test_runs_grid_true(self):
		"""tests that code runs to completion when grid=True """
		try:
			plot.heatmap_plot(matrix=np.random.rand(3,2),
				xticklabels=['label1', 'label2'], yticklabels=['label1', 'label2', 'label3'],
				xlabel='xlabel', ylabel='ylabel',
				title='title',
				save_path='./test',
				grid=True)
			os.remove('./test.png')
		except:
			assert False, "Exception raised for plot.heatmap_plot(...)."
		else:
			assert True
