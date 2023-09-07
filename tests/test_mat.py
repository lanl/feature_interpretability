# MATRIX OPERATIONS FUNCTIONS UNIT TESTS
"""
Contains tests for fns.mat functions
"""

#############################
## Packages
#############################
import pytest
import sys
import os
import numpy as np
import pandas as pd
import math
import skimage

sys.path.insert(0, os.path.abspath('../src/'))
import fns.mat as mat

#############################
## Define Fixtures
#############################
@pytest.fixture
def matrix1D():
	return np.array([0, 0, 5, 0])

@pytest.fixture
def matrix2D():
	return np.array([[0, 0], [5, 0]])

@pytest.fixture
def matrix3D():
	return np.array([[[0, 0], [5, 0]], [[0, 0], [0, 10]]])

@pytest.fixture
def bigmatrix2D():
	return np.ones((5,5))

#############################
## Normalize Matricies
#############################
class Test_normalize01():
	"""tests :func:`fns.mat.normalize01`"""

	def test_1D(self, matrix1D):
		"""tests on a 1D matrix"""
		assert np.all(mat.normalize01(matrix1D) == np.array([0, 0, 1, 0]))

	def test_2D(self, matrix2D):
		"""tests on a 2D matrix"""
		assert np.all(mat.normalize01(matrix2D) == np.array([[0, 0], [1, 0]]))

	def test_3D(self, matrix3D):
		"""tests on a 3D matrix"""
		assert np.all(mat.normalize01(matrix3D) == np.array([[[0, 0], [0.5, 0]], [[0, 0], [0, 1]]]))


class Test_normalize_mat():
	"""tests :func:`fns.mat.normalize_mat`"""

	def test_ft01(self, matrix3D):
		"""tests norm='ft01', normalizing by feature slice"""
		assert np.all(mat.normalize_mat(matrix3D, 'ft01') == np.array([[[0, 0], [1, 0]], [[0, 0], [0, 1]]]))

	def test_all01(self, matrix3D):
		"""tests norm='all01', normalizing across all dimensions"""
		assert np.all(mat.normalize_mat(matrix3D, 'all01') == np.array([[[0, 0], [0.5, 0]], [[0, 0], [0, 1]]]))

	def test_none(self, matrix3D):
		"""tests norm='none', no normalizing"""
		assert np.all(mat.normalize_mat(matrix3D, 'none') == matrix3D)

	def test_invalid(self, matrix3D):
		"""tests invalid norm string raises assertion error"""
		with pytest.raises(AssertionError):
			mat.normalize_mat(matrix3D, 'invalid')

#############################
## Resize Matricies
#############################
class Test_matrix_padding():
	"""tests :func:`fns.mat.matrix_padding`"""

	def test_x_too_small(self, bigmatrix2D):
		"""tests that a too small desired size in x raises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_padding(bigmatrix2D, x=3, y=7)

	def test_y_too_small(self, bigmatrix2D):
		"""tests that a too small desired size in y raises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_padding(bigmatrix2D, x=7, y=3)

	def test_x_not_symmetric(self, bigmatrix2D):
		"""tests that a desired size in x that does not allow for symmetric padding rasises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_padding(bigmatrix2D, x=6, y=7)

	def test_y_not_symmetric(self, bigmatrix2D):
		"""tests that a desired size in y that does not allow for symmetric padding rasises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_padding(bigmatrix2D, x=7, y=6)

	def test_correct_size(self, bigmatrix2D):
		"""tests that valid desired dimensions return a matrix of the correct size"""
		assert mat.matrix_padding(bigmatrix2D, x=7, y=7).shape == (7,7)

class Test_matrix_cutting():
	"""tests :func:`fns.mat.matrix_cutting`"""

	def test_x_too_big(self, bigmatrix2D):
		"""tests that a too large desired size in x raises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_cutting(bigmatrix2D, x=7, y=3)

	def test_y_too_big(self, bigmatrix2D):
		"""tests that a too large desired size in y raises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_cutting(bigmatrix2D, x=3, y=7)

	def test_x_not_symmetric(self, bigmatrix2D):
		"""tests that a desired size in x that does not allow for symmetric cutting rasises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_cutting(bigmatrix2D, x=4, y=3)

	def test_y_not_symmetric(self, bigmatrix2D):
		"""tests that a desired size in y that does not allow for symmetric cutting rasises an assertion error"""
		with pytest.raises(AssertionError):
			mat.matrix_cutting(bigmatrix2D, x=3, y=4)

	def test_correct_size(self, bigmatrix2D):
		"""tests that valid desired dimensions return a matrix of the correct size"""
		assert mat.matrix_cutting(bigmatrix2D, x=3, y=3).shape == (3,3)

class Test_matrix_scale():
	"""tests :func:`fns.mat.matrix_scale`"""

	def test_correct_size_smaller(self, bigmatrix2D):
		"""tests that smaller desired dimensions return a matrix of the correct size"""
		assert mat.matrix_scale(bigmatrix2D, x=3, y=3).shape == (3,3)

	def test_correct_size_biggerer(self, bigmatrix2D):
		"""tests that bigger desired dimensions return a matrix of the correct size"""
		assert mat.matrix_scale(bigmatrix2D, x=7, y=7).shape == (7,7)

#############################
## Add to Matricies
#############################
class Test_concat_avg():
	"""tests :func:`fns.mat.concat_avg`"""

	def test_invalid_axis(self, bigmatrix2D):
		"""tests that an invalid axis value raises an assertion error"""
		with pytest.raises(AssertionError):
			mat.concat_avg(bigmatrix2D, axis=2)

	def test_size_axis0_spacerTrue(self, bigmatrix2D):
		"""tests that the correct size is returned for axis=0 and spacer=True"""
		assert mat.concat_avg(bigmatrix2D, axis=0, spacer=True).shape == (7, 5)

	def test_size_axis0_spacerFalse(self, bigmatrix2D):
		"""tests that the correct size is returned for axis=0 and spacer=False"""
		assert mat.concat_avg(bigmatrix2D, axis=0, spacer=False).shape == (6, 5)

	def test_size_axis1_spacerTrue(self, bigmatrix2D):
		"""tests that the correct size is returned for axis=1 and spacer=True"""
		assert mat.concat_avg(bigmatrix2D, axis=1, spacer=True).shape == (5, 7)

	def test_size_axis1_spacerFalse(self, bigmatrix2D):
		"""tests that the correct size is returned for axis=1 and spacer=False"""
		assert mat.concat_avg(bigmatrix2D, axis=1, spacer=False).shape == (5, 6)

#############################
## Matrix Correlation
#############################
class Test_scalar_2Dcorr():
	"""tests :func:`fns.mat.scalar_2Dcorr`"""

	def test_identical_corr(self):
		"""tests that the correlation between two identical matricies is one"""
		random = np.random.rand(5,5)
		assert mat.scalar_2Dcorr(random, random) == 1.0

#############################
## Extracting Statistically Significant Values
#############################
class Test_get_statsig():
	"""tests :func:`fns.mat.get_statsig`"""

	def test_high_pvals(self, bigmatrix2D):
		"""tests that values are removed when p-values are higher than the threshold"""
		assert np.all(mat.get_statsig(bigmatrix2D, np.ones((5,5)), threshold=0.5) == np.zeros((5,5)))

	def test_low_pvals(self, bigmatrix2D):
		"""tests that values are kept when p-values are lower than the threshold"""
		assert np.all(mat.get_statsig(bigmatrix2D, np.zeros((5,5)), threshold=0.5) == bigmatrix2D)


