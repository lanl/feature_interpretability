# SAVE FUNCTIONS UNIT TESTS
"""
Contains tests for fns.save functions
"""

#############################
## Packages
#############################
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath('../src/'))
import fns.save as save

#############################
## Define Fixtures
#############################
@pytest.fixture
def tensor():
	return np.random.rand(10,7,3)

#############################
## Save to .npz File
#############################
class Test_features2npz():
	"""tests :func:`fns.save.features2npz`"""

	def test_runs(self, tensor):
		"""tests that code runs to completion"""
		try:
			save.features2npz(tensor, save_path='./test')
			os.remove('./test.npz')
		except:
			assert False, "Exception raised for save.features2npz(...)."
		else:
			assert True

class Test_fields2npz():
	"""tests :func:`fns.save.fields2npz`"""

	def test_runs(self, tensor):
		"""tests that code runs to completion"""
		try:
			save.fields2npz(tensor, fields=['field1', 'field2', 'field3'], save_path='./test')
			os.remove('./test.npz')
		except:
			assert False, "Exception raised for save.fields2npz(...)."
		else:
			assert True