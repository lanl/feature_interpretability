# MISC OPERATIONS FUNCTIONS UNIT TESTS
"""
Contains tests for fns.misc functions
"""

#############################
## Packages
#############################
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath('../src/'))
import fns.misc as misc


#############################
## Define Fixtures
#############################
@pytest.fixture
def testlist():
	return ['test']*5


#############################
## List Operations
#############################
class Test_search_list():
	"""tests :func:`fns.misc.search_list`"""

	def test_return_empty(self, testlist):
		"""tests that when query is not in any elements of the list, the function returns an empty set"""
		assert misc.search_list(testlist, 'a') == []

	def test_return_all(self, testlist):
		"""tests that when query is in all elements of the list, the function returns the original list"""
		assert misc.search_list(testlist, 't') == testlist
