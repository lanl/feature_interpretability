# MISC FUNCTIONS
"""
Contains functions that don't fit into any existing category.
"""

#############################
## Packages
#############################
import os

#############################
## Warning Formatting
#############################
def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
	""" Function to reformat warnings.warn() to display all on one line

		Code from: http://pymotw.com/2/warnings/

		To use: ``warnings.formatwarning = fns.misc.warning_on_one_line``

	"""
	return '%s:%s: %s: %s\n' % (filename, lineno, category.__name__, message)

#############################
## List Operations
#############################
def search_list(list_l: list, query: str):
	""" Function to return the list of elements that contain the query from a masterlist

        Args:
            list_l (list): masterlist of strings to be searched
            query (str): key to search elements of masterlist for

        Returns:
			elements (list): list of elemenmts from the masterlist that contain the query
    """
	elements = [x for x in list_l if query in x]
	return elements

#############################
## Path Operations
#############################
def find_path(me: str, goal: str):
	""" Function to return the path to the goal directory, looking starting at the me directory and moving outwards
		
		Args:
			me (str): the directory to start searching from
			goal (str): the directory to find
		
		Returns:
			path (str): absoulte path to the goal directory; will return "none" if path could not be found with less than 5 steps out of starting directory
	"""
	path = os.path.dirname(os.path.abspath(me))
	tries = 0
	limit = 5

	while goal not in os.listdir(path) and tries<limit:
	    path = os.path.dirname(path)
	    tries+= 1

	if tries == limit:
	    path = 'none'
	    print('Path Search was unsuccesful; '+goal+' was not found.')
	else:
	    path += '/'+goal
	return path

def load_filelist(file_list_path: str):
	""" Function to parse a file list path into an array of sample paths

        Args:
            file_list_path (str): path to .txt file containing list of samples

        Returns:
			num_samples (int): number of samples in the sample list
			sample_list (np.array): array of file paths to .npz samples
    """
	with open(file_list_path) as f:
		sample_list = [line.rstrip() for line in f]
	n_samples = len(sample_list)

	return n_samples, sample_list
