# CLEAR TEST_CHACHE DIRECTORY FUNCTION
"""
Deletes all files in the temp_figures directory.

The temp_figures directory is for temporary storage of files only, and is ignored by the git system.

"""

#############################
## Packages
#############################
import os
import glob
from fns.misc import find_path

#############################
## Function
#############################
def clear_test_cache():
    """ Function to find the temp_figures directory and delete all of its contents
    """

    test_cache_path = find_path(__file__, 'temp_figures')

    filelist = glob.glob(test_cache_path+'/*')

    for file in filelist:
        if 'README' not in file:
            os.remove(file)

    print('temp_figures directory has been cleared')


#############################
## Executable
#############################
if __name__ == '__main__':

    clear_test_cache()