# TEST SCRIPTS SCRIPT
"""
Runs all scripts to test for errors. Does not save script output.

Input Line for Tensorflow Models on Coupon Data:
``python test_scripts.py -P tensorflow -E coupon``

Input Line for PYT Nested Cylinder Models:
``python test_scripts.py -P pytorch -E nestedcylinder``
"""

#############################
## Packages
#############################
import os
import sys
import time
import argparse
import subprocess

## Custom Functions
sys.path.insert(0, os.path.abspath('../src/'))
from fns.clear_test_cache import clear_test_cache
import fns.setup as setup


#############################
## Set Up Parser
#############################
def test_run_parser():
    descript_str = "Runs all feature interpretability scripts to test for errors."

    parser = argparse.ArgumentParser(prog='Test Run',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')
    ## Package & Experiment
    setup.args.package(parser) # -P
    setup.args.experiment(parser) # -E

    return parser

parser = test_run_parser()

#############################
## Executable
#############################
if __name__ == '__main__':

    start = time.time()

    args = parser.parse_args()

    ## Package & Experiment
    PACKAGE = args.PACKAGE
    EXP = args.EXPERIMENT

    #############################
    ## Define Input Arguments
    #############################
    ## Define universal input arguments
    FL = 'MAKE'
    NM = 'ft01'
    NR = '2'
    XK = 'None'
    NS = '25'
    S = '../temp_figures/'

    ## Define package-dependent input arguments
    if PACKAGE=='tensorflow':
    	M = '../examples/tf_coupon/trained_pRad2TePla_model.h5'
    	IF = 'pRad'
    	L = 'activation_15'

    elif PACKAGE=='pytorch':
    	M = '../examples/pyt_nestedcyl/trained_hrMOICyl2sclPTW_model.pth'
    	IF = 'hr_MOICyl'
    	L = 'interpActivations.14'

    ## Define experiment-dependent input arguments
    if EXP == 'coupon':
    	IN = '../examples/tf_coupon/data/r60um_tpl112_complete_idx00110.npz'
    	ID = '../examples/tf_coupon/data/'
    	DF = '../examples/tf_coupon/coupon_design_file.csv'
    	NF = '../examples/tf_coupon/coupon_normalization.npz'

    elif EXP == 'nestedcylinder':
        raise NotImplementedError('Nested cylinder examples not included in open source.')
    	IN = '../examples/pyt_nestedcyl/data/ncyl_sclPTW_327_pvi_idx00130.npz'
    	ID = '../examples/pyt_nestedcyl/data/'
    	DF = '../examples/pyt_nestedcyl/nestedcyl_design_file.csv'
    	NF = '' #input not required for nested cylinder experiment 

    #############################
    ## Run Scripts
    #############################

    print('\n-------------------------')
    print('Running plot_features.py:')
    subprocess.run(['python', 'plot_features.py', '-P', PACKAGE, '-E', EXP, '-M', M, '-IF', IF, '-IN', IN, '-DF', DF, '-L', L, '-T', 'Grid', '-NM', NM, '-S', S], check=True)

    print('\n-------------------------')
    print('Running feature_over_field.py:')
    subprocess.run(['python', 'feature_over_field.py', '-P', PACKAGE, '-E', EXP, '-M', M, '-IF', IF, '-IN', IN, '-DF', DF, '-L', L, '-T', 'All', '-NM', NM, '-F', 'All', '-S', S], check=True)

    print('\n-------------------------')
    print('Running feature_sensitivity.py:')
    subprocess.run(['python', 'feature_sensitivity.py', '-P', PACKAGE, '-E', EXP, '-M', M, '-IF', IF, '-ID', ID, '-FL', FL, '-DF', DF, '-L', L, '-T', 'Grid', '-NM', NM, '-XK', XK, '-NS', NS, '-S', S], check=True)

    print('\n-------------------------')
    print('Running field_sensitivity.py:')
    subprocess.run(['python', 'field_sensitivity.py', '-E', EXP, '-ID', ID, '-FL', FL, '-F', 'All', '-XK', XK, '-NS', NS, '-S', S], check=True)

    print('\n-------------------------')
    print('Running field_autocorr.py:')
    subprocess.run(['python', 'field_autocorr.py', '-E', EXP, '-ID', ID, '-FL', FL, '-F', 'All', '-XK', XK, '-NS', NS, '-S', S], check=True)

    print('\n-------------------------')
    print('Running feature_field_corr.py:')
    subprocess.run(['python', 'feature_field_corr.py', '-P', PACKAGE, '-E', EXP, '-M', M, '-IF', IF, '-ID', ID, '-FL', FL, '-DF', DF, '-L', L, '-T', 'All', '-NM', NM, '-F', 'All', '-XK', XK, '-NS', NS, '-S', S], check=True)

    print('\n-------------------------')
    print('Running feature_fieldstd_corr.py:')
    subprocess.run(['python', 'feature_fieldstd_corr.py', '-P', PACKAGE, '-E', EXP, '-M', M, '-IF', IF, '-ID', ID, '-FL', FL, '-DF', DF, '-L', L, '-T', 'All', '-NM', NM, '-F', 'All', '-XK', XK, '-NS', NS, '-S', S], check=True)

    print('\n-------------------------')
    print('Running feature_pred_corr.py:')
    subprocess.run(['python', 'feature_pred_corr.py', '-P', PACKAGE, '-E', EXP, '-M', M, '-IF', IF, '-ID', ID, '-FL', FL, '-DF', DF, '-L', L, '-T', 'All', '-NR', NR, '-XK', XK, '-NS', NS, '-S', S], check=True)

    print('\n-------------------------')
    print('Running feature_derivatives.py:')
    if PACKAGE=='tensorflow' and EXP=='coupon':
        subprocess.run(['python', 'ftderivatives_tf_coupon.py', '-M', M, '-IF', IF, '-ID', ID, '-FL', FL, '-DF', DF, '-NF', NF, '-L', L, '-T', '1', '-DS', '0.001', '-XK', XK, '-NS', NS, '-S', S], check=True)

    elif PACKAGE=='pytorch' and EXP=='nestedcylinder':
        subprocess.run(['python', 'ftderivatives_pyt_nestedcyl.py', '-M', M, '-IF', IF, '-ID', ID, '-FL', FL, '-DF', DF, '-L', L, '-T', '1', '-DS', '0.001', '-XK', XK, '-NS', NS, '-S', S], check=True)

    #############################
    ## Calculate Timing
    #############################
    clear_test_cache()
    
    end = time.time()
    duration = (end - start) / 60 # in minutes
    print('\n-------------------------')
    print('Program test_run.py took', round(duration, 2), 'minutes to run.')
