# FILED SENSITIVITY SCRIPT
"""
Plots the average and standard deviation of fields from a set of samples
 - Averages each pixel across multiple inputs
 - Plots each field on its own plot

Fixed key (``-XK``) specifies what subset of data to consider
 - 'None' can be passed to consider any input with no restrictions
 - For coupon data, fixed keys must be in the form 'tpl###' or 'idx#####'
 - For nested cylinder data, fixed keys must be in the form 'id####' or 'idx#####'

Saves all averages and standard deviations to a .npz file.

| Samples can be preselected and listed in a .txt file (``-FL filepath``) OR
| Number of samples can be specified and a random selection satisfying the fixed key requirement will be made (``-FL MAKE -NS #``)

Input Line for Coupon Data:
``python field_sensitivity.py -E coupon -ID ../examples/tf_coupon/data/ -F All -S ../examples/tf_coupon/figures/``

Input Line for Nested Cylinder Data:
``python field_sensitivity.py -E nestedcylinder -ID ../examples/pyt_nestedcyl/data/ -F All -S ../examples/pyt_nestedcyl/figures/``
"""

#############################
## Packages
#############################
import os
import sys
import glob
import argparse
import random
import numpy as np
import pandas as pd

## Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('pdf') ## To change which backend is used

## Custom Functions
sys.path.insert(0, os.path.abspath('../src/'))
import fns
import fns.setup as setup
import fns.plot

#############################
## Set Up Parser
#############################
def field_sensitivity_parser():
    descript_str = 'Plots the average and standard deviation of hydrodynamic fields extracted from a set of related inputs'
    parser = argparse.ArgumentParser(prog='Field Sensitivity',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')
    ## Experiment
    setup.args.experiment(parser) # -E

    ## Model Imports
    setup.args.input_dir(parser) # -ID
    setup.args.file_list(parser) # -FL

    ## Print Toggles
    setup.args.print_fields(parser) # -PF
    setup.args.print_keys(parser) # -PK
    setup.args.print_samples(parser) # -PS

    ## Data Properties
    setup.args.fields(parser) # -F
    setup.args.fixed_key(parser) # -XK
    setup.args.num_samples(parser) # -NS

    ## Color Properties
    setup.args.alpha1(parser, dft=0.25) # -A1
    setup.args.alpha2(parser, dft=1.00) # -A2
    setup.args.color1(parser, dft='yellow') # -C1
    setup.args.color2(parser, dft='red') # -C2

    ## Save Properties
    setup.args.save_fig(parser) # -S

    return parser

parser = field_sensitivity_parser()

#############################
## Executable
#############################
if __name__ == '__main__':

    args = parser.parse_args()

    ## Experiment
    EXP = args.EXPERIMENT

    ## Model Imports
    input_dir = args.INPUT_DIR
    file_list_path = args.FILE_LIST

    ## Print Toggles
    PRINT_FIELDS = args.PRINT_FIELDS
    PRINT_KEYS = args.PRINT_KEYS
    PRINT_SAMPLES = args.PRINT_SAMPLES

    ## Data Properties
    fields = args.FIELDS
    fixed_key = args.FIXED_KEY
    num_samples = args.NUM_SAMPLES

    ## Color Properties
    alpha1 = args.ALPHA1
    alpha2 = args.ALPHA2
    color1 = args.COLOR1
    color2 = args.COLOR2

    ## Save Properties
    fig_path = args.SAVE_FIG

    #############################
    ## File List Path Processing
    #############################
    ## Import Custom Packages
    if EXP == 'coupon':
        import fns.coupondata as cp
        prefix = 'cp.'
        import fns.coupondata.prints as cpprints
        search_dir = os.path.join(input_dir, 'r*tpl*idx*.npz')
    elif EXP == 'nestedcylinder':
        import fns.nestedcylinderdata as nc
        prefix = 'nc.'
        import fns.nestedcylinderdata.prints as ncprints
        search_dir = os.path.join(input_dir, 'nc231213*pvi*.npz')

    if file_list_path != 'MAKE':
        ## Import given file list
        n_samples, sample_list = fns.misc.load_filelist(file_list_path)
        fixed_key = eval(prefix + 'process.findkey(sample_list)')

        ## Prints
        if PRINT_KEYS: print('Key present in '+file_list_path+' is '+fixed_key)
        if PRINT_SAMPLES: print(str(num_samples)+' samples found in '+file_list_path+'.\n')
        if PRINT_KEYS or PRINT_SAMPLES: sys.exit()
        

    elif file_list_path == 'MAKE':

        if EXP == 'coupon':
            ## Prints Keys
            if PRINT_KEYS: 
                cpprints.print_keys(search_dir)

                if PRINT_SAMPLES: 
                    print('Print Samples -PS command is not compatible with Print Keys -PK.')
                    print('Specify a fixed key -XK to use -PS.')

                sys.exit()

            ## Set Fixed Key
            if fixed_key != 'None':
                search_dir = os.path.join(input_dir, 'r*'+fixed_key+'*.npz')

        elif EXP == 'nestedcylinder':
            ## Prints Keys
            if PRINT_KEYS: 
                ncprints.print_keys(search_dir)

                if PRINT_SAMPLES: 
                    print('Print Samples -PS command is not compatible with Print Keys -PK.')
                    print('Specify a fixed key -XK to use -PS.')

                sys.exit()

            ## Set Fixed Key
            if fixed_key != 'None':
                search_dir = os.path.join(input_dir, 'nc231213*'+fixed_key+'*.npz')
            # END if EXP=='nestedcylinder'

        ## Check Fixed Key
        setup.data_checks.check_key(fixed_key, search_dir)

        ## Print Samples
        if PRINT_SAMPLES:
            setup.data_prints.print_samples(search_dir)
            sys.exit()

        ## Check Samples
        setup.data_checks.check_samples(num_samples, search_dir)
        
        ## Make File List
        file_path = os.path.join(fig_path, 'field_sensitivity')
        n_samples, sample_list = fns.save.makefilelist(search_dir, num_samples=num_samples, save_path=file_path, save=True)
        # END if file_list_path=='MAKE'

    print('File list loaded and processed sucessfully.')
    
    #############################
    ## Data Processing
    #############################
    ## Getting a Test Data File
    test_path = os.path.join(input_dir, sample_list[0])
    test_npz =  np.load(test_path)

    ## Data Prints
    if PRINT_FIELDS: 
        setup.data_prints.print_fields(test_npz)
        sys.exit()

    ## Data Checks
    setup.data_checks.check_fields(test_npz, fields)

    if EXP == 'coupon':
        ## Package Imports
        import fns.coupondata as cp
        prefix = 'cp.'

        ## Plotting Settings
        Rlabels, Rticks, Zlabels, Zticks = cp.process.get_ticks(test_npz)
        smallfigsize = (10,5)
        RMinorTicks = False
        ZMinorTicks = False
        fld_labels =  np.array(fields, dtype='<U13')
        fld_change = ['rho', 'eqps_rate', 'eff_stress', 'bulk_mod', 'sound_speed']
        fld_latex = [r'$\rho$', '$eqps_{rate}$', 'eff. stress', 'bulk mod.', 'sound speed']
        for i, fld in enumerate(fld_change):
            try:
                fld_labels[fld_labels==fld] = fld_latex[i]
            except:
                pass

        ## Parse Fields
        fields, n_fields = cp.process.remove_fields(test_npz, fields)
        test_input = cp.process.get_field(test_npz, fields[0])
        in_y, in_x = test_input.shape

    elif EXP == 'nestedcylinder':
        ## Package Imports
        import fns.nestedcylinderdata as nc
        prefix = 'nc.'

        ## Plotting Settings
        Rlabels, Rticks, Zlabels, Zticks = nc.process.get_ticks(test_npz)
        smallfigsize = (5,10)
        RMinorTicks = True
        ZMinorTicks = False
        fld_labels =  np.array(fields, dtype='<U16')
        fld_change = ['rho', 'hr_MOICyl', 'eqps_rate', 'eff_stress', 'bulk_mod', 'sound_speed']
        fld_latex = [r'$\rho$',r'$\rho_{MOI-CYL}$', '$eqps_{rate}$', 'eff. stress', 'bulk mod.', 'sound speed']
        for i, fld in enumerate(fld_change):
            try:
                fld_labels[fld_labels==fld] = fld_latex[i]
            except:
                pass

        ## Parse Fields
        fields, n_fields = nc.process.remove_fields(test_npz, fields)
        test_input = nc.process.get_field(test_npz, fields[0])
        in_y, in_x = test_input.shape

    print('Data paths loaded and processed sucessfully.')

    #############################
    ## Extracting Fields
    #############################
    field_tensor = np.empty([n_samples, n_fields, in_y, in_x])

    for s, sample in enumerate(sample_list):
        path = os.path.join(input_dir, sample)
        npz = np.load(path)

        for f, fld in enumerate(fields):
            field_tensor[s, f, :, :] = eval(prefix+'process.get_field(npz, fld)')

        if n_samples > 20 and s%10 == 0:
            print(str(s)+' samples computed.\n')

    print('Fields extracted sucessfully.')

    #############################
    ## Computing Mean & Standard Deviation
    #############################

    fld_avgs = np.empty([in_y, in_x, n_fields])
    fld_stds = np.empty([in_y, in_x, n_fields])

    for n in range(n_fields):
        fld_avgs[:, :, n] = np.nanmean(field_tensor[:, n, :, :], axis=0)
        fld_stds[:, :, n] = np.nanstd(field_tensor[:, n, :, :], axis=0)

    ## Saving the Avgerage and Standard Deivation
    if fixed_key=='None': fixed_key=''
    fns.save.fields2npz(fld_avgs, fields, save_path=fig_path+'fields_avg_'+fixed_key, suffix='_avg')
    fns.save.fields2npz(fld_stds, fields, save_path=fig_path+'fields_std_'+fixed_key, suffix='_std')

    print('Feature averages & standard deivaitons computed and saved sucessfully.')

    #############################
    ## Plot Features
    #############################
    zeros = np.zeros((in_y, in_x))
    fns.plot.custom_colormap(color1, color2, alpha1, alpha2)
    plt.rcParams.update({'font.size': 14})
    if fixed_key=='': fixed_key=str(n_samples)+'_Samples'

    for f, fld in enumerate(fields):
        ## Plotting the Average
        lims = [np.min(fld_avgs[:, :, f]), np.max(fld_avgs[:, :, f])]
        fig = plt.figure(figsize=smallfigsize, layout="constrained")
        im = fns.plot.feature_plot(zeros, fld_avgs[:, :, f], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, lims, resize='None')
        plt.suptitle('Average '+fld+' for '+fixed_key)
        fig.colorbar(im, label='Activation Intensity')
        plt.clim(lims)
        
        fig.savefig(os.path.join(fig_path, fld+'_avg_'+fixed_key+'.png'))
        plt.close()

        ## Plotting the Standard Deviation
        lims = [np.min(fld_stds[:, :, f]), np.max(fld_stds[:, :, f])]
        fig = plt.figure(figsize=smallfigsize, layout="constrained")
        im = fns.plot.feature_plot(zeros, fld_stds[:, :, f], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, lims, resize='None')
        plt.suptitle('Standard Deviation of\n'+fld+' for '+fixed_key)
        fig.colorbar(im, label='Activation Intensity')
        plt.clim(lims)
        
        fig.savefig(os.path.join(fig_path, fld+'_std_'+fixed_key+'.png'))
        plt.close()

    print('Plots generated and saved.')