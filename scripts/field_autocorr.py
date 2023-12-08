# FILED AUTOCORRELATION SCRIPT
"""
Generates a heatmap plot of the 2D auto-cross-correlation-coefficients across hydrodynamic fields, averaged across multiple samples
 - Can plot all fields (``-F All``) or some selected fields (``-F str str``)

Fixed key (``-XK``) specifies what subset of data to consider
 - 'None' can be passed to consider any input with no restrictions
 - For coupon data, fixed keys must be in the form 'tpl###' or 'idx#####'
 - For nested cylinder data, fixed keys must be in the form 'sclPTW_###' or 'idx#####'

Exports correlation coeffients as a pandas-readable .csv

| Samples can be preselected and listed in a .txt file (``-FL filepath``) OR
| Number of samples can be specified and a random selection satisfying the fixed key requirement will be made (``-FL MAKE -NS #``)

Input Line for Coupon Data:
``python field_autocorr.py -E coupon -ID ../examples/tf_coupon/data/ -F All -S ../examples/tf_coupon/figures/``

Input Line for Nested Cylinder Data:
``COMING SOON``
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
def field_autocorr_parser():
    descript_str = 'Generates a matrix of the scalar 2D auto-cross-correlation-coefficient between hydrodymanic fields'
    parser = argparse.ArgumentParser(prog='Field Auto-Correlation',
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

    ## Save Properties
    setup.args.save_fig(parser) # -S

    return parser

parser = field_autocorr_parser()

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
        search_dir = input_dir+'r*tpl*idx*.npz'
    elif EXP == 'nestedcylinder':
        raise NotImplementedError('Nested cylinder examples not included in open source.')
        import fns.nestedcylinderdata as nc
        prefix = 'nc.'
        import fns.nestedcylinderdata.prints as ncprints
        search_dir = input_dir+'ncyl_sclPTW*idx*.npz'

    if file_list_path != 'MAKE':
        ## Import given file list
        n_samples, sample_list = fns.misc.load_filelist(file_list_path)
        fixed_key = cp.process.findkey(sample_list)

        ## Prints
        if PRINT_KEYS: print('Key present in '+file_list_path+' is '+fixed_key)
        if PRINT_SAMPLES: print(str(num_samples)+' samples found in '+file_list_path+'.\n')
        if PRINT_KEYS or PRINT_SAMPLES: sys.exit()

        file_path = fig_path+'field_autocorr'

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
                search_dir = input_dir+'r*'+fixed_key+'*.npz'

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
                search_dir = input_dir+'ncyl*'+fixed_key+'*.npz'
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
        file_path = fig_path+'field_autocorr'
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

        fields, n_fields = cp.process.remove_fields(test_npz, fields)

        ## Plotting Settings
        fld_labels_vert = np.concatenate((['Abs. Avg.\nCorr.', ''], fields))
        fld_labels_horz =  np.array(fields, dtype='<U13')
        fld_change = ['rho', 'eqps_rate', 'eff_stress', 'bulk_mod', 'sound_speed']
        fld_latex = [r'$\rho$', '$eqps_{rate}$', 'eff. stress', 'bulk mod.', 'sound speed']
        for i, fld in enumerate(fld_change):
            try:
                fld_labels_horz[fld_labels_horz==fld] = fld_latex[i]
                fld_labels_vert[fld_labels_vert==fld] = fld_latex[i]
            except:
                pass

    elif EXP == 'nestedcylinder':
        ## Package Imports
        import fns.nestedcylinderdata as nc
        prefix = 'nc.'

        fields, n_fields = nc.process.remove_fields(test_npz, fields)

        ## Plotting Settings
        fld_labels_vert = np.concatenate((['Abs. Avg.\nCorr.', ''], fields))
        fld_labels_vert = fld_labels_vert.astype('<U16')
        fld_labels_horz =  np.array(fields, dtype='<U16')
        fld_change = ['rho', 'hr_MOICyl', 'eqps_rate', 'eff_stress', 'bulk_mod', 'sound_speed']
        fld_latex = [r'$\rho$',r'$\rho_{MOI-CYL}$', '$eqps_{rate}$', 'eff. stress', 'bulk mod.', 'sound speed']
        for i, fld in enumerate(fld_change):
            try:
                fld_labels_horz[fld_labels_horz==fld] = fld_latex[i]
                fld_labels_vert[fld_labels_vert==fld] = fld_latex[i]
            except:
                pass

    print('Data paths loaded and processed sucessfully.')
    
    #############################
    ## Computing Correlations
    #############################
    ## Making an Empty Array
    corrs = np.empty((n_fields, n_fields, n_samples))

    ## Go Through the Sample List
    for s, sample in enumerate(sample_list):
        ## Load Input
        sample_path = os.path.join(input_dir, sample)
        npz = np.load(sample_path)

        ## Compute All the Correlations
        for i, field_i in enumerate(fields):
            field_pic_i = eval(prefix+'process.get_field(npz, field_i)')
            
            for j, field_j in enumerate(fields):
                field_pic_j = eval(prefix+'process.get_field(npz, field_j)')

                corrs[j, i, s] = fns.mat.scalar_2Dcorr(field_pic_i, field_pic_j)

        if n_samples > 20 and s%10 == 0:
            print(str(s)+' samples computed.\n')

    ## Compute the Average
    avg_corr = np.nanmean(corrs, axis=2)

    ## Adding Column Average
    avg_corr = fns.mat.concat_avg(avg_corr, axis=0, spacer=True)

    ## Save the correlations
    if fixed_key=='None': fixed_key=''
    field_labels = np.concatenate((['Abs. Avg. Corr.', 'Spacer'], fields))
    avg_corr_df = pd.DataFrame(data=avg_corr, index=field_labels, columns=fields)
    avg_corr_df.to_csv(file_path+fixed_key+'.csv')

    print('Correlations computed and saved sucessfully.')

    #############################
    ## Plot Heatmap
    #############################
    if fixed_key!='': fixed_key = ' '+fixed_key
    title = 'Average Auto-Correlation of Hydrodynamic Fields\nfrom '+str(n_samples)+fixed_key+' Sample(s)'
    axislabel = 'Radiograph & Hydrodynamic Fields'
    fns.plot.heatmap_plot(avg_corr, xticklabels=fld_labels_horz, yticklabels=fld_labels_vert, xlabel=axislabel, ylabel=axislabel, title=title, save_path=file_path, grid=False)

    print('Plots generated and saved.')