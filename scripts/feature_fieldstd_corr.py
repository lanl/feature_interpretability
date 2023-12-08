# FEATURE - FIELD STD CORRELATION SCRIPT
"""
Generates a matrix of the scalar 2D cross-correlation-coefficient between selected features and the standard deviations of selected fields, averaged across multiple samples
 - Can plot all features (``-T All``) or some selected features (``-T # #``)
 - Can plot all fields (``-F All``) or some selected features (``-F str str``)

Fixed key (``-XK``) specifies what subset of data to consider
 - 'None' can be passed to consider any input with no restrictions
 - For coupon data, fixed keys must be in the form 'tpl###' or 'idx#####'
 - For nested cylinder data, fixed keys must be in the form 'sclPTW_###' or 'idx#####'

Exports correlation coeffients as a pandas-readable .csv

| Samples can be preselected and listed in a .txt file (``-FL filepath``) OR
| Number of samples can be specified and a random selection satisfying the fixed key requirement will be made (``-FL MAKE -NS #``)

Input Line for TF Coupon Models:
``python feature_fieldstd_corr.py -P tensorflow -E coupon -M ../examples/tf_coupon/trained_pRad2TePla_model.h5 -IF pRad -ID ../examples/tf_coupon/data/ -DF ../examples/tf_coupon/coupon_design_file.csv -L activation_15 -T All -NM ft01 -F All -S ../examples/tf_coupon/figures/``

Input Line for PYT Nested Cylinder Models:
``COMING SOON``
"""

#############################
## Packages
#############################
import os
import sys
import glob
import argparse
import warnings
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
warnings.formatwarning = fns.misc.warning_on_one_line

#############################
## Set Up Parser
#############################
def feature_fieldstd_corr_parser():
    descript_str = 'Generates a matrix of the scalar 2D cross-correlation-coefficient between features and the standard deviations of hydrodymanic fields'
    parser = argparse.ArgumentParser(prog='Feature Field Standard Deviation Correlation',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')
    ## Package & Experiment
    setup.args.package(parser) # -P
    setup.args.experiment(parser) # -E

    ## Model Imports
    setup.args.model(parser) # -M
    setup.args.input_field(parser) # -IF
    setup.args.input_dir(parser) # -ID
    setup.args.design_file(parser) # -DF
    setup.args.file_list(parser) # -FL

    ## Print 
    setup.args.print_layers(parser) # -PL
    setup.args.print_features(parser) # -PT
    setup.args.print_fields(parser) # -PF
    setup.args.print_keys(parser) # -PK
    setup.args.print_samples(parser) # -PS

    ## Layer Properties
    setup.args.layer(parser) # -L
    setup.args.features(parser, dft=['All']) # -T
    setup.args.mat_norm(parser) # -NM

    ## Data Properties
    setup.args.fields(parser) # -F
    setup.args.fixed_key(parser) # -XK
    setup.args.num_samples(parser) # -NS

    ## Save Properties
    setup.args.save_fig(parser) # -S

    return parser

parser = feature_fieldstd_corr_parser()

#############################
## Executable
#############################
if __name__ == '__main__':

    args = parser.parse_args()

    ## Package & Experiment
    PACKAGE = args.PACKAGE
    EXP = args.EXPERIMENT

    ## Model Imports
    model_path = args.MODEL
    in_field = args.INPUT_FIELD
    input_dir = args.INPUT_DIR
    file_list_path = args.FILE_LIST
    design_path = args.DESIGN_FILE
    file_list_path = args.FILE_LIST

    ## Print 
    PRINT_LAYERS = args.PRINT_LAYERS
    PRINT_FEATURES = args.PRINT_FEATURES
    PRINT_FIELDS = args.PRINT_FIELDS
    PRINT_KEYS = args.PRINT_KEYS
    PRINT_SAMPLES = args.PRINT_SAMPLES

    ## Layer Properties
    lay = args.LAYER
    features = args.FEATURES
    norm = args.MAT_NORM

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

    ## Create the File list for samples to become features
    if file_list_path != 'MAKE':
        ## Import given file list
        num_samples_ft, sample_list_ft = fns.misc.load_filelist(file_list_path)

        if EXP == 'coupon':
            fixed_key = cp.process.findkey(sample_list_ft)
            if fixed_key != 'None':
                search_dir = input_dir+'r*'+fixed_key+'*.npz'

        if EXP == 'nestedcylinder':
            fixed_key = nc.process.findkey(sample_list_ft)
            if fixed_key != 'None':
                search_dir = input_dir+'ncyl*'+fixed_key+'*.npz'

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

        ## Check Fixed Key
        setup.data_checks.check_key(fixed_key, search_dir)

        ## Print Samples
        if PRINT_SAMPLES:
            setup.data_prints.print_samples(search_dir)
            sys.exit()

        ## Check Samples
        setup.data_checks.check_samples(num_samples, search_dir)
        
        ## Make File List
        if fixed_key=='None': fixed_key=''
        file_path_ft = fig_path+'ftsamples_featurefieldstdcorr_'+fixed_key
        num_samples_ft, sample_list_ft = fns.save.makefilelist(search_dir, num_samples=num_samples, save_path=file_path_ft, save=True)
        # END if file_list_path=='MAKE'

    ## Create the File List for samples to become fields
    if fixed_key=='None': fixed_key=''
    threshold = str(30) #the minimum number of samples to create meaningful standard deviations

    if num_samples_ft >= int(threshold): 
        ## If the feature list has more samples than the threshold, use the same list to compute field standard deviations
        print('Since more than '+threshold+' samples were included in the feature samples list, the field samples list will be identical.')
        num_samples_fld, sample_list_fld = num_samples_ft, sample_list_ft
    else:
        ## Check if the directory has enough samples to meet the threshold
        try:
            setup.data_checks.check_samples(threshold, search_dir)
        except:
            print('\n')
            warnings.warn('Less than '+threshold+' samples are available to compute the standard deviations of the fields. The standard deviations of the fields may not be representative.\n')
            file_path_fld = fig_path+'fldsamples_featurefieldstdcorr_'+fixed_key
            num_samples_fld, sample_list_fld = fns.save.makefilelist(search_dir, num_samples='All', save_path=file_path_fld, save=True)
        else:
            file_path_fld = fig_path+'fldsamples_featurefieldstdcorr_'+fixed_key
            num_samples_fld, sample_list_fld = fns.save.makefilelist(search_dir, num_samples=threshold, save_path=file_path_fld, save=True)


    print('File lists loaded and processed sucessfully.')

    #############################
    ## Data Processing
    #############################
    ## Getting a Test Data File
    test_path = os.path.join(input_dir, sample_list_fld[0])
    test_npz =  np.load(test_path)

    ## Data Prints
    if PRINT_FIELDS: 
        setup.data_prints.print_fields(test_npz)
        sys.exit()

    ## Data Checks
    setup.data_checks.check_fields(test_npz, fields)
    setup.data_checks.check_fields(test_npz, [in_field])

    if EXP == 'coupon':
        ## Parse Fields
        fields, n_fields = cp.process.remove_fields(test_npz, fields)
        test_input = cp.process.get_field(test_npz, fields[0])
        in_y, in_x = test_input.shape

        ## Plotting Settings
        fld_labels =  np.concatenate((fields, ['', 'Abs. Row\nCorr.']), dtype='<U16')
        fld_change = ['rho', 'eqps_rate', 'eff_stress', 'bulk_mod', 'sound_speed']
        fld_latex = [r'$\rho$', '$eqps_{rate}$', 'eff. stress', 'bulk mod.', 'sound speed']
        for i, fld in enumerate(fld_change):
            try:
                fld_labels[fld_labels==fld] = fld_latex[i]
            except:
                pass

    elif EXP == 'nestedcylinder':
        ## Parse Fields
        fields, n_fields = nc.process.remove_fields(test_npz, fields)
        test_input = nc.process.get_field(test_npz, fields[0])
        in_y, in_x = test_input.shape

        ## Plotting Settings
        fld_labels =  np.concatenate((fields, ['', 'Abs. Row\nCorr.']), dtype='<U16')
        fld_change = ['rho', 'hr_MOICyl', 'eqps_rate', 'eff_stress', 'bulk_mod', 'sound_speed']
        fld_latex = [r'$\rho$',r'$\rho_{MOI-CYL}$', '$eqps_{rate}$', 'eff. stress', 'bulk mod.', 'sound speed']
        for i, fld in enumerate(fld_change):
            try:
                fld_labels[fld_labels==fld] = fld_latex[i]
            except:
                pass

    print('Data paths loaded and processed sucessfully.')

    #############################
    ## Extracting Fields
    #############################
    field_tensor = np.empty([num_samples_fld, n_fields, in_y, in_x])

    for s, sample in enumerate(sample_list_fld):
        path = os.path.join(input_dir, sample)
        npz = np.load(path)

        for f, fld in enumerate(fields):
            field_tensor[s, f, :, :] = eval(prefix+'process.get_field(npz, fld)')

        if num_samples_fld > 20 and s%10 == 0:
            print(str(s)+' field samples computed.\n')

    print('Fields extracted sucessfully.')

    #############################
    ## Computing Field Standard Deviation
    #############################

    fld_stds = np.empty([in_y, in_x, n_fields])

    for n in range(n_fields):
        fld_stds[:, :, n] = np.nanstd(field_tensor[:, n, :, :], axis=0)

    ## Saving the Field Standard Deivation
    file_path = fig_path+'fldstds_featurefieldstdcorr_'
    fns.save.fields2npz(fld_stds, fields, save_path=file_path+fixed_key, suffix='_std')

    print('Field standard deivaitons computed and saved sucessfully.')

    #############################
    ## Model Processing
    #############################

    if PACKAGE == 'tensorflow':
        ## Package Imports
        import tensorflow as tf
        from tensorflow import keras
        import fns.tfcustom as tfc

        ## Load Model
        model = keras.models.load_model(model_path)
        print('\n') #Seperate other output from Tensorflow warnings

        ## Prints
        if PRINT_LAYERS: tfc.prints.print_layers(model)
        if PRINT_FEATURES: tfc.prints.print_features(model, lay)
        if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

        ## Checks
        tfc.checks.check_layer(model, lay)
        tfc.checks.check_features(model, lay, features)

        ## Parse Features Input
        n_features, features = tfc.fts.parse_features(model, lay, features)
        _, ft_y, ft_x, _ = model.get_layer(name=lay).output_shape

        ## Make Extractor
        extractor = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(name=lay).output)


    elif PACKAGE == 'pytorch':
        ## Package Imports
        import torch
        import torchvision.models.feature_extraction as ftex
        import fns.pytorchcustom as pytc
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## Load Model
        model_class = 'NCylANN_V1'
        model = pytc.fts.load_model(model_path, model_class, device)

        ## Prints
        if PRINT_LAYERS: pytc.prints.print_layers(model)
        if PRINT_FEATURES: pytc.prints.print_features(model)
        if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

        ## Checks
        pytc.checks.check_layer(model, lay)
        pytc.checks.check_features(model, features)

        ## Parse Features Input
        n_features, features = pytc.fts.parse_features(model, features)
        ft_y, ft_x = in_y, in_x

        ## Make Extractor 
        extractor = ftex.create_feature_extractor(model, return_nodes={lay: lay})
    
    print('Model loaded sucessfully.')

    #############################
    ## Extracting Features
    #############################
    ## Making an Empty Array
    corrs = np.empty((n_features, n_fields, num_samples_ft))

    ## Go Through the Sample List
    for s, sample in enumerate(sample_list_ft):
        ## Load Inputs
        sample_path = os.path.join(input_dir, sample)
        npz = np.load(sample_path)

        ## Data Processing Model Input
        if EXP == 'coupon':
            img_in = cp.process.get_field(npz, in_field) 
            key = cp.process.npz2key(sample_path)
            dcj = cp.process.csv2dcj(design_path, key)
            model_in = [img_in.reshape((1,in_y,in_x,1)),dcj]

        elif EXP == 'nestedcylinder':
            img_in = nc.process.get_field(npz, in_field)
            model_in = img_in

        ## Extracting Features
        if PACKAGE == 'tensorflow':
            extracted_fts = extractor(model_in)[0, :, :, :]
            extracted_fts = np.array(extracted_fts)

        elif PACKAGE == 'pytorch':
            model_in = torch.tensor(model_in.reshape((1, 1, in_y, in_x))).to(torch.float32).to(device)
            extracted_fts_pyt = extractor(model_in)[lay].detach().numpy()[0, :, :, :]
            ## Fix the pytorch dimensions convention
            extracted_fts = np.zeros((ft_y, ft_x, n_features))
            for i in range(n_features):
                extracted_fts[:, :, i] = extracted_fts_pyt[i, :, :]

        out_y, out_x, _ = np.shape(extracted_fts)

        ## Compute All the Correlations
        for n in range(n_fields):
            cut_field = fns.mat.matrix_cutting(fld_stds[:, :, n], out_x, out_y)

            for f, feature in enumerate(features):
                ft = extracted_fts[:,:,feature]
                corrs[f, n, s] = fns.mat.scalar_2Dcorr(cut_field, ft)

        if num_samples_ft > 20 and s%10 == 0:
            print(str(s)+' feature samples computed.\n')

    ## Compute the Average
    avg_corr = np.nanmean(corrs, axis=2)

    ## Adding Averages to Matrix
    avg_corr = fns.mat.concat_avg(avg_corr, axis=0, spacer=True)
    avg_corr = fns.mat.concat_avg(avg_corr, axis=1, spacer=True)
    avg_corr[0,-1] = 0 #zero out the top right corner

    ## Save the correlations
    file_path = fig_path+'featurefieldstdcorr_'+fixed_key
    features1 = features+1
    df_ft_labels = np.concatenate((['Abs. Col. Avg.', 'Spacer'], features1.astype(str)))
    df_fld_labels = np.concatenate((fields, ['Spacer', 'Abs. Row Avg.']))
    avg_corr_df = pd.DataFrame(data=avg_corr, index=df_ft_labels, columns=df_fld_labels)
    avg_corr_df.to_csv(file_path+'.csv')

    print('Correlations computed and saved sucessfully.')

    #############################
    ## Plot Features
    #############################
    ft_labels = np.concatenate((['Abs. Col.\nAvg.', ''], features1.astype(str)))
    xlabel = 'Standard Deviations of\nRadiograph & Hydrodynamic Fields'
    ylabel = 'Features from Selected Layer'
    title = 'Average Correlation of \nFeatures from '+lay+'\n& Std. of Hydrodymanic Fields\nfrom '+str(num_samples_ft)+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(avg_corr, xticklabels=fld_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=file_path, grid=False)

    print('Plots generated and saved.')