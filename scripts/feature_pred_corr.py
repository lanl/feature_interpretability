# FEATURE - PREDICTION CORRELATION SCRIPT
"""
Generates a matrix of the cross-correlation-coefficient between the vector of feature norms across a given number of samples and the vector of model predictions across the same samples
 - Can plot all features (``-T All``) or some selected features (``-T # #``)

Calculates three correlation metrics:
 - 2D cross-correlation
 - Partial correlation taking other features as confounding factors
 - Partial rank correlationtaking other features as confounding factors

For each metric, generates a matrix for the:
 - Correlation coefficients
 - P-values
 - Statistically significnat correlation coeffieincts corresponding to a p-value less than some threshold

Fixed key (``-XK``) specifies what subset of data to consider
 - 'None' can be passed to consider any input with no restrictions
 - For coupon data, fixed keys must be in the form 'tpl###' or 'idx#####'
 - For nested cylinder data, fixed keys must be in the form 'sclPTW_###' or 'idx#####'

Exports correlation coeffients as a pandas-readable .csv

| Samples can be preselected and listed in a .txt file (``-FL filepath``) OR
| Number of samples can be specified and a random selection satisfying the fixed key requirement will be made (``-FL MAKE -NS #``)

Input Line for TF Coupon Models:
``python feature_pred_corr.py -P tensorflow -E coupon -M ../examples/tf_coupon/trained_pRad2TePla_model.h5 -IF pRad -ID ../examples/tf_coupon/data/ -DF ../examples/tf_coupon/coupon_design_file.csv -L activation_15 -T All -NR 2 -S ../examples/tf_coupon/figures/``

Input Line for PYT Nested Cylinder Models:
``python feature_pred_corr.py -P pytorch -E nestedcylinder -M ../examples/pyt_nestedcyl/trained_hrMOICyl2sclPTW_model.pth -IF hr_MOICyl -ID ../examples/pyt_nestedcyl/data/ -L interpActivations.14 -T All -NR 2 -S ../examples/pyt_nestedcyl/figures/``
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
import pingouin as pg

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
def feature_pred_corr_parser():
    descript_str = 'Generates a matrix of the cross-correlation-coefficient between the vector of norms of features across a given number of samples and the vector of model predictionss across the same samples'
    parser = argparse.ArgumentParser(prog='Feature Prediction Correlation',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')

    ## Package & Experiment
    setup.args.package(parser) # -P
    setup.args.experiment(parser) # -E

    ## Model Imports
    setup.args.model(parser) # -M
    setup.args.input_field(parser) # -IF
    setup.args.input_dir(parser) # -ID
    setup.args.file_list(parser) # -FL
    setup.args.design_file(parser) # -DF

    ## Print Toggles
    setup.args.print_layers(parser) # -PL
    setup.args.print_features(parser) # -PT
    setup.args.print_fields(parser) # -PF
    setup.args.print_keys(parser) # -PK
    setup.args.print_samples(parser) # -PS

    ## Layer Properties
    setup.args.layer(parser) # -L
    setup.args.features(parser, dft=['All']) # -T
    setup.args.sclr_norm(parser) #-NR

    ## Data Properties
    setup.args.fixed_key(parser) # -XK
    setup.args.num_samples(parser) # -NS

    ## Save Properties
    setup.args.save_fig(parser) # -S

    return parser

parser = feature_pred_corr_parser()

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

    ## Print Toggles
    PRINT_LAYERS = args.PRINT_LAYERS
    PRINT_FEATURES = args.PRINT_FEATURES
    PRINT_FIELDS = args.PRINT_FIELDS
    PRINT_KEYS = args.PRINT_KEYS
    PRINT_SAMPLES = args.PRINT_SAMPLES

    ## Layer Properties
    lay = args.LAYER
    features = args.FEATURES
    norm = args.SCLR_NORM

    ## Data Properties
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

    ## Create the File list
    if file_list_path != 'MAKE':
        ## Import given file list
        num_samples, sample_list = fns.misc.load_filelist(file_list_path)
        fixed_key = eval(prefix + 'process.findkey(sample_list)')

        ## Prints
        if PRINT_KEYS: print('Key present in '+file_list_path+' is '+fixed_key)
        if PRINT_SAMPLES: print(str(num_samples)+' samples found in '+file_list_path+'.\n')
        if PRINT_KEYS or PRINT_SAMPLES: sys.exit()
        
        if fixed_key=='None': fixed_key=''

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
        file_path = fig_path+'featurepredcorr_'+fixed_key
        num_samples, sample_list = fns.save.makefilelist(search_dir, num_samples=num_samples, save_path=file_path, save=True)
        # END if file_list_path=='MAKE'

    ## Check that there are enough samples for statistical significance
    """ In order to calculate the p-value for the partial correlation coefficient, 
        dof = (number of samples) - (number of covariates) - 2
            = (number of samples) - 11 - 2
        must be greater than zero (see pingouin/correlation.py lines 58-59).
        Therefore, number of samples > 13.
    """
    if num_samples <= 13:
        raise ValueError('Number of samples must be greater than 13 to allow the calculation of the p-value.')

    print('File lists loaded and processed sucessfully.')

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
    setup.data_checks.check_fields(test_npz, [in_field])

    if EXP == 'coupon':
        ## Data Formatting
        test_input = cp.process.get_field(test_npz, in_field)
        in_y, in_x = test_input.shape

        ## Create List of Predicted Values
        pred_names = ['yspall', 'phi0', 'eta', 'sim_time']
        pred_labels = ['$Y_{spall}$', r'$\phi_0$', r'$\eta$', '$t_{sim}$']
        num_preds = np.size(pred_names)

    elif EXP == 'nestedcylinder':
        ## Data Formatting
        test_input = nc.process.get_field(test_npz, in_field)
        in_y, in_x = test_input.shape

        ## Create List of Predicted Values
        pred_names = ['sclPTW']
        pred_labels = ['Scaled\nPTW']
        num_preds = np.size(pred_names)

    ## Process Norm Choice
    if norm.isdigit():
        norm = int(norm)
    elif norm=='inf':
        norm = np.inf
    elif norm=='-inf':
        norm = -1*np.inf

    print('Data paths loaded and processed sucessfully.')

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
        model.eval()

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
    ## Extracting Features & Predictions
    #############################
    ## Making Empty Dataframe
    features1 = features+1
    ft_names = np.core.defchararray.add(['ft'] * n_features, features1.astype(str))
    norm_names = np.core.defchararray.add(['_norm'] * n_features, [str(norm)] * n_features)
    ftnorm_names = np.core.defchararray.add(ft_names, norm_names)
    col_names = np.concatenate((ftnorm_names, pred_names))
    dataframe = pd.DataFrame(columns=col_names, index=range(num_samples))

    ## Go through the Sample List
    for s, sample in enumerate(sample_list):
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
            model_in = img_in.reshape((1, 1, in_y, in_x))

        ## Extracting Features & Predictions
        if PACKAGE == 'tensorflow':
            extracted_fts = extractor(model_in)[0, :, :, :]
            extracted_fts = np.array(extracted_fts)

            pred = model.predict(model_in)
            dataframe.loc[s, pred_names] = pred.flatten()

        elif PACKAGE == 'pytorch':
            model_in = torch.tensor(model_in).to(torch.float32).to(device)
            extracted_fts_pyt = extractor(model_in)[lay].detach().numpy()[0, :, :, :]
            ## Fix the pytorch dimensions convention
            extracted_fts = np.zeros((ft_y, ft_x, n_features))
            for i in range(n_features):
                extracted_fts[:, :, i] = extracted_fts_pyt[i, :, :]

            pred = model.forward(model_in)[0].item()
            dataframe.loc[s, pred_names] = pred

        out_y, out_x, _ = np.shape(extracted_fts)
        
        ## Record Feature Norms
        for f, feature in enumerate(features):
            col = ftnorm_names[f]
            dataframe.loc[s, col] = np.linalg.norm(extracted_fts[:,:,feature], ord=norm)

        ## Print Status
        if num_samples > 20 and s%10 == 0:
            print(str(s)+' samples computed.\n')

    ## Convert the dataframe from containing objects to containing numerics
    for c in col_names: dataframe[c] = pd.to_numeric(dataframe[c])

    print('Features and predictions computed sucessfully.')

    #############################
    ## Computing Correlations
    #############################
    ## Making Empty Arrays
    corrs = np.empty((n_features, num_preds))
    pvals = np.empty((n_features, num_preds))
    partial_corrs = np.empty((n_features, num_preds))
    partial_pvals = np.empty((n_features, num_preds))
    partial_rank_corrs = np.empty((n_features, num_preds))
    partial_rank_pvals = np.empty((n_features, num_preds))

    ## Compute the Correlations
    for i, ft in enumerate(ftnorm_names):
        covars = list(np.setdiff1d(ftnorm_names, ft)) # create a list of the non-selected features to be the covariates

        for j, pred in enumerate(pred_names):

            ## Standard Correlation
            corr_out = pg.corr(dataframe[ft], dataframe[pred], method='pearson')
            corrs[i, j] = corr_out.loc['pearson', 'r']
            pvals[i, j] = corr_out.loc['pearson', 'p-val']

            ## Partial Correlation
            partial_out = dataframe.partial_corr(x=ft, y=pred, covar=covars, method='pearson')
            partial_corrs[i, j] = partial_out.loc['pearson', 'r']
            partial_pvals[i, j] = partial_out.loc['pearson', 'p-val']

            ## Partial Rank Correlation
            partial_rank_out = dataframe.partial_corr(x=ft, y=pred, covar=covars, method='spearman')
            partial_rank_corrs[i, j] = partial_rank_out.loc['spearman', 'r']
            partial_rank_pvals[i, j] = partial_rank_out.loc['spearman', 'p-val']

    ## Getting the Statiscally Significant Correlations
    threshold = 0.05 #threshold for if a correlation coefficient is statistically significant (if p-val > threshold, correlation coefficient is discarded)
    statsig_corrs = fns.mat.get_statsig(corrs, pvals, threshold=threshold)
    statsig_partial_corrs = fns.mat.get_statsig(partial_corrs, partial_pvals, threshold=threshold)
    statsig_partial_rank_corrs = fns.mat.get_statsig(partial_rank_corrs, partial_rank_pvals, threshold=threshold)

    ## Adding Averages to the Standard Correlation
    corrs = fns.mat.concat_avg(corrs, axis=0, spacer=True)
    corrs = fns.mat.concat_avg(corrs, axis=1, spacer=True)
    corrs[0,-1] = 0 #zero out the top right corner

    ## Adding Averages to the Partial Correlation
    partial_corrs = fns.mat.concat_avg(partial_corrs, axis=0, spacer=True)
    partial_corrs = fns.mat.concat_avg(partial_corrs, axis=1, spacer=True)
    partial_corrs[0,-1] = 0 #zero out the top right corner

    ## Adding Averages to the Partial Rank Correlation
    partial_rank_corrs = fns.mat.concat_avg(partial_rank_corrs, axis=0, spacer=True)
    partial_rank_corrs = fns.mat.concat_avg(partial_rank_corrs, axis=1, spacer=True)
    partial_rank_corrs[0,-1] = 0 #zero out the top right corner

    ## Making the Labels 
    if fixed_key=='None': fixed_key=''
    ft_labels = features1.astype(str)
    ft_labels_long = np.concatenate((['Abs. Col.\nAvg.', ''], features1))
    pred_labels_long = np.concatenate((pred_labels, ['', 'Abs. Row\nAvg.']))
    df_ft_labels = np.concatenate((['Abs. Col. Avg.', 'Spacer'], features1))
    df_pred_labels = np.concatenate((pred_names, ['Spacer', 'Abs. Row Avg.']))

    ## Saving the Standard Correlations
    save_path = fig_path + 'ft_pred_corr_'+fixed_key
    pd.DataFrame(data=corrs, index=df_ft_labels, columns=df_pred_labels).to_csv(save_path+'.csv')
    pd.DataFrame(data=pvals, index=ft_labels, columns=pred_names).to_csv(save_path+'_pvals.csv')
    pd.DataFrame(data=statsig_corrs, index=ft_labels, columns=pred_names).to_csv(save_path+'_statsig.csv')

    ## Saving the Partial Correlations
    save_path = fig_path + 'ft_pred_partialcorr_'+fixed_key
    pd.DataFrame(data=partial_corrs, index=df_ft_labels, columns=df_pred_labels).to_csv(save_path+'.csv')
    pd.DataFrame(data=partial_pvals, index=ft_labels, columns=pred_names).to_csv(save_path+'_pvals.csv')
    pd.DataFrame(data=statsig_partial_corrs, index=ft_labels, columns=pred_names).to_csv(save_path+'_statsig.csv')

    ## Saving the Partial Correlations
    save_path = fig_path + 'ft_pred_partialrankcorr_'+fixed_key
    pd.DataFrame(data=partial_rank_corrs, index=df_ft_labels, columns=df_pred_labels).to_csv(save_path+'.csv')
    pd.DataFrame(data=partial_rank_pvals, index=ft_labels, columns=pred_names).to_csv(save_path+'_pvals.csv')
    pd.DataFrame(data=statsig_partial_rank_corrs, index=ft_labels, columns=pred_names).to_csv(save_path+'_statsig.csv')

    print('Correlations computed and saved sucessfully.')

    #############################
    ## Generate Plots
    #############################
    xlabel = 'Model Prediction'
    ylabel = 'Norm of Features from Selected Layer'

    ## Standard Correlation
    save_path = fig_path + 'ft_pred_corr_'+fixed_key
    ## Correlation
    title = 'Correlation of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(corrs, xticklabels=pred_labels_long, yticklabels=ft_labels_long, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path)
    ## P-Values
    title = 'Correlation P-Vals of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(pvals, xticklabels=pred_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path+'_pvals')
    ## Statistically Significant Correlations
    title = 'Stat. Significant Correlation of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(statsig_corrs, xticklabels=pred_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path+'_statsig')

    ## Partial Correlation
    save_path = fig_path + 'ft_pred_partialcorr_'+fixed_key
    ## Correlation
    title = 'Partial Corr. of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(partial_corrs, xticklabels=pred_labels_long, yticklabels=ft_labels_long, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path)
    ## P-Values
    title = 'Partial Corr. P-Vals of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(partial_pvals, xticklabels=pred_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path+'_pvals')
    ## Statistically Significant Correlations
    title = 'Stat. Significant Partial Corr. of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(statsig_partial_corrs, xticklabels=pred_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path+'_statsig')

    ## Partial Rank Correlation
    save_path = fig_path + 'ft_pred_partialrankcorr_'+fixed_key
    ## Correlation
    title = 'Partial Rank Corr. of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(partial_rank_corrs, xticklabels=pred_labels_long, yticklabels=ft_labels_long, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path)
    ## P-Values
    title = 'Partial Rank Corr. P-Vals of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(partial_rank_pvals, xticklabels=pred_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path+'_pvals')
    ## Statistically Significant Correlations
    title = 'Stat. Significant Partial Rank Corr. of \nFeatures '+str(norm)+' Norm from '+lay+'\nwith Model Predictions\nover '+str(num_samples)+' '+fixed_key+' Sample(s)'
    fns.plot.heatmap_plot(statsig_partial_rank_corrs, xticklabels=pred_labels, yticklabels=ft_labels, xlabel=xlabel, ylabel=ylabel, title=title, save_path=save_path+'_statsig')

    print('Plots generated and saved.')
    