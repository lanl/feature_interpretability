# FEATURE SENSITIVITY SCRIPT
"""
Plots the average and standard deviation of features extracted from a set of related inputs
 - Averages each pixel across multiple inputs

Plots average and standard deviations in seperate figures
 - Can plot all features from a layer on the same plot (``-T Grid``)
 - Can plot all features from a layer on their own plots (``-T All``)
 - Can plot some features from a layer on their own plots (``-T # #``)

Fixed key (``-XK``) specifies what subset of data to consider
 - 'None' can be passed to consider any input with no restrictions
 - For coupon data, fixed keys must be in the form 'tpl###' or 'idx#####'
 - For nested cylinder data, fixed keys must be in the form 'id####' or 'idx#####'

Saves all averages and standard deviations to a .npz file.

| Samples can be preselected and listed in a .txt file (``-FL filepath``) OR
| Number of samples can be specified and a random selection satisfying the fixed key requirement will be made (``-FL MAKE -NS #``)

Input Line for TF Coupon Models: 
``python feature_sensitivity.py -P tensorflow -E coupon -M ../examples/tf_coupon/trained_pRad2TePla_model.h5 -IF pRad -ID ../examples/tf_coupon/data/ -DF ../examples/tf_coupon/coupon_design_file.csv -L activation_15 -T Grid -NM ft01 -XK idx00110 -S ../examples/tf_coupon/figures/``

Input Line for PYT Nested Cylinder Models: 
``python feature_sensitivity.py -P pytorch -E nestedcylinder -M ../examples/pyt_nestedcyl/trained_rho2PTW_model.path -IF rho -ID ../examples/pyt_nestedcyl/data/ -DF ../examples/pyt_nestedcyl/nestedcyl_design_file.csv -L interp_module.interpActivations.10 -T Grid -NM ft01 -XK idx00112 -S ../examples/pyt_nestedcyl/figures/``
"""


#############################
## Packages
#############################
import os
import sys
import argparse
import glob
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
def feature_sensitivity_parser():
    descript_str = 'Plots the average and standard deviation of features extracted from a set of related inputs'

    parser = argparse.ArgumentParser(prog='Forward Sensitivity',
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
    setup.args.features(parser, dft=['Grid']) # -T
    setup.args.mat_norm(parser) # -NM

    ## Data Properties
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

parser = feature_sensitivity_parser()

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
    norm = args.MAT_NORM

    ## Data Properties
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
        file_path = os.path.join(fig_path, 'feature_sensitivity')
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
    setup.data_checks.check_fields(test_npz, [in_field])

    if EXP == 'coupon':
        ## Package Imports
        import fns.coupondata as cp

        test_input = cp.process.get_field(test_npz, in_field)
        in_y, in_x = test_input.shape

        ## Plotting Settings
        Rlabels, Rticks, Zlabels, Zticks = cp.process.get_ticks(test_npz)
        bigfigsize = (20,10)
        smallfigsize = (10,5)
        RMinorTicks = False
        ZMinorTicks = False

    elif EXP == 'nestedcylinder':
        ## Package Imports
        import fns.nestedcylinderdata as nc

        test_input = nc.process.get_field(test_npz, in_field)
        in_y, in_x = test_input.shape

        ## Plotting Settings
        Rlabels, Rticks, Zlabels, Zticks = nc.process.get_ticks(test_npz)
        bigfigsize = (10,20)
        smallfigsize = (5,10)
        RMinorTicks = True
        ZMinorTicks = False

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

        ## Getting Size Information
        _, ft_y, ft_x, ft_n = model.get_layer(name=lay).output_shape

        ## Make Extractor
        extractor = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer(name=lay).output)


    elif PACKAGE == 'pytorch':
        ## Package Imports
        import torch
        import torchvision.models.feature_extraction as ftex
        import fns.pytorchcustom as pytc
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## Load Model
        import fns.pytorchcustom.field2PTW_model_definition as field2PTW_model
        model = field2PTW_model.field2PTW(img_size = (1, 1700, 500),
                                            size_threshold = (8, 8),
                                            kernel = 5,
                                            features = 12, 
                                            interp_depth = 12,
                                            conv_onlyweights = False,
                                            batchnorm_onlybias = False,
                                            act_layer = torch.nn.GELU,
                                            hidden_features = 20)

        ## Prints
        if PRINT_LAYERS: pytc.prints.print_layers(model)
        if PRINT_FEATURES: pytc.prints.print_features(model)
        if PRINT_LAYERS or PRINT_FEATURES: sys.exit()

        ## Checks
        pytc.checks.check_layer(model, lay)
        pytc.checks.check_features(model, features)

        ## Getting Size Information
        ft_y, ft_x = in_y, in_x
        ft_n = model.features

        ## Make Extractor 
        extractor = ftex.create_feature_extractor(model, return_nodes={lay: lay})
        

    print('Model loaded sucessfully.')

    #############################
    ## Extracting Features
    #############################
    ft_tensor = np.empty([n_samples, ft_y, ft_x, ft_n])

    for s, sample in enumerate(sample_list):
        path = os.path.join(input_dir, sample)
        npz = np.load(path)

        ## Data Processing Model Input
        if EXP == 'coupon':
            img_in = cp.process.get_field(npz, in_field) 
            key = cp.process.npz2key(path)
            dcj = cp.process.csv2dcj(design_path, key)
            model_in = [img_in.reshape((1,in_y,in_x,1)),dcj]

        elif EXP == 'nestedcylinder':
            img_in = nc.process.get_field(npz, in_field)
            model_in = img_in

        ## Extracting Features
        if PACKAGE == 'tensorflow':
            extracted_fts = extractor(model_in)[0, :, :, :]

        elif PACKAGE == 'pytorch':
            model_in = torch.tensor(model_in.reshape((1, 1, in_y, in_x))).to(torch.float32).to(device)
            extracted_fts_pyt = extractor(model_in)[lay].detach().numpy()[0, :, :, :]
            ## Fix the pytorch dimensions convention
            extracted_fts = np.zeros((ft_y, ft_x, ft_n))
            for i in range(ft_n):
                extracted_fts[:, :, i] = extracted_fts_pyt[i, :, :]

        ## Normalize Features
        ft_mat = fns.mat.normalize_mat(extracted_fts, norm)

        ## Store Fatures
        ft_tensor[s, :, :, :] = ft_mat

        ## Print Status
        if n_samples > 20 and s%10 == 0:
            print(str(s)+' samples computed.\n')

    print('Features extracted sucessfully.')

    #############################
    ## Computing Mean & Standard Deviation
    #############################

    ft_avgs = np.empty([ft_y, ft_x, ft_n])
    ft_stds = np.empty([ft_y, ft_x, ft_n])

    for n in range(ft_n):
        ft_avgs[:, :, n] = np.nanmean(ft_tensor[:,:,:,n], axis=0)
        ft_stds[:, :, n] = np.nanstd(ft_tensor[:,:,:,n], axis=0)

    ## Saving the Avgerage and Standard Deivation
    lay = lay.replace('.', '_')
    if fixed_key=='None': fixed_key=''
    fns.save.features2npz(ft_avgs, save_path=os.path.join(fig_path, lay+'_all_features_avg_'+fixed_key), ft_suffix='_avg')
    fns.save.features2npz(ft_stds, save_path=os.path.join(fig_path, lay+'_all_features_std_'+fixed_key), ft_suffix='_std')

    print('Feature averages & standard deivaitons computed and saved sucessfully.')

    #############################
    ## Plot Features
    #############################
    zeros_in = np.zeros((in_y, in_x))
    fns.plot.custom_colormap(color1, color2, alpha1, alpha2)
    if fixed_key=='': fixed_key=str(n_samples)+'_Samples'

    if features == ['Grid']: #Plot all features in a grid
        plt.rcParams.update({'font.size': 22})

        ## Plotting the Average
        avg_lims = [np.min(ft_avgs), np.max(ft_avgs)]
        fig, axs = plt.subplots(4, 3, figsize=bigfigsize, layout="constrained")
        for i in range(ft_n):
            plt.subplot(4,3,i+1)
            im = fns.plot.feature_plot(zeros_in, ft_avgs[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, avg_lims)
            plt.title('Feature '+str(i+1), fontsize=16)

        fig.colorbar(im, ax=axs, label='Activation Intensity')
        plt.clim(avg_lims)
        fig.suptitle('Average Features\nFrom '+lay+' \nfor '+fixed_key)

        fig.savefig(os.path.join(fig_path, lay+'_all_features_avg_'+fixed_key+'.png'))
        plt.close()

        ## Plotting the Standard Deviation
        std_lims = [np.min(ft_stds), np.max(ft_stds)]
        fig, axs = plt.subplots(4, 3, figsize=bigfigsize, layout="constrained")
        for i in range(ft_n):
            plt.subplot(4,3,i+1)
            im = fns.plot.feature_plot(zeros_in, ft_stds[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, std_lims)
            plt.title('Feature '+str(i+1), fontsize=16)

        fig.colorbar(im, ax=axs, label='Activation Intensity')
        plt.clim(std_lims)
        fig.suptitle('Standard Deviation of\nFeatures from '+lay+'\nfor '+fixed_key)

        fig.savefig(os.path.join(fig_path, lay+'_all_features_std_'+fixed_key+'.png'))
        plt.close()


    elif features == ['All']: #Plot all features in thier own 
        plt.rcParams.update({'font.size': 14})

        for i in range(ft_n):
            ## Plotting the Average
            avg_lims = [np.min(ft_avgs[:, :, i]), np.max(ft_avgs[:, :, i])]
            fig = plt.figure(figsize=smallfigsize, layout="constrained")
            im = fns.plot.feature_plot(zeros_in, ft_avgs[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, avg_lims)
            plt.suptitle('Average Feature '+str(i+1)+'\nFrom '+lay+' for '+fixed_key)
            fig.colorbar(im, label='Activation Intensity')
            plt.clim(avg_lims)
            
            fig.savefig(os.path.join(fig_path, lay+'_feature'+str(i+1)+'_avg_'+fixed_key+'.png'))
            plt.close()

            ## Plotting the Standard Deviation
            std_lims = [np.min(ft_stds[:, :, i]), np.max(ft_stds[:, :, i])]
            fig = plt.figure(figsize=smallfigsize, layout="constrained")
            im = fns.plot.feature_plot(zeros_in, ft_stds[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, std_lims)
            plt.suptitle('Standard Deviation of Feature '+str(i+1)+'\nFrom '+lay+' for '+fixed_key)
            fig.colorbar(im, label='Activation Intensity')
            plt.clim(std_lims)
            
            fig.savefig(os.path.join(fig_path, lay+'_feature'+str(i+1)+'_std_'+fixed_key+'.png'))
            plt.close()


    else: #Plotting a specific list of features
        plt.rcParams.update({'font.size': 14})
        features = [int(i)-1 for i in features]

        for i in features:
            ## Plotting the Average
            avg_lims = [np.min(ft_avgs[:, :, i]), np.max(ft_avgs[:, :, i])]
            fig = plt.figure(figsize=smallfigsize, layout="constrained")
            im = fns.plot.feature_plot(zeros_in, ft_avgs[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, avg_lims)
            plt.suptitle('Average Feature '+str(i+1)+'\nFrom '+lay+' for '+fixed_key)
            fig.colorbar(im, label='Activation Intensity')
            plt.clim(avg_lims)
            
            fig.savefig(os.path.join(fig_path, lay+'_feature'+str(i+1)+'_avg_'+fixed_key+'.png'))
            plt.close()

            ## Plotting the Standard Deviation
            std_lims = [np.min(ft_stds[:, :, i]), np.max(ft_stds[:, :, i])]
            fig = plt.figure(figsize=smallfigsize, layout="constrained")
            im = fns.plot.feature_plot(zeros_in, ft_stds[:,:,i], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, std_lims)
            plt.suptitle('Standard Deviation of Feature '+str(i+1)+'\nFrom '+lay+' for '+fixed_key)
            fig.colorbar(im, label='Activation Intensity')
            plt.clim(std_lims)
            
            fig.savefig(os.path.join(fig_path, lay+'_feature'+str(i+1)+'_std_'+fixed_key+'.png'))
            plt.close()

    print('Plots generated and saved.')

