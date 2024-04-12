# FEATURES OVER FIELDS SCRIPT
"""
Plots a grid of features over hydrodynamic/radiographic fields
 - Can plot all features (``-T All``) or some selected features (``-T # #``)
 - Can plot all fields (``-F All``) or some selected features (``-F str str``)

Input Line for TF Coupon Models:
``python feature_over_field.py -P tensorflow -E coupon -M ../examples/tf_coupon/trained_pRad2TePla_model.h5 -IF pRad -IN ../examples/tf_coupon/data/r60um_tpl112_complete_idx00110.npz -DF ../examples/tf_coupon/coupon_design_file.csv -L activation_15 -T 1 2 3 4 -NM ft01 -F rho pRad eqps eqps_rate eff_stress -S ../examples/tf_coupon/figures/``

Input Line for PYT Nested Cylinder Models:
``python feature_over_field.py -P pytorch -E nestedcylinder -M ../examples/pyt_nestedcyl/trained_rho2PTW_model.path -IF rho -IN ../examples/pyt_nestedcyl/data/nc231213_Sn_id0643_pvi_idx00112.npz -DF ../examples/pyt_nestedcyl/nestedcyl_design_file.csv -L interp_module.interpActivations.10 -T 8 11 12 -F rho eqps eff_stress sound_speed -S ../examples/pyt_nestedcyl/figures/``
"""


#############################
## Packages
#############################
import os
import sys
import argparse
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
def feature_over_field_parser():
    descript_str = 'Plots a grid of selected features over selected fields'

    parser = argparse.ArgumentParser(prog='Feature Ploting over Fields',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')

    ## Package & Experiment
    setup.args.package(parser) # -P
    setup.args.experiment(parser) # -E

    ## Model Imports
    setup.args.model(parser) # -M
    setup.args.input_field(parser) # -IF
    setup.args.input_npz(parser) # -IN
    setup.args.design_file(parser) # -DF

    ## Print Toggles
    setup.args.print_layers(parser) # -PL
    setup.args.print_features(parser) # -PT
    setup.args.print_fields(parser) # -PF

    ## Layer Properties
    setup.args.layer(parser) # -L
    setup.args.features(parser, dft=['All']) # -T
    setup.args.mat_norm(parser) # -NM

    ## Data Properties
    setup.args.fields(parser) # -F

    ## Color Properties
    setup.args.alpha1(parser, dft=0.25) # -A1
    setup.args.alpha2(parser, dft=1.00) # -A2
    setup.args.color1(parser, dft='yellow') # -C1
    setup.args.color2(parser, dft='red') # -C2

    ## Save Properties
    setup.args.save_fig(parser) # -S

    return parser

parser = feature_over_field_parser()

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
    input_path = args.INPUT_NPZ
    design_path = args.DESIGN_FILE

    ## Print Toggles
    PRINT_LAYERS = args.PRINT_LAYERS
    PRINT_FEATURES = args.PRINT_FEATURES
    PRINT_FIELDS = args.PRINT_FIELDS

    ## Layer Properties
    lay = args.LAYER
    features = args.FEATURES
    norm = args.MAT_NORM

    ## Data Properties
    fields = args.FIELDS

    ## Color Properties
    alpha1 = args.ALPHA1
    alpha2 = args.ALPHA2
    color1 = args.COLOR1
    color2 = args.COLOR2

    ## Save Properties
    fig_path = args.SAVE_FIG

    #############################
    ## Data Processing
    #############################

    npz = np.load(input_path)

    ## Data Prints
    if PRINT_FIELDS: 
        setup.data_prints.print_fields(npz)
        sys.exit()

    ## Data Checks
    setup.data_checks.check_fields(npz, [in_field])
    setup.data_checks.check_fields(npz, fields)

    if EXP == 'coupon':
        ## Package Imports
        import fns.coupondata as cp
        prefix = 'cp.'

        ## Input Processing
        img_in = cp.process.get_field(npz, in_field)
        Rlabels, Rticks, Zlabels, Zticks = cp.process.get_ticks(npz)
        key = cp.process.npz2key(input_path)
        dcj = cp.process.csv2dcj(design_path, key)
        model_in = [img_in, dcj]

        in_y, in_x = img_in.shape
        model_in = [img_in.reshape((1,in_y,in_x,1)),dcj]

        ## Fields Processing
        fields, n_fields = cp.process.remove_fields(npz, fields, NoneValid=True)

        ## Plotting Settings
        bigfigsize = (20,10)
        RMinorTicks = False
        ZMinorTicks = False

    elif EXP == 'nestedcylinder':
        ## Package Imports
        import fns.nestedcylinderdata as nc
        prefix = 'nc.'

        ## Input Processing
        img_in = nc.process.get_field(npz, in_field)
        Rlabels, Rticks, Zlabels, Zticks = nc.process.get_ticks(npz)
        model_in = img_in

        ## Fields Processing
        fields, n_fields = nc.process.remove_fields(npz, fields, NoneValid=True)

        ## Plotting Settings
        bigfigsize = (15,25)
        RMinorTicks = True
        ZMinorTicks = False

    print('Data loaded and processed sucessfully.')

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

        ## Extract Features
        ft_mat = tfc.fts.feature_extractor(model, lay, model_in, norm)
        n_features, features = tfc.fts.parse_features(model, lay, features)

    elif PACKAGE == 'pytorch':
        ## Package Imports
        import torch
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

        ## Extract Features
        ft_mat = pytc.fts.feature_extractor(model, lay, model_in, norm, device)
        n_features, features = pytc.fts.parse_features(model, features)

    ## Save Extracted Features
    lay = lay.replace('.', '_')
    fns.save.features2npz(ft_mat, save_path = os.path.join(fig_path, lay+'_all_features') )

    print('Model loaded sucessfully; features extracted and saved.')

    #############################
    ## Plot Features
    #############################
    plt.rcParams.update({'font.size': 22})
    fns.plot.custom_colormap(color1, color2, alpha1, alpha2)
    zeros_field = np.zeros_like(img_in)
    n_fields = np.size(fields)
    lims = [0,1]

    fig, axs = plt.subplots(n_features, n_fields, figsize=bigfigsize, layout="constrained")

    for j, field in enumerate(fields):
        if field == 'none':
            field_pic = zeros_field
        else:
            field_pic = eval(prefix+'process.get_field(npz, field)')

        for i, feature in enumerate(features):
            plot_ix = (n_fields*i) + j
            plot_ix+=1 #for subplot correctly
            plt.subplot(n_features,n_fields,plot_ix)
            im = fns.plot.feature_plot(field_pic, ft_mat[:,:,feature], Rlabels, Rticks, Zlabels, Zticks, RMinorTicks, ZMinorTicks, lims)

            #Title over first row
            if i == 0:
                plt.title(field, fontsize=16)
            #Label left of first column
            if j == 0:
                plt.ylabel('Feature '+str(feature+1), rotation=0, fontsize=16, labelpad=40)

    fig.colorbar(im, ax=axs, label='Activation Intensity')
    plt.clim(lims)
    fig.suptitle('Features from '+lay+'\nOver Radiograph/Hydrodynamic Fields')

    fig.savefig(os.path.join(fig_path, lay+'_features_over_fields.png'))
    plt.close()

    print('Plots generated and saved.')


