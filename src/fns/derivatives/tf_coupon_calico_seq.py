# CALICO SEQUENCE DEFINITION FOR BRANCHED TENSORFLOW COUPON MODELS
"""
Contains the keras sequence classes to use with the when calculating feature derivities with the calico models

Execution will print unit test information, perform unit tests, and print the results to the terminal.

Input Line:
``python tf_coupon_calico_seq.py -M ../../../../network_files/tfmodels/study_02_221216_prad2tepla_deterministic_model_0100.h5 -IF pRad -ID ../../../../network_files/coupondata/ -DF ../../../../network_files/coupondata/design_res60um_tepla_study220620_MASTER.csv -NF ../../../../network_files/coupondata/r60um_normalization.npz -L activation_15 -T 1 -NS 40``
"""

#############################
## Packages
#############################
import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import glob
import re
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.utils import Sequence
import tensorflow.keras.preprocessing.image as tfkpi

#############################
## Custom Functions Import Code
#############################

sys.path.insert(0, os.path.abspath('../../'))
import fns
import fns.setup as setup
import fns.coupondata as cp
import fns.tfcustom as tfc
import fns.tfcustom.calico as tfcalico
from fns.derivatives.tf_coupon_calico_model import make_calico

       
#############################
## Calico Sequence Definition
############################# 
class calicoSEQ(Sequence):
    """ The definition of a sequence object used as input to the tensorflow coupon calico neural networks.
        
        Args:
            input_field (str): The radiographic/hydrodynamic field the model is trained on
            input_dir (str): The directory path where all of the .npz files are stored 
            filelist (str): Text file listing file names to read.
            design_file (str): .csv file with master design study parameters
            normalization_file (str): Full-path to file containing normalization information.
            batch_size (int): Number of samples in each batch
            epoch_length (int): Number of batches in an epoch
            layshape (3 tuple): the size of the output of the specified layer (outY, outX, Nfeatures)
            ftIDX (int): index of the feature to scale; feature w.r.t. the derivative is taken
            dScale (float): derivative scaling factor
    """

    def __init__(self,
                 #Model Arguments
                 input_field: str='rho',
                 input_dir: str='/data/coupon_data/',
                 filelist: str='../../coupon_ml/yellow_r60um_tpl_testing.txt',
                 design_file: str='../../coupon_ml/design_res60um_tepla_study220620_MASTER.csv',
                 normalization_file: str='../../coupon_ml/r60um_normalization.npz',
                 batch_size: int=8,
                 epoch_length: int=10,
                 #Derivative Argumets
                 layshape: tuple=(300, 1000, 12),
                 ftIDX: int=0,
                 dScale: float=0.001):
        
        ## Model Arguments
        self.input_field = input_field
        self.input_dir = input_dir
        self.filelist = filelist
        self.design_file = design_file
        self.normalization_file = normalization_file
        self.batch_size = batch_size
        self.epoch_length = epoch_length
        ## Derivative Arguments
        self.layshape = layshape
        self.ftIDX = ftIDX
        self.dScale = dScale

        ## Create filelist
        with open(filelist, 'r') as f:
            self.filelist = [line.rstrip() for line in f]

        self.Nsamples = len(self.filelist)

        ## Read normalizations
        npzFile = np.load(self.normalization_file)
        self.sim_params_mean = npzFile['sim_params_mean']
        self.sim_params_std = npzFile['sim_params_std']
        self.tepla_mean = npzFile['tepla_mean']
        self.tepla_std = npzFile['tepla_std']
        npzFile.close()

        ## Read design file into pandas dataframe
        self.designDF = pd.read_csv(self.design_file,
                                        sep=',',
                                        header=0,
                                        index_col=0,
                                        engine='python')


    def __len__(self):
        """
        Return the epoch length. Epoch length is the number of batches that will be trained on per epoch.
        """
        return self.epoch_length

    def __getitem__(self, idx):
        """Return a tuple of a batch's input and output data for training at a 
        given index within the epoch.

        """
        start = idx*self.batch_size
        if (start+self.batch_size) < self.Nsamples:
            batch_list = self.filelist[start:(start+self.batch_size)]
        else:
            self.filelist = self.filelist + self.filelist
            self.Nsamples = len(self.filelist)
            batch_list = self.filelist[start:(start+self.batch_size)]
        
        imageList = []
        scaleList = []
        TePlaList = []
        DcjList = []
        for filename in batch_list:
            ## Get index to design file from NPZ filename
            key = cp.process.npz2key(filename)
            tpl = float(re.split('tpl', key)[-1])
            idx_long = cp.process.npz2idx(filename)
            idX = float(re.split('idx', idx_long)[-1])

            ## Grab TePla parameters from the design file
            yspall = self.designDF.loc[key, 'yspall']
            log10_phi0 = self.designDF.loc[key, 'log10_phi0']
            eta = self.designDF.loc[key, 'eta']

            ## Get input Dcj value
            Dcj = self.designDF.loc[key, 'Dcj']
            Dcj = np.array([float(Dcj.split(' ')[3])])
            
            ## Read the file
            npz = np.load(os.path.join(self.input_dir, filename))
            in_image = cp.process.get_field(npz, self.input_field)
            sim_time = npz['sim_time']

            ## Append quantities to lists
            TePla = np.array([sim_time, yspall, log10_phi0, eta])
            TePla = (TePla - self.tepla_mean) / self.tepla_std
            TePla = np.append(TePla, [tpl, idX])
            TePla = np.expand_dims(TePla, axis=0)
            # TePla: [sim_time, yspall, log10_phi0, eta, tpl, idx]
            TePlaList.append(TePla)

            Dcj = (Dcj - self.sim_params_mean) / self.sim_params_std
            Dcj = np.expand_dims(Dcj, axis=0)
            DcjList.append(Dcj)

            in_image = np.expand_dims(in_image, axis=2)
            in_image = np.expand_dims(in_image, axis=0)
            imageList.append(in_image)

            npz.close()

            scale_mat = np.ones(self.layshape)
            scale_mat[:, :, self.ftIDX] = scale_mat[:, :, self.ftIDX] + self.dScale
            scale_mat = np.expand_dims(scale_mat, axis=0)
            scaleList.append(scale_mat)
            
        ## Concatenate arrays
        imageSet = np.concatenate(imageList, axis=0)
        scaleSet = np.concatenate(scaleList, axis=0)
        DcjSet = np.concatenate(DcjList, axis=0)
        TePlaSet = np.concatenate(TePlaList, axis=0)
        
        return [imageSet, scaleSet, DcjSet, TePlaSet], TePlaSet


#############################
## Set Up Parser
#############################
def calico_seq_parser():
    descript_str = 'Creates and tests a calcio sequence (for input to a calico model) given an input model, layer, and feature'

    parser = argparse.ArgumentParser(prog='Calico Sequence',
                                     description=descript_str,
                                     fromfile_prefix_chars='@')

    ## Model Imports
    setup.args.model(parser) # -M
    setup.args.input_field(parser) # -IF
    setup.args.input_dir(parser) # -ID
    setup.args.file_list(parser) # -FL
    setup.args.design_file(parser) # -DF
    setup.args.norm_file(parser) # -NF

    ## Print Toggles
    setup.args.print_layers(parser) # -PL
    setup.args.print_features(parser) # -PT
    setup.args.print_samples(parser) # -PS

    ## Layer Properties
    setup.args.layer(parser) # -L
    setup.args.features(parser, dft=[1]) # -T
    setup.args.dScale(parser) #-DS

    ## Data Properties
    setup.args.num_samples(parser) # -NS

    return parser

parser = calico_seq_parser()

#############################
## Unit Tests for Calico Sequence
#############################
if __name__ == '__main__':

    ## Parse Args
    args = parser.parse_args()

    ## Model Imports
    model_path = args.MODEL
    in_field = args.INPUT_FIELD
    input_dir = args.INPUT_DIR
    file_list_path = args.FILE_LIST
    design_path = args.DESIGN_FILE
    norm_path = args.NORM_FILE

    ## Print Toggles
    PRINT_LAYERS = args.PRINT_LAYERS
    PRINT_FEATURES = args.PRINT_FEATURES
    PRINT_SAMPLES = args.PRINT_SAMPLES

    ## Layer Properties
    lay = args.LAYER
    features = args.FEATURES
    dScale = args.D_SCALE

    ## Data Properties
    num_samples = args.NUM_SAMPLES

    ## Load Model
    model = keras.models.load_model(model_path)
    print('\n') #Seperate other output from Tensorflow 

    ## Try to Rename the SIM_PRMS Input Layer (not all models require this)
    try:
        prm_in_lay =  model.get_layer('param_input')
        prm_in_lay._name = 'SIM_PRMS_input'
        print('\nLayer "param_input" has been renamed to "SIM_PRMS_input".\n')
    except:
        pass
    print('\n') #Seperate other output from Tensorflow warnings

    ## Get a Test File
    search_dir = input_dir+'r*tpl*idx*.npz'
    test_path = glob.glob(search_dir)[0]
    test_npz = np.load(test_path)

    ## Prints
    if PRINT_LAYERS: tfc.prints.print_layers(model)
    if PRINT_FEATURES: tfc.prints.print_features(model, lay)
    if PRINT_SAMPLES: setup.data_prints.print_samples(search_dir)
    if PRINT_LAYERS or PRINT_FEATURES or PRINT_SAMPLES: sys.exit()

    ## Checks
    tfcalico.check_calico_layer(model, lay, branch='SIM_PRMS', catlay='SIM_PRMS_IMG_concat')
    tfcalico.check_calico_features(model, lay, features)
    setup.data_checks.check_fields(test_npz, [in_field])
    if file_list_path == 'MAKE': 
        setup.data_checks.check_samples(num_samples, search_dir)

    ## Printing Unit Test Information
    print('This script runs unit tests for the Calico sequences.\n')
    print('The calcio sequence creates four inputs to the calico network: [img_input, const_input, prms_input, truth_input]')
    print('and one output: truth_output.\n')
    print('The unit tests print the shapes of all the inputs and outputs. The user must determine if the shapes are correct.\n')
    print('The unit tests check if the const_input contians all ones in unselected features and contains 1+dScale for the selected feature.\n')
    print('The unit tests check if the truth input and truth output are identical.\n')
    print('-----\n')

    ## Processing the File List
    ft = int(features[0])-1
    if file_list_path == 'MAKE':
        num_samples, sample_list = fns.save.makefilelist(search_dir, num_samples, save_path='./calico_seq_test', save=True)
        file_list_path = './calico_seq_test_SAMPLES.txt'
    else:
        num_samples, sample_list = fns.misc.load_filelist(file_list_path)

    ## Create the Sequence Inputs
    layshape = model.get_layer(name=lay).output_shape[1:]

    ## Initiate the Sequence
    seq = calicoSEQ(#Model Arguments
                     input_field=in_field,
                     input_dir=input_dir,
                     filelist='./calico_seq_test_SAMPLES.txt',
                     design_file=design_path,
                     normalization_file=norm_path,
                     batch_size=8,
                     epoch_length=10,
                     #Derivative Argumets
                     layshape=layshape,
                     ftIDX=ft,
                     dScale=dScale)

    ## Evaluate the Sequence
    IDX = 2 #Sample one batch
    InputList, OutputSet = seq.__getitem__(IDX)

    ## Delete the Samples .txt file
    if file_list_path == './calico_seq_test_SAMPLES.txt': os.remove('./calico_seq_test_SAMPLES.txt')

    ## Complete Checks
    scale_mats = InputList[1]
    ones_check = np.all(scale_mats[0, 0, 0, 0:ft]==1) and np.all(scale_mats[0, 0, 0, ft+1:]==1)
    dScale_check = np.all(scale_mats[0, 0, 0, ft]==1+dScale)

    TePla = InputList[3]
    TePla_check = TePla - OutputSet

    ## Print Unit Tests to Terminal
    print('Calico Sequence Unit Test Results\n')
    print('Image input array shape:', InputList[0].shape)
    print('Constant input (scale matrix) array shape:', InputList[1].shape)
    if ones_check:
        print('Constant input (scale matrix)  passes the Ones Check: all values at non-selected features equal 1.')
    else:
        print('Constant input (scale matrix)  FAILS the Ones Check: NOT all values at non-selected features equal 1.')
    if dScale_check:
        print('Constant input (scale matrix)  passes the dScale Check: all values at the selected feature equal ', 1+dScale,'.')
    else:
        print('Constant input (scale matrix)  FAILS the dScale Check: NOT all values the selected feature equal ', 1+dScale,'.')
    print('Prms (SIM_PRMS) input array shape:', InputList[2].shape)
    print('Truth input (TePla Parameters) array shape:', InputList[3].shape)
    print('Truth input = [sim_time, yspall, log10_phi0, eta, tpl, idx]')
    print('\t', TePla[0])
    print('Truth output (TePla Parameters) array shape:', OutputSet.shape)
    if np.all(TePla_check==0):
        print('Truth (TePla Parameters) inputs and outputs are identical.')
    else:
        print('Truth (TePla Parameters) inputs and outputs are NOT identical; error present.')
