# PYTORCH MODEL DEFINITION FOR NESTED CYLINDER PROBLEM
"""
Contains the model class for the pytorch nested cylinder models.

Named **N**ested **Cyl**inder **A**rtificial **N**eural **N**etwork **V**ersion **1**
"""

#############################
## Packages
#############################
import torch
import torch.nn as nn

#############################
## Model Definition
#############################
class NCylANN_V1(nn.Module):
    def __init__(self, 
                 img_input_shape: tuple[int, int, int, int],
                 NOutput: int=1, 
                 Nfilters: int=12, 
                 Ninterp: int=15,
                 outputHiddenLayerSize: int=20,
                 BNmomentum: float=0.99,
                 kernelSize: tuple[int, int]=(5,5)):
        """ Model definition for nested cylinder -> scaled PTW value neural network

            Args:
                img_input_shape (tuple[int, int, int, int]): shape of image input (batchsize, channels, height, width)
                NOutput (int): number of predictions; =1 for scaled PTW prediction
                Nfilters (int): number of features
                Ninterp (int): number of interpretability blocks
                outputHiddenLayerSize (int): number of hidden features in dense layers
                BNmomentum (float): momentum value for batch normalization layers
                kernelSize (tuple[int, int]): size of kernels in the convolutional layers

        """
        super(NCylANN_V1,self).__init__()
        self.img_input_shape = img_input_shape
        self.Nfilters = Nfilters
        self.Ninterp = Ninterp
        self.NOutput = NOutput
        self.BNmomentum = BNmomentum
        self.kernelSize = kernelSize
        
        ## Input Layers
        self.inConv = nn.Conv2d(in_channels = self.img_input_shape[1],
                           out_channels = self.Nfilters, 
                           kernel_size = self.kernelSize, 
                           stride = (1,1), 
                           padding='same', 
                           bias=False)
        self.inConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.inConvActivation = nn.SiLU()
        
        # Interpretability Layers 
        self.interpLayers = nn.ModuleList()
        self.interpBatchNorms = nn.ModuleList()
        self.interpActivations = nn.ModuleList()
        for i in range(0,self.Ninterp):
            interpLayer = nn.Conv2d(in_channels = self.Nfilters,
                           out_channels = self.Nfilters, 
                           kernel_size = self.kernelSize,
                           stride = (1,1),
                           padding='same', 
                           bias=False)
            self.interpLayers.append(interpLayer)
            self.interpBatchNorms.append(nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True))
            self.interpActivations.append(nn.SiLU())
        
        ## Reducing Layers
        self.r1Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,1), 
                                       padding=(2,2), 
                                       bias=False)
        self.r1ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r1ConvActivation = nn.SiLU()

        self.r2Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r2ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r2ConvActivation = nn.SiLU()

        self.r3Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r3ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r3ConvActivation = nn.SiLU()

        self.r4Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r4ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r4ConvActivation = nn.SiLU()

        self.r5Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r5ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r5ConvActivation = nn.SiLU()

        self.r6Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,1), 
                                       padding=(4,2), 
                                       bias=False)
        self.r6ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r6ConvActivation = nn.SiLU()

        self.r7Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(3,2), 
                                       bias=False)
        self.r7ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r7ConvActivation = nn.SiLU()

        self.r8Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r8ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r8ConvActivation = nn.SiLU()
        
        ## Final Convolutional Layer: no batchnorm, non-reducing
        self.endConv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (1,1), 
                                       padding=(2,2), 
                                       bias=False)
        self.endConvActivation = nn.SiLU()
        
        ## Hidden Dense Layer
        self.flattenLayer = nn.Flatten()
        self.h1 = nn.Linear(768, outputHiddenLayerSize)
        self.h1Activation = nn.SiLU()
        
        ## Linear Output Layer
        self.linOut = nn.Linear(outputHiddenLayerSize,NOutput)
        
    def forward(self,x):
        ## Input Layers
        x = self.inConv(x)
        x = self.inConvBatch(x)
        x = self.inConvActivation(x)
        
        ## Interpretability Layers
        for i in range(0,self.Ninterp):
            x = self.interpLayers[i](x)
            x = self.interpBatchNorms[i](x)
            x = self.interpActivations[i](x)
        
        ## Reducing Layers
        x = self.r1Conv(x)
        x = self.r1ConvBatch(x)
        x = self.r1ConvActivation(x)

        x = self.r2Conv(x)
        x = self.r2ConvBatch(x)
        x = self.r2ConvActivation(x)

        x = self.r3Conv(x)
        x = self.r3ConvBatch(x)
        x = self.r3ConvActivation(x)

        x = self.r4Conv(x)
        x = self.r4ConvBatch(x)
        x = self.r4ConvActivation(x)

        x = self.r5Conv(x)
        x = self.r5ConvBatch(x)
        x = self.r5ConvActivation(x)

        x = self.r6Conv(x)
        x = self.r6ConvBatch(x)
        x = self.r6ConvActivation(x)

        x = self.r7Conv(x)
        x = self.r7ConvBatch(x)
        x = self.r7ConvActivation(x)

        x = self.r8Conv(x)
        x = self.r8ConvBatch(x)
        x = self.r8ConvActivation(x)

        ## Final Convolutional Layer
        x = self.endConv(x)
        x = self.endConvActivation(x)
        
        ## Hidden Dense Layer
        x = self.flattenLayer(x)
        x = self.h1(x)
        x = self.h1Activation(x)
        
        ## Linear Output Layer
        x = self.linOut(x)
        x = x.flatten()
        
        return(x)