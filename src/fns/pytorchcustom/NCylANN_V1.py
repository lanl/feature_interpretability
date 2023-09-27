# PYTORCH MODEL DEFINITION FOR NESTED CYLINDER PROBLEM
"""
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
                 img_input_shape,
                 NOutput=1, 
                 Nfilters = 12, 
                 Ninterp = 15,
                 outputHiddenLayerSize = 20,
                 useBN=True,
                 BNmomentum=0.99,
                 kernelSize=(5,5)):
        """Model construction function. The output model will take a sequence and output
        a vector of length NOutput.
        """
        super(NCylANN_V1,self).__init__()
        self.img_input_shape = img_input_shape
        self.Nfilters = Nfilters
        self.Ninterp = Ninterp
        self.NOutput = NOutput
        self.useBN = useBN
        self.BNmomentum = BNmomentum
        self.kernelSize = kernelSize
        
        # Input: torch.Size([4, 12, 1700, 500])
        ###INPUT LAYER###:
        self.inConv = nn.Conv2d(in_channels = self.img_input_shape[1],
                           out_channels = self.Nfilters, 
                           kernel_size = self.kernelSize, 
                           stride = (1,1), 
                           padding='same', 
                           bias=False)
        #self.inConv.double()
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
        
        ###REDUCING LAYERS###:
        #r1
        self.r1Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,1), 
                                       padding=(2,2), 
                                       bias=False)
        self.r1ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r1ConvActivation = nn.SiLU()
        # torch.Size([4, 12, 850, 500])
       
        #r2
        self.r2Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r2ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r2ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 425, 250])
        
        #r3
        self.r3Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r3ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r3ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 213, 125])

        #r4
        self.r4Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r4ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r4ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 107, 63])
        
        #r5
        self.r5Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r5ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r5ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 54, 32])
        
        #r6
        self.r6Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,1), 
                                       padding=(4,2), 
                                       bias=False)
        self.r6ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r6ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 29, 32])
        
        #r7
        self.r7Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(3,2), 
                                       bias=False)
        self.r7ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r7ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 16, 16])
        
        #r8
        self.r8Conv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (2,2), 
                                       padding=(2,2), 
                                       bias=False)
        self.r8ConvBatch = nn.BatchNorm2d(num_features = self.Nfilters, momentum = self.BNmomentum, track_running_stats = False, affine = True)
        self.r8ConvActivation = nn.SiLU()
        #torch.Size([4, 12, 8, 8])
        
        #endConv 
        #No Batch Norm on last Conv Layer, I made this non-reducing. Correct choice?
        self.endConv = nn.Conv2d(in_channels = self.Nfilters, 
                                       out_channels= self.Nfilters, 
                                       kernel_size = self.kernelSize, 
                                       stride = (1,1), 
                                       padding=(2,2), 
                                       bias=False)
        self.endConvActivation = nn.SiLU()
        ##### OPTION grab the size of the image, and pass through a kernel of the same size with valid padding
        ##### That is another way to make a dsnse layer. But it reduces the size to 12X1X1
        #torch.Size([4, 12, 8, 8])
        
        self.flattenLayer = nn.Flatten()
        #torch.Size([4, 768])
        
        ###HIDDEN LAYERS###
        # First Dense Layer
        self.h1 = nn.Linear(768, outputHiddenLayerSize)
        self.h1Activation = nn.SiLU()
        
        #Linear Output Layer
        self.linOut = nn.Linear(outputHiddenLayerSize,NOutput)
        
        
        
        
    
    def forward(self,x):
        #print("Input " + str(x.shape))
        
        ###Run Input Layer Here###
        x = self.inConv(x)
        x = self.inConvBatch(x)
        x = self.inConvActivation(x)
        #print("After Input Layer " + str(x.shape))
        
        ###RUN INTERPRETABILITY LAYERS HERE###
        for i in range(0,self.Ninterp):
            x = self.interpLayers[i](x)
            x = self.interpBatchNorms[i](x)
            x = self.interpActivations[i](x)
        #print("After Interpretability Layers: " + str(x.shape))
        
        ###RUN REDUCING LAYERS HERE###
        #r1
        x = self.r1Conv(x)
        x = self.r1ConvBatch(x)
        x = self.r1ConvActivation(x)
        #print(x.shape)
        #r2
        x = self.r2Conv(x)
        x = self.r2ConvBatch(x)
        x = self.r2ConvActivation(x)
        #print(x.shape)
        #r3
        x = self.r3Conv(x)
        x = self.r3ConvBatch(x)
        x = self.r3ConvActivation(x)
        #print(x.shape)
        #r4
        x = self.r4Conv(x)
        x = self.r4ConvBatch(x)
        x = self.r4ConvActivation(x)
        #print(x.shape)
        #r5
        x = self.r5Conv(x)
        x = self.r5ConvBatch(x)
        x = self.r5ConvActivation(x)
        #print(x.shape)
        #r6
        x = self.r6Conv(x)
        x = self.r6ConvBatch(x)
        x = self.r6ConvActivation(x)
        #print(x.shape)
        #r7
        x = self.r7Conv(x)
        x = self.r7ConvBatch(x)
        x = self.r7ConvActivation(x)
        #print(x.shape)
        #r8
        x = self.r8Conv(x)
        x = self.r8ConvBatch(x)
        x = self.r8ConvActivation(x)
        #print(x.shape)
        #endConv No Batch Norm on last Conv Layer, I made this non-reducing. Correct choice?
        x = self.endConv(x)
        x = self.endConvActivation(x)
        #print(x.shape)
        #Flaten
        x = self.flattenLayer(x)
        #print(x.shape)
        
        ###Run HIDDEN LAYER###
        x = self.h1(x)
        x = self.h1Activation(x)
        #print("after hidden layer " + str(x.shape))
        
        ###Linear Output Layer With No Activiation###
        x = self.linOut(x)
        #print("after end linear output layer" + str(x.shape))
        x = x.flatten()
        
        return(x)