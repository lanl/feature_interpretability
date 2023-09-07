#################################################
Pytorch Nested Cylinder Models
#################################################

Straightfoward CNNs were developed in pytorch to estimate the scaling of a Preston-Tonks-Wallace (PTW) strength model from a nested cylinder experiment.

.. contents:: Table of Contents:
  :local:
  :depth: 2 

Model Inputs & Outputs
=============================

Networks trained on nested cylinder data take as input an image showing the density of the Material Of Interest (MOI) in the nested cylinder experiment. An example MOI density is shown below.

 .. image:: nestedcyl_MOI_input.png
   :scale: 100 %
   :alt: nested cylinder experiment density of material of interest
   :align: center

These models predict the value that the PTW strength model was scaled by.

Running the Nested Cylinder Example
=======================================

To run the scripts on the pytorch nested cylinder networks, use the following arguments: 

 - ``--PACKAGE pytorch``
 - ``--EXPERIMENT nestedcylinder``
 - ``--MODEL ../examples/pyt_nestedcyl/trained_hrMOICyl2sclPTW_model.pth``
 - ``--INPUT_FIELD hr_MOICyl``
 - ``--INPUT_NPZ`` use any .npz file in ``../examples/pyt_nestedcyl/data/``
 - ``--INPUT_DIR ../examples/pyt_nestedcyl/data/``
 - ``--DESING_FILE ../examples/pyt_nestedcyl/nestedcyl_design_file.csv``
 - ``--FIXED_KEY idx00130``, or ``--FIXED_KEY None``  

Nested Cylinder Data
=========================

The file names of nested cylinder data contain multiple pieces of information about their contents. The two relevent componets are the **sclPTW** and the **idx**:

- The *sclPTW_* is followed by a number identifying the experiment ID. This ID corresponds with a scale value for the PTW strength model. This scale value is the only prediction from the nestedcylinder networks. 
- The *idx* specifies what time step the simulation was at. Nested cylinder networks are only trained on samples from idx00130.

Each ``.npz`` nested cylinder data file contains the following fields:

 - *sim_time* (scalar): simulation time stamp corresponding to a unique *idx* value
 - *rho* (2D array): density of entire experiment
 - *hr_wallBottom1*, *hr_wallBottom2*, *hr_wallBottom3* (2D arrays): density of a bottom wall component
 - *hr_wallCorner*, *hr_wallRight1*, *hr_wallRight2*, *hr_wallRight3* (2D arrays): density of a wall component
 - *hr_mainchargeBottom1*, *hr_mainchargeBottom2*, *hr_mainchargeCorner*, *hr_mainchargeRight1*, *hr_mainchargeRight2* (2D arrays): density of a main charge component
 - *hr_innerCylBottom*, *hr_innerCylCorner*, *hr_innerCylRight* (2D arrays): density of an inner cylinder component
 - *hr_MOICyl* (2D array): density of the material of interest training field (use ``-IN_FIELD hr_MOICyl``)
 - *pressure* (2D array)
 - *temperature* (2D array)
 - *melt_state* (2D array): binary array of if a cell has melted
 - *porosity* (2D array)
 - *eqps* (2D array): equivalent plastic stress
 - *eqps_rate* (2D array): equivalent plastic stress rate
 - *eff_stress* (2D array): effective stress
 - *bulk_mod* (2D array): bulk modulus of the material
 - *sound_speed* (2D array): speed of sound in the material
 - *rVel* (2D array): veliocty of material in the R-axis direction
 - *zVel* (2D array): velocity of material in the Z-axis direction
 - *Rcoord* (1D vector): vector containing position in cm of all cells along the R-axis
 - *Zcoord* (1D vector): vector containing position in cm of all cells along the Z-axis

Model Architecture
=============================

These models consist of a single branch that passes the image input straight forward through the network. 

 .. image:: pytnestedcyl_networkdiagram.png
   :scale: 100 %
   :alt: tensorflow coupon branched nerual network diagram
   :align: center

Model Layers
=============================

The layers in model follow the following naming convention:

- **in???**: layer near top of the network
- **??Conv**: 2D convolutional layer
- **??ConvBatch**: 2D batch normalization layer
- **??ConvActivation**: SiLU activation layer
- **interp???.##**: layer in "interpretability stack"
- **interpLayer.##**: 2D convolutional layer in "interpretability stack"
- **interpBatchNorms.##**: 2D batch normalization layer in "interpretability stack"
- **interpActivations.##**: SiLU activation layer in "interpretability stack"
- **r#???**: layer that reduces internal layer size by using a stride ≠ (1,1)
- **end???**: layer near the end of the layer
- **h#**: linear hidden layer
- **h#Activation**: SiLU activation layer after a hidden layer
- **linOut**: linear layer that generates output
- **flattenLayer**: ``torch.nn.Flatten()`` layer