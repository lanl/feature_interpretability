#################################################
Pytorch Nested Cylinder Models
#################################################

Straightfoward CNNs were developed in pytorch to estimate the scaling of a Preston-Tonks-Wallace (PTW) strength model from a nested cylinder experiment.

.. contents:: Table of Contents:
  :local:
  :depth: 2 

Model Inputs & Outputs
=============================

Networks trained on nested cylinder data take as input an image showing the density throughout the experiment. An example density field is shown below.

 .. image:: nestedcyl_density_input.png
   :scale: 100 %
   :alt: nested cylinder experiment density of material of interest
   :align: center

These models predict the value that the PTW strength model was scaled by.

Running the Nested Cylinder Example
=======================================

To run the scripts on the pytorch nested cylinder networks, use the following arguments: 

 - ``--PACKAGE pytorch``
 - ``--EXPERIMENT nestedcylinder``
 - ``--MODEL ../examples/pyt_nestedcyl/trained_rho2PTW_model.pth``
 - ``--INPUT_FIELD rho``
 - ``--INPUT_NPZ`` use any .npz file in ``../examples/pyt_nestedcyl/data/``
 - ``--INPUT_DIR ../examples/pyt_nestedcyl/data/``
 - ``--DESING_FILE ../examples/pyt_nestedcyl/nestedcyl_design_file.csv``
 - ``--FIXED_KEY id0643``, ``--FIXED_KEY idx00130``, or ``--FIXED_KEY None``  

Nested Cylinder Data
=========================

The file names of nested cylinder data contain multiple pieces of information about their contents. The two relevent componets are the **id** and the **idx**:

- The *id* is followed by a number identifying the experiment ID. 
   - The value of ``PTW_scale`` is identical across files with identical *id* 's.
   - Data is included for *id0643* at all 22 timesteps.
- The *idx* specifies what time step the simulation was at. 
   - The value of the ``sim_time`` will be identical across files with identical *idx* 's.
   - Data is included at *idx00112* for 31 differnt simulations.

Each ``.npz`` nested cylinder data file contains the following fields:

 - *sim_time* (scalar): simulation time stamp corresponding to a unique *idx* value
 - *rho* (2D array): density training field (use ``-IN_FIELD rho``)
 - *hr_outerWall*, *hr_bottomWall*, *hr_mcSide*, *hr_mcBottom*, *hr_innerCylClide*, *hr_innerCylBottom*, *hr_MOI (2D arrays): density of simulation compoment
 - *hr_innerCylBottom*, *hr_innerCylCorner*, *hr_innerCylRight* (2D arrays): density of an inner cylinder component
 - *volhr_outerWall*, *volhr_bottomWall*, *volhr_mcSide*, *volhr_mcBottom*, *volhr_innerCylSide*, *volhr_innerCylBottom*, *volhr_MOI* (shape): thing
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
- **??Conv??**: 2D convolutional layer
- **??Norm??**: 2D batch normalization layer
- **??Activation**: GELU activation layer
- **interp_module.??.##**: layer in "interpretability stack"
- **reduction_module.??.##**: layer in the "reduction stack", which reduced layer size by using a stride ≠ (1,1)
- **end???**: layer near the end of the model
- **hidden**: linear hidden layer
- **hidden#Activation**: GELU activation layer after a hidden layer
- **linOut**: linear layer that generates output
- **flattenLayer_##**: ``torch.nn.Flatten()`` layer