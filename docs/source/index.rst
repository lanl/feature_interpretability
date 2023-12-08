############################################
Feature Interpretability Documentation
############################################

This folder of code contains tools for extracting, plotting, and computing with features extracted from nerual networks. 

These tools were developed by **Skylar Callis**. They developed this code while working as a post-bachelors student at `Los Alamos National Lab <https://www.lanl.gov/?source=globalheader>`_ from 2022 - 2024. To see what they are up to these days, visit `Skylar's Website <https://skylar-jean.com>`_.

Module Use
==========================

The scripts in the feature interpretability module are command-line executable scripts that run some interperability tool on a given model. All scripts take `argparse <https://docs.python.org/3/library/argparse.html#module-argparse>`_ command-line inputs to specify the model and accessory files. 

All scripts take a ``--PACKAGE`` argument that specifies if the model was built in ``tensorflow`` or ``pytorch``. These two packages don't tend to play nice with each other, so typically they must be kept in seperate enviorments. The user is responsible for activating the correct enviroments for the package they want to run.

All scripts also take an ``--EXPERIMENT`` argument that specifies what dataset the model was trained on. This data process has been established for two experients at LANL: the ``coupon`` experiment and the ``nestedcylinder`` experiment. If the user wants to analyze networks trained on other datasets, they would first have to develop a submodule to process date from that experiment, and then integrate that code into existing scripts.

.. attention::
   
   Nested cylinder examples are not currently included in the open source edition of this code. As such, when a script is passed ``--EXPERIMENT nestedcylinder``, it raises a ``NotImplementedError``. Nested cylinder examples are expected to be added in February 2024.

While these tools are intended to be easily maluable to new networks and new datasets, they are not guarenteed to work outside of the tested use cases.

Network Nomenclature
==========================

Unfortunatly, no one agrees on what the inside of neural networks should be called. To prevent Skylar from going insane, they developed the following Neural Network Nomenclature:

- A **Model** is a trained neural network.
- A **Layer** is a single level of a *model*. Common types of layers include *convolutions*, *batch normalizations*, and *activations*.
- A **Feature** is the intermediate output of the network at a single *layer*. Most layers will output multiple features. When not stated, a *feaure* is implied to be extracted from some specified *layer*.
- A **Field** refers to either a radiograph or a hydrodynamic field related to the input for a given model; some *fields* are used as training data for *models*, while others remain unseen by the *models*.
- An **Prediction** is the output from a *model*. 
- The **Ground Truth** or **Truth** refers to the true value of whatever quantity the *model* is *predicting*. The *model* attempts to estimate the *ground truth* with its *predictions*.

Please note that **Activation** does NOT have a specific meaning. Commonly in ML spaces, *activation* is used to refer to either a **Layer** or a **Feature**. Skylar decided the best path forward ignored *activation* as a term entirely to prevent confusion.

..
   hidden toctrees to make the sidebar render

.. toctree::
   :maxdepth: 1
   :caption: Tested Use Cases
   :hidden:

   exp/tfcoupon
   exp/pytnestedcyl

.. toctree::
   :maxdepth: 1
   :caption: Scripts
   :hidden:

   scripts/plot_features
   scripts/feature_over_fields
   scripts/feature_sensitivity
   scripts/field_sensitivity
   scripts/field_autocorr
   scripts/feature_field_corr
   scripts/feature_fieldstd_corr
   scripts/feature_pred_corr

.. toctree::
   :maxdepth: 2
   :caption: Feature Derivatives
   :hidden:

   derivatives/landing
   derivatives/tfcoupon
   derivatives/pytnestedcyl

.. toctree::
   :maxdepth: 2
   :caption: Helper Functions
   :hidden:

   fns/outer
   fns/setup
   fns/coupondata
   fns/nestedcylinderdata
   fns/tfcustom
   fns/pytorchcustom

.. toctree::
   :maxdepth: 2
   :caption: Developer Tools
   :hidden:

   dev/enviroments
   dev/testing
   dev/test_scripts
   dev/docs
