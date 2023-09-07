############################################
Feature Sensitivity
############################################

.. collapse:: Collapse Tensorflow Coupon Figure
  :open:

  |

  .. image:: ./figures/tf_coupon/activation_15_all_features_std_idx00110.png
   :alt: Standard Deviation of Features Extracted from the Tensorflow Coupon Network on idx001100 Data
   :align: center

  .. centered:: *Standard Deviation of Features Extracted from the Tensorflow Coupon Network on idx001100 Data*

.. collapse:: Collapse Pytorch Nested Cylinder Figure
  :open:

  |

  .. image:: ./figures/pyt_nestedcyl/interpActivations14_feature12_std_idx00130.png
   :alt: Standard Deviation of Features Extracted from the Pytorch Nested Cylinder Network on idx001300 Data
   :align: center
   :scale: 80%

  .. centered:: *Standard Deviation of Feature 12 Extracted from the Pytorch Nested Cylinder Network on idx001300 Data*

|

Code Documentation
===================

.. automodule:: feature_sensitivity
    :members:

Arguments
===============

.. argparse::
   :module: feature_sensitivity
   :func: feature_sensitivity_parser
   :prog: python feature_sensitivity.py