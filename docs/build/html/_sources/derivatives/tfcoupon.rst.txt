############################################
Derivatives on Tensorflow Coupon Models
############################################

Feature dervivatives have been implemented on the :doc:`/exp/tfcoupon/`.

.. contents:: Table of Contents:
	:local:
	:depth: 2 

Feature Derivatives Script
==============================================

.. automodule:: ftderivatives_tf_coupon
    :members:

Arguments
--------------

.. collapse:: Uncollapse Arguments

    .. argparse::
       :module: ftderivatives_tf_coupon
       :func: feature_derivatives_parser
       :prog: python ftderivatives_tf_coupon

|

Calico Model Diagram
==============================

 .. image:: tfcoupon_calicomodel.jpg
   :scale: 100 %
   :alt: tensorflow coupon model calico network diagram
   :align: center


Calico Model Creation
==============================================

.. automodule:: fns.derivatives.tf_coupon_calico_model
    :members:

Unit Tests
--------------

- **Unit Test for the Difference Output**: The Calico difference output is the difference between the original branch adn the multiple branch. This test sets the tensor multiplier to one. Therefore, the difference should be zero.
- **Unit Test for the Prediction Output**: The Calico prediction output is the output from the origial branch. This test compares the Calico prediction output to the original model prediction output. The difference should be zero.
- **Unit Test for the Truth Output**: The Calico truth output is the the same as the ground truth passed into Calico. This test compares the truth inputed to Calcio with the truth output. The difference should be zero.
    + It is commonly observed that this difference is :math:`O(10^{-8})`. The developers hypothesize that this is due to the conversion between float64 and float32 that occurs when loading the inputs onto and off of GPUs.]

Arguments
--------------

.. collapse:: Uncollapse Arguments

    .. argparse::
       :module: fns.derivatives.tf_coupon_calico_model
       :func: calico_model_parser
       :prog: python tf_coupon_calico_model

|


Calico Sequence Creation
==================================================

.. automodule:: fns.derivatives.tf_coupon_calico_seq
    :members:

Unit Tests
--------------

- **Unit Test of Input and Output Shapes**: The calcio sequence creates four inputs to the calico network: \[img_input, const_input, prms_input, truth_input]; and one output: truth_output. The unit tests print the shapes of all the inputs and outputs. The user must determine if the shapes are correct
- **Unit Test for Const_Input Construction**: The unit tests check if the const_input contians all ones in unselected features and contains 1+dScale for the selected feature.
- **Unit Test for the Truth Output**: The unit tests check if the truth input and truth output are identical.

Arguments
--------------

.. collapse:: Uncollapse Arguments

    .. argparse::
       :module: fns.derivatives.tf_coupon_calico_seq
       :func: calico_seq_parser
       :prog: python tf_coupon_calico_seq

|
