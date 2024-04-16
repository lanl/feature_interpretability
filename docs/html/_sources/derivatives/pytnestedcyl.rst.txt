#################################################
Derivatives on Pytorch Nested Cylinder Models
#################################################

Feature dervivatives have been implemented on the :doc:`/exp/pytnestedcyl/`.

.. contents:: Table of Contents:
	:local:
	:depth: 2 

Feature Derivatives Script
==============================================

.. automodule:: ftderivatives_pyt_nestedcyl
    :members:

Arguments
--------------

.. collapse:: Uncollapse Arguments

    .. argparse::
       :module: ftderivatives_pyt_nestedcyl
       :func: feature_derivatives_parser
       :prog: python ftderivatives_pyt_nestedcyl

|

Calico Model Diagram
==============================

 .. image:: pytnestedcyl_calicomodel.png
   :scale: 100 %
   :alt: tensorflow coupon model calico network diagram
   :align: center


Calico Model Creation
==============================================

.. automodule:: fns.derivatives.pyt_nestedcyl_calico_model
    :members:

Unit Tests
--------------

- **Unit Test for the Difference Output**: The Calico difference output is the difference between the original branch and the multiply branch. This test sets the dScale value to zero, meaning the multiply branch is scaled by 1. Therefore, the difference should be zero.
- **Unit Test for the Prediction Output**: The Calico prediction output is the output from the origial branch. This test compares the Calico prediction output to the original model prediction output. The difference should be zero.

Arguments
--------------

.. collapse:: Uncollapse Arguments

    .. argparse::
       :module: fns.derivatives.pyt_nestedcyl_calico_model
       :func: calico_model_parser
       :prog: python pyt_nestedcyl_calico_model

|


Calico Dataset & Dataloader Creation
==================================================

.. automodule:: fns.derivatives.pyt_nestedcyl_calico_dataloader
    :members:

Unit Tests
--------------

- **Unit Test of Length Method**: The unit tests print the length of the dataset to confirm that is is the same length as the number of samples provided.
- **Unit Test for Input and Output Shapes**: The unit tests print the shapes of the batched input and ground truth. The user must check that these sizes are correct. Batch size 8 is used.

Arguments
--------------

.. collapse:: Uncollapse Arguments

    .. argparse::
       :module: fns.derivatives.pyt_nestedcyl_calico_dataloader
       :func: calico_dataloader_parser
       :prog: python pyt_nestedcyl_calico_dataloader

|
