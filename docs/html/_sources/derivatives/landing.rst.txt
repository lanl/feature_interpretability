############################################
Introduction
############################################

Feature derivatives are an interpretability method that examines how a small change to an internal feature value impacts the prediction of a network. In order to calculate feature derivatives of a given model, a new model is constructed from an original model with minor internal changes. These new models are called *Calico networks*, as they are *copycats* of the orignal model. 

.. contents:: Table of Contents:
	:local:
	:depth: 2 

Definition of Feature Derivatives
=====================================

Feature derivatives are defined as the change in prediction relative to the change to an internal feature. The change made to the internal feature is a multiplication by :math:`1 + dScale`, where *dScale* is a small value selected by the user.

Specifically, the feature derivative is calculated as the difference between the orignal model prediction and the prediction of the model with the internal feature scaling, divided by the *dScale* value.

Feature derivatives are dependent on the input to the model, as well as the selected *layer* and *feature* to adjust by the *dScale* value.

Calico Network Development
==============================

Calico networks must be individually impelemented for each combination of model architecture and experiemnt. Note that model architecture here does not refer to the package used for a model, but the specific layout of model layers. 

Each calico model implemented also requires a custom data generator. For models implemented in tensorflow, this iterator is a ``tensorflow.keras.utils.Sequence`` class. For models implemented in pytorch, this iterator is a ``torch.utils.data.Dataset`` class wrapped in a ``torch.utils.data.DataLoader`` iterable.

Calico models have been created for:

 - branched models trained to estimate damage model calibration for the coupon experiment, implemented in tensorflow (see :doc:`/derivatives/tfcoupon`)
 - single branch models trained to estimate the strength scaling of nested cylinder experiments, implemented in pytorch (see :doc:`/derivatives/pytnestedcyl`)


Calico Model Components
==============================

The documentation for each specific calico model gives more information on their specific inputs, outputs, and internal structure. However, all calico networks have the following components:

 - **Original Branch**: a branch of the calico network that is identical to the original model
 - **Split Layer**: layer from the original model that acts as the splitting point between the *original branch* and the *multiply branch*; the feature to be scaled is in the *split layer*
 - **Multiply Branch**: a branch of the calico network that diverges at the split layer; this branch multiples a feature of the split layer by :math:`1 + dScale`, and then passes through layers identical to the *original branch*

Calico Model Outputs
==============================

The scripts that calculate feature derivatives save a ``pandas.DataFrame`` to a ``.csv`` file. This dataframe contains rows for each input sample and columns for each output. Some outputs may have multiple columns if the model is making multiple predictions. The outputs are:

 - **Prediction Output**: model predictions from the *original branch* that are identical to the prediction from the original model
 - **Difference Output**: the difference between the *prediction output* and the output of the *multiply branch*
 - **Derivative Output**: the derivative of a prediction, calculated as the *difference output* divided by the *dScale* value
 - **Truth Output**: the ground truth value for the sample
 - **Sample Information**: the identifying sample information

