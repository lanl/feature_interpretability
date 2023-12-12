############################################
Enviroments
############################################

These interpretability tools are functioanl with both the `tensorflow <https://www.tensorflow.org/api_docs/python/tf>`_ ML package and the `pytorch <https://pytorch.org/docs/stable/index.html>`_ ML package. These two packages don't tend to play nice with each other, so typically they must be kept in seperate enviorments. The user is responsible for activating the correct enviroments for the package they want to run.

The ``enviorments`` directory contains ``.yml`` files describing enviroments that were used to create and test the code library. However, it is reccomended that users create thier own enviroments as necessary. 

Here is a list of non-machinelearning packages used by the scripts. These should be installed regardless of which ML package is being used. This list may not contain all necessary packages; defer to what is imported in the scripts.

 - os
 - sys
 - glob
 - argparse
 - numpy
 - pandas
 - matplotlib
 - pingouin
 - scikit-image (skimage)