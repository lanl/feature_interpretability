############################################
*Test Scripts* Script
############################################

The *test scripts* script runs all of the other scripts, including the feature derivative scripts, to check for bugs. This is not a full coverage test, and should not be taken as such. Instead, it is intended to catch simple bugs introduced in the fns module. It can also be used to check that the python enviroment has all of the required packages to run the feature interpretability module.

This does not save any individual script output. It does print the time it took to run to the command line.

.. attention::
   
   Nested cylinder examples are not currently included in the open source edition of this code. As such, when a script is passed ``--EXPERIMENT nestedcylinder``, it raises a ``NotImplementedError``. Nested cylinder examples are expected to be added in February 2024.

Code Documentation
===================

.. automodule:: test_scripts
   :members:


Arguments
===============

.. argparse::
   :module: test_scripts
   :func: test_run_parser
   :prog: python test_scripts.py