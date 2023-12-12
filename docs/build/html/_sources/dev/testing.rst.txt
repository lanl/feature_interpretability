############################################
Testing with Pytest
############################################

Testing is implemented on the :doc:`/fns/outer` with `pytest <https://docs.pytest.org/en/7.4.x/>`_. Testing is not completed on any functions that require either neural network files or input data files. Hence, testing was not completed on any other submodules of the ``fns`` module.

.. contents:: Table of Contents:
    :local:
    :depth: 2

Running Tests
========================================

 .. image:: pytest_all.png
   :alt: pytest terminal printout for all tests
   :align: center

Test files are in ./tests. Tests are run from the outermost layer of the feature_interpretability directory, while in an enviroment with ``pytest`` installed.

 - Run ``pytest`` to run all tests.
 - Run ``pytest --cov`` to run all tests and generate a coverage report on ``fns``.
 - Run ``pytest --cov=fns.submodule`` to run tests and generate a coverage report on a specific submodule of ``fns``.

Testing on fns.mat
========================================

 .. image:: pytest_fns_mat.png
   :alt: pytest terminal printout for fns.mat tests
   :align: center

.. automodule:: tests.test_mat
    :members:

Testing on fns.misc
========================================

 .. image:: pytest_fns_misc.png
   :alt: pytest terminal printout for fns.misc tests
   :align: center

.. automodule:: tests.test_misc
    :members:

Testing on fns.plot
========================================

 .. image:: pytest_fns_plot.png
   :alt: pytest terminal printout for fns.plot tests
   :align: center

.. automodule:: tests.test_plots
    :members:

Testing on fns.save
========================================

 .. image:: pytest_fns_save.png
   :alt: pytest terminal printout for fns.misc tests
   :align: center

.. automodule:: tests.test_save
    :members: