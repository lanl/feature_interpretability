############################################
Documentation
############################################

The documentation for the feature interpretability module is made with `sphinx <https://www.sphinx-doc.org/en/master/>`_. All of the files to make the sphinx documentation is found in ``./docs/``. Source code is found in ``./docs/source/``, while built ``.html`` files are found in ``./docs/build/``.

To generate the most up-to-date documetaion, run the following commands in the ``./docs/`` directory:

 - ``make clean`` to remove existing build docs
 - ``make html`` to create the new docs
 - open ``./docs/build/html/index.html`` to view rendered docs

Rendered documentation is `hosted on GitHub <https://lanl.github.io/feature_interpretability/html/index.html>`_.