############################################
Documentation
############################################

Packages
========================================

The documentation for the feature interpretability module is made with `sphinx <https://www.sphinx-doc.org/en/master/>`_. 

Below is a list of `sphinx extensions <https://www.sphinx-doc.org/en/master/usage/extensions/index.html>`_ that are required to build the docs:

 - `sphinx.ext.autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_
 - `sphinx.ext.coverage <https://www.sphinx-doc.org/en/master/usage/extensions/coverage.html>`_
 - `sphinx.ext.napoleon <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_
 - `sphinx.ext.githubpages <https://www.sphinx-doc.org/en/master/usage/extensions/githubpages.html>`_

 Additionally, the `sphinx-toolbox <https://sphinx-toolbox.readthedocs.io/en/stable/index.html>`_ and `sphinx-argparse <https://sphinx-argparse.readthedocs.io/en/stable/install.html>`_ packages are installed to provide additional functionality. They are included in the extensions list as follows:

  - `sphinx_toolbox.collapse <https://sphinx-toolbox.readthedocs.io/en/stable/extensions/collapse.html>`_
  - sphinxarg.ext

These are all included in the ``./docs/source/conf.py`` file.

Generating Docs
========================================

All of the files to make the sphinx documentation is found in ``./docs/``. Source code is found in ``./docs/source/``, while built ``.html`` files are found in ``./docs/build/``.

To generate the most up-to-date documetaion, run the following commands in the ``./docs/`` directory:

 - ``make clean`` to remove existing build docs
 - ``make html`` to create the new docs
 - open ``./docs/build/html/index.html`` to view rendered docs

Rendered documentation is also `hosted on GitHub <https://lanl.github.io/feature_interpretability/html/index.html>`_.