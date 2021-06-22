|pre-commit| |black|

SQUIDLE Downloader
==================

Download benthic image data from SQUIDLE_.

.. _SQUIDLE: https://squidle.org/


Developmental setup
-------------------

You can download the repository from GitHub with::

    git clone git@github.com:DalhousieAI/squidle-downloader.git

Install it in editable mode with::

    pip install -e .

This repository uses the `black code style <https://black.readthedocs.io/>`__, and `numpy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`__ (`example <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`__).
Your code should also be `flake8 compliant <https://flake8.pycqa.org/en/latest/>`__.

So that you don't have to manually modify code to fit the style guide, we have a git hook using `pre-commit <https://pre-commit.com/>`__.
This will execute every time you make a commit, and will blacken your code before it is committed.
It will also check for common code errors.

To set up the pre-commit hook, run the following code::

    pip install -r requirements-dev.txt
    pre-commit install

Whenever you try to commit code which needs to be modified by the commit hook, you'll have to add the commit hooks changes and then redo your commit.

You can also manually run the pre-commit stack on all the files at any time::

    pre-commit run --all-files


.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: black
