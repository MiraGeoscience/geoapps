|coverage| |precommit_ci| |docs| |style| |version| |pyversions|


.. |docs| image:: https://readthedocs.org/projects/geoapps/badge/
    :alt: Documentation Status
    :target: https://geoapps.readthedocs.io/en/latest/?badge=latest

.. |coverage| image:: https://codecov.io/gh/MiraGeoscience/geoapps/branch/main/graph/badge.svg
    :alt: Code coverage
    :target: https://codecov.io/gh/MiraGeoscience/geoapps

.. |style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Coding style
    :target: https://github.com/pf/black

.. |version| image:: https://img.shields.io/pypi/v/geoapps.svg
    :alt: version on PyPI
    :target: https://pypi.python.org/pypi/geoapps/

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/geoapps.svg
    :alt: Python versions
    :target: https://pypi.python.org/pypi/geoapps/

.. |precommit_ci| image:: https://results.pre-commit.ci/badge/github/MiraGeoscience/geoapps/main.svg
    :target: https://results.pre-commit.ci/latest/github/MiraGeoscience/geoapps/main
    :alt: pre-commit.ci status


**Geoapps** - Open-source applications in geosciences
=====================================================

.. image:: https://github.com/MiraGeoscience/geoapps/raw/v0.10.0-rc.2/docs/images/index_page.png
    :align: right
    :width: 50%


The **geoapps** project has been created by `Mira Geoscience`_ for the development and sharing of open-source
applications in geoscience. Users will be able to directly leverage the powerful visualization capabilities of
`Geoscience ANALYST`_ along with open-source code from the Python ecosystem.

.. _Mira Geoscience: https://mirageoscience.com/
.. _Geoscience ANALYST: https://mirageoscience.com/mining-industry-software/geoscience-analyst/


Documentation
^^^^^^^^^^^^^

- `Online documentation <https://mirageoscience-geoapps.readthedocs-hosted.com>`_


Current sponsors
^^^^^^^^^^^^^^^^

- `Anglo American <http://www.angloamerican.ca/>`_
- `Barrick <https://www.barrick.com/English/home/default.aspx>`_
- `BHP <https://www.bhp.com/>`_
- `Cameco <https://www.cameco.com/>`_
- `Glencore <https://www.glencore.com/>`_
- `Mira Geoscience`_
- `Rio Tinto <https://www.riotinto.com/en>`_
- `Teck <https://www.teck.com/>`_
- `Vale <https://www.vale.com/canada>`_


Installation
^^^^^^^^^^^^
**geoapps** is currently written for Python 3.10.

Install Conda
-------------

To install **geoapps**, you need to install **Conda** first.

We recommend to install **Conda** using `miniforge`_.

.. _miniforge: https://github.com/conda-forge/miniforge

Quick installation
-------------------

To install (or re-install) a conda environment to run **geoapps**, simply execute the **install.bat** file.

To install in editable mode, so that changes in the source code are immediately reflected in the
running application, execute with the ``-e`` option: ``install.bat -e``.

.. warning::

    In editable mode, the source folder must not be moved or deleted after installation.


Manual installation
-------------------

You should not install the package directly with ``pip``, as the app requires conda packages to run.

First create a Conda environment with all the required dependencies,
then activate it and install the package in this environment using
``pip install --no-deps ...``

See instructions below for more details and options.

Prepare a Conda environment with dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can create a Conda environment with all the required dependencies ``conda`` and the ``.lock``
files from a conda prompt::

    $ conda env create --solver libmamba -n geoapps -f environments/[the_desired_env].lock.yml

.. note::

    The package itself is not install yet in the Conda environment. See following instructions.

.. warning::

    All the following ``pip`` commands are meant to be executed in the Conda environment you just created.
    Activate it with::

    $ conda activate geoapps

From PyPI
~~~~~~~~~

To install the **geoapps** package published on PyPI::

    $ pip install --no-deps -U geoapps

From a Git tag or branch
~~~~~~~~~~~~~~~~~~~~~~~~
If the revision of the package is not on PyPI yet, you can install it from a Git tag::

    $ pip install --no-deps -U --force-reinstall https://github.com/MiraGeoscience/geoapps/archive/refs/tags/TAG.zip

Or to install the latest changes available on a given Git branch::

    $ pip install --no-deps -U --force-reinstall https://github.com/MiraGeoscience/geoapps/archive/refs/heads/BRANCH.zip

.. note::

    The ``--force-reinstall`` option is used to make sure the updated version
    of the sources is installed, and not the cached version, even if the version number
    did not change.

    The ``-U`` or ``--upgrade`` option is used to make sure to get the latest version,
    on not merely reinstall the same version.

    The option ``--no-deps`` is used to avoid installing the dependencies with pip,
    as they have dependencies are already installed within the **Conda environment**.

From a local copy of the sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have a git clone of the package sources locally, you can install **geoapps** from the
local copy of the sources. At the root of the sources, you will find a ``pyproject.toml`` file.

Change directory to the root of the sources::

    $ cd path/to/project_folder_with_pyproject_toml

Then run::

    $ pip install --no-deps -U --force-reinstall .

Or in **editable mode**, so that you can edit the sources and see the effect immediately at runtime::

    $ pip install --no-deps -U --force-reinstall -e .

Setup for development
^^^^^^^^^^^^^^^^^^^^^
To configure the development environment and tools, please see `README-dev.rst`_.

.. _README-dev.rst: README-dev.rst


License
^^^^^^^
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Third Party Software
^^^^^^^^^^^^^^^^^^^^
The geoapps Software may provide links to third party libraries or code (collectively “Third Party Software”)
to implement various functions. Third Party Software does not comprise part of the Software.
The use of Third Party Software is governed by the terms of such software license(s).
Third Party Software notices and/or additional terms and conditions are located in the
`THIRD_PARTY_SOFTWARE.rst`_ file.

.. _THIRD_PARTY_SOFTWARE.rst: docs/THIRD_PARTY_SOFTWARE.rst

Copyright
^^^^^^^^^
Copyright (c) 2020-2025 Mira Geoscience Ltd.
