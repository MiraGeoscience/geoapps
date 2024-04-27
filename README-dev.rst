
Setup for development
=====================

.. contents:: Table of Contents
   :local:
   :depth: 2

After you have cloned the Git repository, you will need to:
    #. create the Conda environment lock files for the dependencies
    #. create a virtual Conda environment for development, where to install the dependencies
       of  the project
    #. execute the tests
    #. setup Git LFS if needed
    #. configure the pre-commit hooks for static code analysis and auto-formatting
    #. configure the Python IDE (PyCharm)


Create the Conda environment lock files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, you need to create the Conda environment lock files (``*.conda.lock.yml``) for the dependencies defined
in `pyproject.toml`_.

.. note::
    As a prerequisite, you need to install some packages in your base Conda environment. To do so,
    simply execute ``devtools\setup-conda-base.bat``.

Then, to create the Conda environment lock files, execute ``devtools\run_conda_lock.bat``,
or run from command line::

    $ (base) python devtools/run_conda_lock.py

It will create or update ``.conda.lock.yml`` files in the ``environments`` folder:
one for runtime dependencies, and one for development dependencies (with the ``-dev`` suffix),
for each combinations of Python versions and platforms.

The platforms are specified in ``conda-lock`` section of the ``pyproject.toml`` file:

.. code-block:: toml

    [tool.conda-lock]
    platforms = ['win-64', 'linux-64']

The python versions are specified at the beginning of the ``devtools/run_conda_lock.py`` file:

.. code-block:: python

    _PYTHON_VERSIONS = ["3.10", "3.11"]

The ``Install_or_Update.bat`` and the ``setup-dev.bat`` will use them to install the environment.


Install the Conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For development, you need a **Conda** environments. you can install it running the ``setup-dev.bat`` or::

    $ [path\to\geoapps]\devtools\setup-dev.bat

This command install a local environment at the base of your repository: ``.conda-env``.
This environment should automatically be recognized by the Conda installation.

To activate this environment, run the following command from the root of the project::

    $ conda activate ./.conda-env


Updating dependencies
^^^^^^^^^^^^^^^^^^^^^

Dependencies are listed in `pyproject.toml`_ with version constraints.
Versions are then locked using ``conda-lock`` as previously described.

Anytime dependencies are added to or removed from the ``pyproject.toml`` file,
regenerate the Conda environment lock files, using ``devtools\run_conda_lock.bat``,
or directly from command line::

    (base) $ python devtools/run_conda_lock.py

Regenerate the Conda environment lock files as well when you want to fetch newly
available versions of the dependencies (typically patches, still in accordance with
the specifications expressed in ``pyproject.toml``).


Adding a dependency
-------------------

First install the dependency using ``conda``:

    (path/to/.conda-env) $ conda install my_new_dep

Then update the list of dependencies in `pyproject.toml`_ with a suited version constraint
(if for development only, place it under section ``[tool.poetry.group.dev.dependencies]``).

For example, if ``conda`` installed version 1.5.2 of ``my_new_dep``,
then add ``my_new_dep="^1.5.2"``.

Do not forget to regenerate the Conda environment lock files.


How to use **Poetry** to update the dependency list (optional)
--------------------------------------------------------------

`Poetry <https://python-poetry.org/>`_ provides a command line interface to easily add or remove dependencies:

    (path/to/.conda-env) $ poetry add another_package --lock

Note the ``--lock`` option, that simple creates or updates the lock file, without Poetry installing anything.
``poetry`` would install the package through ``pip`` while we want dependencies to be installed through ``conda``
so that they match the version pinned by ``conda-lock``.

One limitation though: Poetry will look for packages in PyPI only and not in the Conda channels.
The version selected by Poetry might thus not be aviaible for Conda.

To install ``Poetry`` on your computer, refer to the `Poetry documentation`_.

.. _Poetry documentation: https://pre-commit.com/


Configure the pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`pre-commit`_ is used to automatically run static code analysis upon commit.
The list of tools to execute upon commit is configured in the file `.pre-commit-config.yaml`_.

pre-commit can be installed using a Python installation on the system, or one from a Conda environment.

- To install pre-commit using Python (and pip) in your system path:

..  code-block:: bash

    pip install --user pre-commit

- Or to install from an activated Conda environment:

..  code-block:: bash

    conda install -c conda-forge pre-commit

Then, in either way, install the pre-commit hooks as follow (**current directory is the project folder**):

..  code-block:: bash

    pre-commit install

To prepare and check the commit messages, you can also use the following commands:

.. code-block:: bash

    pre-commit install -t prepare-commit-msg -t commit-msg

It configures ``pre-commit`` to prepares and checks the commit ensuring it has a JIRA issue ID:
if no ID was provided, it extracts it from the branch name;
if one was provided, it checks it is the same one as in the branch name.

To run pre-commit manually, use the following command:

..  code-block:: bash

    pre-commit run --all-files

To run only on changes staged for commit:

.. code-block:: bash

    pre-commit run

If a tool fails running, it might be caused by an obsolete versions of the tools that pre-commit is
trying to execute. Try the following command to update them:

..  code-block:: bash

    pre-commit autoupdate

Upon every commit, all the pre-commit checks run automatically for you, and reformat files when required. Enjoy...

If you prefer to run pre-commit upon push, and not upon every commit, use the following commands:

..  code-block:: bash

    pre-commit uninstall -t pre-commit
    pre-commit install -t pre-push

.. _pre-commit: https://pre-commit.com/


Running the tests
^^^^^^^^^^^^^^^^^
Test files are placed under the ``tests`` folder. Inside this folder and sub-folders,
Python test files are to be named with ``_test.py`` as a suffix.
The test function within this files must have a ``test_`` prefix.


Install pytest
--------------

.. _pytest: https://docs.pytest.org/

If you installed  your environment through ``setup-dev.bat``, pytest is already installed.
You can run it from the Conda command (**in your project folder**):

.. code-block:: bash

    pytest tests


Code coverage with Pytest
-------------------------
.. _pytest-cov: https://pypi.org/project/pytest-cov/

If you installed  your environment through ``setup-dev.bat``, `pytest-cov`_ is already installed.
It allows you to visualize the code coverage of your tests.
You can run the tests from the console with coverage:

.. code-block:: bash

    pytest --cov --cov-report html

The html report is generated in the folder ``htmlcov`` at the root of the project.
You can then explore the report by opening ``index.html`` in a browser.

In ``pyproject.toml``, the section ``[tool.coverage.report]`` defines the common options
for the coverage reports. The minimum accepted percentage of code coverage is specified
by the option ``fail_under``.

The section ``[tool.coverage.html]`` defines the options specific to the HTML report.

Git LFS
^^^^^^^
In the case your package requires large files, `git-lfs`_ can be used to store those files.
Copy it from the `git-lfs`_ website, and install it.

Then, in the project folder, run the following command to install git-lfs:

.. code-block:: bash

    git lfs install


It will update the file ``.gitattributes`` with the list of files to track.

Then, add the files and the ``.gitattributes`` to the git repository, and commit.

.. _git-lfs: https://git-lfs.com/

Then, add the files to track with git-lfs:

.. code-block:: bash

    git lfs track "*.desire_extension"


IDE : PyCharm
^^^^^^^^^^^^^
`PyCharm`_, by JetBrains, is a very good IDE for developing with Python.


Configure the Python interpreter in PyCharm
--------------------------------------------

First, excluded the ``.conda-env`` folder from PyCharm.
Do so, in PyCharm, right-click on the ``.conda-env`` folder, and ``Mark Directory as > Excluded``.

Then, you can add the Conda environment as a Python interpreter in PyCharm.

    ..  image:: devtools/images/pycharm-exclude_conda_env.png
        :alt: PyCharm: Exclude Conda environment
        :align: center
        :width: 40%


In PyCharm settings, open ``File > Settings``, go to ``Python Interpreter``,
and add click add interpreter (at the top left):

    ..  image:: devtools/images/pycharm-add_Python_interpreter.png
        :alt: PyCharm: Python interpreter settings
        :align: center
        :width: 80%

Select ``Conda Environment``, ``Use existing environment``,
and select the desired environment from the list (the one in the ``.conda-env`` folder):

    ..  image:: devtools/images/pycharm-set_conda_env_as_interpreter.png
        :alt: PyCharm: Set Conda environment as interpreter
        :align: center
        :width: 80%

Then you can check the list of installed packages in the ``Packages`` table. You should see
**geoapps** and its dependencies. Make sure to turn off the ``Use Conda Package Manager``
option to see also the packages installed through pip:

    ..  image:: devtools/images/pycharm-list_all_conda_packages.png
        :alt: PyCharm: Conda environment packages
        :align: center
        :width: 80%


Run the tests from PyCharm
--------------------------
First, right click on the ``tests`` folder and select ``Mark Directory as > Test Sources Root``:

    ..  image:: devtools/images/pycharm-mark_directory_as_tests.png
        :alt: PyCharm: Add Python interpreter
        :align: center
        :width: 40%

You can now start tests with a right click on the ``tests`` folder and
select ``Run 'pytest in tests'``, or select the folder and just hit ``Ctrl+Shift+F10``.

PyCharm will nicely present the test results and logs:

    ..  image:: devtools/images/pycharm-test_results.png
        :alt: PyCharm: Run tests
        :align: center
        :width: 80%


Execute tests with coverage from PyCharm
----------------------------------------

You can run the tests with a nice report of the code coverage, thanks to the pytest-cov plugin
(already installed in the virtual environment as development dependency as per `pyproject.toml`_).


To set up this option in PyCharm, right click on the ``tests`` folder and ``Modify Run Configuration...``,
then add the following option in the ``Additional Arguments`` field:

    ..  image:: devtools/images/pycharm-menu_modify_test_run_config.png
        :alt: PyCharm tests contextual menu: modify run configuration
        :width: 30%

    ..  image:: devtools/images/pycharm-dialog_edit_test_run_config.png
        :alt: PyCharm dialog: edit tests run configuration
        :width: 60%

Select ``pytest in tests``, and add the following option in the ``Additional Arguments`` field::

    --cov --cov-report html

Then, run the tests as usual, and you will get a nice report of the code coverage.

.. note::
    Running tests with coverage disables the debugger, so breakpoints will be ignored.

Some useful plugins for PyCharm
--------------------------------
Here is a suggestion for some plugins you can install in PyCharm.

- `Toml`_, to edit and validate ``pyproject.toml`` file.
- `IdeaVim`_, for Vim lovers.
- `GitHub Copilot`_, for AI assisted coding.

.. _PyCharm: https://www.jetbrains.com/pycharm/

.. _Toml: https://plugins.jetbrains.com/plugin/8195-toml/
.. _IdeaVim: https://plugins.jetbrains.com/plugin/164-ideavim/
.. _GitHub Copilot: https://plugins.jetbrains.com/plugin/17718-github-copilot

.. _pyproject.toml: pyproject.toml
.. _.pre-commit-config.yaml: .pre-commit-config.yaml


License
^^^^^^^
# TODO: ADD LICENSE TERMS


Copyright
^^^^^^^^^
Copyright (c) 2024 Mira Geoscience Ltd.
