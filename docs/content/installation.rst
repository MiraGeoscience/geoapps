.. _getting_started:

Getting Started
===============

Whether you are developer or an end-user, this page will help you get started with the **geoapps**.

Installation
------------

1- Install Conda for Python 3.7 or higher. Two recommended options:
    - `Miniconda <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`_: ~400 MB of disk space
    - `Anaconda <https://www.anaconda.com/download/>`_: ~3 GB of disk space

	.. figure:: ../images/Anaconda_Install.png
	    :align: center
	    :width: 200

2- `Download the latest <https://github.com/MiraGeoscience/geoapps/archive/main.zip>`_ **geoapps** directory.

3- Extract the package to your drive (SSD if available).

	.. figure:: ../images/extract.png
	    :align: center
	    :width: 50%


4- Run ``Install_Update.bat`` **(see notes below)**.

  The same batch file can be used to update the package dependencies.
  A conda environment named ``geoapps`` will be created to prevent conflicts with other software that may rely on Python.

	.. figure:: ../images/run_install.png
	    :align: center
	    :width: 50%

.. note:: The assumption is made that Conda has been installed in one
   of the default directories:

    - %USERPROFILE%\\ana[mini]conda3\\
    - %LOCALAPPDATA%\\Continuum\\ana[mini]conda3\\
    - C:\\ProgramData\\ana[mini]conda3\\

   If Conda gets installed in a different directory, users will need to add/edit a
   ``get_custom_conda.bat`` file to add their custom path to the ``conda.bat`` file:

        .. figure:: ../images/Install_start_bat.png
            :align: center
            :width: 75%

Running the applications
------------------------
At this point, you will have all required packages to run the applications:

- Run ``Start_Applications.bat``

	.. figure:: ../images/run_applications.png
	    :align: center
	    :width: 50%

You should see the index page displayed in your default browser.

	.. figure:: ../images/index_page.png
	    :align: center
	    :width: 100%

.. note:: Applications run best with either Chrome or Firefox.


From PyPI
---------

The **geoapps** can also be installed directly from PyPI without its dependencies::

    $ pip install geoapps

To install the latest development version of **geoapps**, you can use ``pip`` with the
latest GitHub ``development`` branch::

    $ pip install git+https://github.com/MiraGeoscience/geoapps.git

To work with **geoapps** source code in development, install from GitHub::

    $ git clone --recursive https://github.com/MiraGeoscience/geoapps.git
    $ cd geoapps
    $ python setup.py install

.. note:: The Jupyter-Notebook applications can be `download from source <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_ :

	.. figure:: ../images/download.png
	    :align: center
	    :width: 200
