.. _getting_started:

Getting Started
===============

Whether you are developer or an end-user, this page will help you get started with the **geoapps**.

Installation
------------

- Make sure that you have `Anaconda 3.7 <https://www.anaconda.com/download/>`_ or higher installed.

	.. figure:: ../images/Anaconda_Install.png
	    :align: center
	    :width: 200

- `Download the latest <https://github.com/MiraGeoscience/geoapps/archive/main.zip>`_ **geoapps** directory.

- Extract the package to your drive (SSD if available).

	.. figure:: ../images/extract.png
	    :align: center
	    :width: 50%

- Run ``Install_Update.bat``. The same batch file can be used to update the package dependencies.
  A conda environment named ``geoapps`` will be create to prevent conflicts with other software that may rely on Python.

	.. figure:: ../images/run_install.png
	    :align: center
	    :width: 50%


.. attention:: The assumption is made that Anaconda has been installed in the default directory:
        ``%USERPROFILE%\anaconda3``

        .. figure:: ../images/Install_start_bat.png
            :align: center
            :width: 100%

      If this is not the case, users will need to edit the ``Install_Update.bat`` and
      ``Start_applications.bat`` files in order to point to the Anaconda directory. Common alternatives are:

        - ``%USERPROFILE%\AppData\Local\Continuum\anaconda3``
        - ``C:\ProgramData\anaconda3``




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
