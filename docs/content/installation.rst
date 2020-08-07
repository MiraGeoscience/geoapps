Installation
============

**geoapps** is currently written for Python 3.6 or higher, and depends on `NumPy <https://numpy.org/>`_ and
`h5py <https://www.h5py.org/>`_.

.. note:: Users will likely want to take advantage of other packages available in the Python ecosystem. We therefore recommend using `Anaconda <https://www.anaconda.com/download/>`_ to manage the installation.

	.. figure:: ../images/Anaconda_Install.png
	    :align: center
	    :width: 200


Install **geoapps** from PyPI::

    $ pip install geoapps

To install the latest development version of **geoapps**, you can use ``pip`` with the
latest GitHub ``development`` branch::

    $ pip install git+https://github.com/MiraGeoscience/geoapps.git

To work with **geoapps** source code in development, install from GitHub::

    $ git clone --recursive https://github.com/MiraGeoscience/geoapps.git
    $ cd geoapps
    $ python setup.py install
