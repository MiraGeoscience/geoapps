:orphan:

.. _dataInterpolation:

Data Interpolation
==================

This application lets users transfer data from one object to another in a fast and
easy way. Alternatively, users can generate a uniform grid (BlockModel) to transfer
data/models at a set resolution and extant.

.. - Choose an object and associated data
.. - Pick a destination object or create a 3D grid
.. - Select the ``Space`` to use for interpolation:
..  - ``Linear``
..  - ``Log``
.. - Select the ``Method``
..  - ``Nearest``: Nearest neighbour interpolation using [scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html) (Fastest)
..  - ``Linear``: Linear interpolation from [scipy.interpolate.LinearNDInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html) (Slowest)
..  - ``Inverse Distance``: Custom method using 8 nearest neighbours and their radial distance from [scipy.spatial.cKDTree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html). (Best for line models) **Skew parameters can be used to compensate for orientated line data with short station separation.**
..    - Azimuth (lines orientation angle from North)
..    - Factor (ratio between along vs cross line distance)
..      - e.g.: For EW orientation @ 200 m line spacing and stations 25 m apart. Use -> Azimuth: 90, Factor: 0.125 (25/200)
.. - Interpolate your data/model !

.. figure:: ./images/data_interp_app.png
        :align: center
        :alt: inv_app



.. note:: The following interactive widgets are for demonstration only.

          The latest version of applications can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

TO BE CONTINUED...
