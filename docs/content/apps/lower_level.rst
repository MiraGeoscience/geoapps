:orphan:

Base Applications
=================

This section provides a list of low-level applications that can be combined
together to form more complex ones.


.. _workspaceselection:

Project panel
-------------

Application adapted from `ipyfilechoose <https://pypi.org/project/ipyfilechooser/>`_ for the selection of a workspace using


.. jupyter-execute::

    from geoapps.base import BaseApplication

    app = BaseApplication()
    app.project_panel

Select
^^^^^^
Prompts browsing menu

Create a copy
^^^^^^^^^^^^^
Make a working copy of the selected ``geoh5`` in place.


.. _trigger_panel:

Run and save
------------

.. jupyter-execute::

    from geoapps.base import BaseApplication

    app = BaseApplication()
    app.trigger_panel

Application for starting some computation and saving the output to ``geoh5``

Compute
^^^^^^^
Generic ToggleButton to trigger some function.

To Group
^^^^^^^^
Create a ``geoh5`` ``Group`` to store new objects created. Results added to the project ``Root`` if left empty.

GA Pro - Live link
^^^^^^^^^^^^^^^^^^
Activate the live link mode for GA Pro users.


.. _objectdataselection:

Object, data selection
----------------------

.. jupyter-execute::

    from geoapps.selection import ObjectDataSelection
    app = ObjectDataSelection(
        h5file="../assets/FlinFlon.geoh5",
        objects="Gravity_Magnetics_drape60m" # [Optional]
    )
    app.widget

Application for the selection an object and associated data from a target
geoh5.


Objects
^^^^^^^
List of objects present in the target ``geoh5``

Data
^^^^
``Dropdown`` or ``SelectMultiple`` widget for the selection of data contained in the selected object



.. _plotselectiondata:

Plot and select data
--------------------

.. jupyter-execute::

    from geoapps.plotting import PlotSelection2D

    app = PlotSelection2D(h5file="../assets/FlinFlon.geoh5")
    app.widget

Application for the selection and downsampling of data in 2D plan view.


Parameters
^^^^^^^^^^

Resolution
""""""""""
Minimum distance between data points

Northing
""""""""
Vertical window position (m)

Easting
"""""""
Horizontal window position (m)

Height
""""""
Window size (m) along orientation

Width
""""""
Window size (m) across orientation

Orientation
"""""""""""
Azimuth of the window (dd.dd)

Zoom on selection
"""""""""""""""""
Limit the plot to the window extent


.. _topo_widget:

Topography Options
------------------

.. jupyter-execute::

    from geoapps.selection import TopographyOptions

    app = TopographyOptions(h5file="../assets/FlinFlon.geoh5")
    app.options

Generic widget to define topography.

Options
^^^^^^^

Object
""""""

.. jupyter-execute::
    :hide-code:

    from geoapps.selection import TopographyOptions
    app = TopographyOptions(h5file="../assets/FlinFlon.geoh5")
    app.options.disabled = True
    app.objects.value = "Topography"
    app.data.value = "Z"
    app.widget

Select an ``Object`` and ``Data`` defining x, y (from vertices or centroids) and vertical position.

Relative to Sensor
""""""""""""""""""

.. jupyter-execute::
    :hide-code:

    from geoapps.selection import TopographyOptions
    app = TopographyOptions(h5file="../assets/FlinFlon.geoh5")
    app.options.disabled = True
    app.options.value = "Relative to Sensor"
    app.options.disabled = True
    app.widget

Topography is defined by a vertical offset from a selected object position (vertices or centroids).


Constant
""""""""

.. jupyter-execute::
    :hide-code:

    from geoapps.selection import TopographyOptions
    app = TopographyOptions(h5file="../assets/FlinFlon.geoh5")
    app.options.disabled = True
    app.options.value = "Constant"
    app.widget

Topography is defined as a flat surface with constant elevation.
