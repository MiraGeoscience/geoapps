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


.. list-table::
   :header-rows: 0

   * - Select
     - Prompts browsing menu
   * - Create a copy
     - Make a working copy of the selected ``geoh5`` in place.


.. _trigger_panel:

Run and save
------------

Application for starting some computation and saving the output to ``geoh5``

.. jupyter-execute::

    from geoapps.base import BaseApplication

    app = BaseApplication()
    app.trigger_panel

.. list-table::
   :header-rows: 0

   * - Compute
     - Trigger some function
   * - To Group
     - Create a ``geoh5`` Group to store new objects created. Results added to the project ``Root`` if left empty.
   * - GA Pro - Live link
     - Activate the live link mode for GA Pro users.


.. _objectdataselection:

Object, data selection
----------------------

Application for the selection an object and associated data from a target
geoh5.

.. jupyter-execute::

    from geoapps.selection import ObjectDataSelection
    app = ObjectDataSelection(
        h5file="../assets/FlinFlon.geoh5",
        objects="Gravity_Magnetics_drape60m" # [Optional]
    )
    app.widget

.. list-table::
   :header-rows: 0

   * - Objects
     - List of objects present in the target ``geoh5``
   * - Data
     - ``Dropdown`` widget for the selection of data contained in the selected object
   * - [OPTIONAL]
     -
   * - ``select_multiple=True``
     - ``Data`` becomes a ``SelectMultiple`` widget
   * - ``object_types=()``
     - Restrict the selectable objects to specific types.
   * - ``add_groups=False``
     - If ``True``, data groups are added to the list.
   * - ``find_label = []``
     - List of strings used to pre-select data if encountered


.. _plotselectiondata:

Plot and select data
--------------------

Application for the selection and downsampling of data in 2D plan view.

.. jupyter-execute::

    from geoapps.plotting import PlotSelection2D

    app = PlotSelection2D(h5file="../assets/FlinFlon.geoh5")
    app.widget

.. list-table::
   :header-rows: 1

   * - **Parameters**
     - **Description**
   * - Resolution
     - Minimum distance between data points
   * - Northing
     - Vertical window position (m)
   * - Easting
     - Horizontal window position (m)
   * - Height
     - Window size (m) along orientation
   * - Width
     - Window size (m) across orientation
   * - Orientation
     - Azimuth of the window (dd.dd)
   * - Zoom on selection
     - Limit the plot to the window extent
