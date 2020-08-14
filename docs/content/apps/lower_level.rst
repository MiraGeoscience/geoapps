Low-level applications
======================

This section provides a list of low-level applications that can be combined
together to form more complex ones.


.. _objectdataselection:

Object, data selection
----------------------

Application for the selection an object and associated data from a target
geoh5.

.. jupyter-execute::

    from geoapps.selection import ObjectDataSelection

    # Create an instance with pointer to the target geoh5
    app = ObjectDataSelection(
        h5file="../assets/FlinFlon.geoh5",
        objects="Gravity_Magnetics_drape60m" # [Optional]
    )

    # Display the app
    app.widget


.. _plotselectiondata:

Plot and select data
--------------------

Application for the selection and downsampling of data in 2D plan view.

.. jupyter-execute::

    from geoapps.plotting import PlotSelection2D

    # Create an instance with pointer to the target geoh5
    app = PlotSelection2D(
        h5file="../assets/FlinFlon.geoh5",
        objects="Gravity_Magnetics_drape60m" # [Optional]
    )

    # Display the app
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
