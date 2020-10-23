.. _scatter_plot:

Scatter Plots
=============

This application lets users visualize up to 5D of data pulled from any
`Geoscience ANALYST
<https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_
objects. The application uses the rich graphical interface of
`Plotly <https://plotly.com/>`_. This application allows users to:

- Easily flip between 2D and 3D cross-plots.
- Plot positive and negative values in log scale (symmetric-log)
- Change the color and size of markers based on property values
- Zoom, pan, rotate and export figures.



.. jupyter-execute::
    :hide-code:

    from geoapps.plotting import ScatterPlots
    import plotly.offline as py

    app = ScatterPlots(
          h5file=r"../assets/FlinFlon.geoh5",
          static=True
    )

    py.iplot(app.crossplot_fig)


.. note:: Active widgets on this page are for demonstration only.

          The latest version of the application can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Input data
^^^^^^^^^^

.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            app = ScatterPlots(
                h5file=r"../assets/FlinFlon.geoh5",
                static=True
            )
            app.project_panel

   * - See :ref:`Workspace selection <workspaceselection>`

.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            from ipywidgets import HBox
            app = ScatterPlots(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            HBox([app.objects, app.data])

   * - List of objects with corresponding data and data groups.

       Data selection available to populate the scatter plot axes.
       See :ref:`Object, data selection <objectdataselection>`


Axes options
^^^^^^^^^^^^

.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            app = ScatterPlots(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app.axes_pannels

   * - Selected axis option panel


.. list-table::
   :header-rows: 1

   * - X, Y and Z axis panels.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            app = ScatterPlots(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app._x_panel

   * - **Active**: Turn the selected axis on/off.

       A 3D scatter plot is displayed if ALL of X, Y and Z axis are active.
   * - **Data**: Select the property to be displayed by the axis.
   * - **Log10**: Scale the values using a symmlog stretch.
   * - **Threshold**: Small value around zero defining the transition between linear to log.
   * - **Min**: Set a lower bound on values displayed by the axis.
   * - **Max**: Set an upper bound on values displayed by the axis.


Additional options
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Color panels.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            app = ScatterPlots(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app._color_maps
   * - **Colormaps**: Choose from the list of color maps.

.. list-table::
   :header-rows: 1

   * - Size panels.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            app = ScatterPlots(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app._size_markers

   * - **Marker size**: Largest marker size.



.. list-table::
   :header-rows: 1

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import ScatterPlots
            import plotly.offline as py

            app = ScatterPlots(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app._trigger

   * - **Save HTML**: Save an interactive HTML file for the current plot layout.
