:orphan:

.. _dataInterpolation:

Data Interpolation
==================

This application lets users transfer data from one object to another.
Alternatively, users can generate a uniform grid (BlockModel) to transfer
data/models at a set resolution and extant.

.. figure:: ./images/data_interp_app.png
        :align: center
        :alt: data_interp


.. note:: Active widgets on this page are for demonstration only.

          The latest version of the application can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Input Parameters
----------------

Project
^^^^^^^

Select a target ``geoh5`` file. See :ref:`Project panel <workspaceselection>`

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
        h5file=r"../assets/FlinFlon_light.geoh5"
    )
    app.project_panel



Source Object/Data
^^^^^^^^^^^^^^^^^^

List of objects with corresponding data groups available for transfer to the
neighboring object. See :ref:`Object, data selection <objectdataselection>`

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    from ipywidgets import VBox
    app = DataInterpolation(
          h5file=r"../assets/FlinFlon_light.geoh5",
    )
    VBox([app.objects, app.data])



Destination
^^^^^^^^^^^

Object to transfer data onto.

To Object
"""""""""

List of objects available in the target ``geoh5``.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
          h5file=r"../assets/FlinFlon_light.geoh5",
    )
    app.out_mode.value = "To Object"
    app.out_mode.disabled = True
    app.out_panel


Create 3D Grid
""""""""""""""

Create a new ``BlockModel`` object to transfer data onto. Useful for merging
EM1D inversion results.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
          h5file=r"../assets/FlinFlon_light.geoh5",
    )
    app.out_mode.value = "Create 3D Grid"
    app.out_mode.disabled = True
    app.out_panel



- **Name**: Name assigned to the new ``BlockModel`` object.
- **Lateral Extent**: Use an object (usually the ``Source`` object) to determine the horizontal extent of the new grid.
- **Smallest cells**: Grid cell size (m) along the x, y and z-axis.
- **Core depth**: Depth of the grid using the smallest cell size.
- **Pad Distance**: Add padding cells outside the core region. Requires six values of distances (m) along: [West, East, North, South, Down and Up]
- **Expansion Factor**: If padding distances are used, the rate of expansion of those cells where: :math:`h_x = h_0 * \alpha^{[0, 1, ..., N_c]}`

Interpolation Parameters
------------------------

List of additional parameters controlling the interpolation.


.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
          use_defaults=False
    )
    app.parameter_choices


Interpolation methods
^^^^^^^^^^^^^^^^^^^^^

Type of algorithm used to interpolate values.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
          use_defaults=False
    )
    app.method

- **Nearest**: (Fastest) Use the nearest neighbors between the ``Source`` and ``Destination`` objects (vertices or centroids). Uses `Scipy.spatial.cKDTree <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html>`_
- **Linear**: (Slowest) Use a Delaunay triangulation from `Scipy.interpolate.LinearNDInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.LinearNDInterpolator.html>`_
- **Inverse Distance**: Weighted averaging using 8 nearest neighbors:

  :math:`val = \frac{\sum_{i=1}^8 val_i * r_i^{-1}}{\sum_{i=1}^8 r_i^{-1}}`

  where :math:`r_i` is the radial distance between a vertex/centroid to its :math:`i^{th}` nearest neighbor.

Skew Parameters
"""""""""""""""
Options for dealing with spatially elongated ``Source`` values

 - *Azimuth*: Angle (degree) from North of ``Source`` object orientation.
 - *Factor*: Aspect ratio between the in-line spacing and the line separation  (i.e. 25 m stations / 100 line spacing => 0.25)


.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
          use_defaults=False
    )
    app.method_skew


Scaling
^^^^^^^

Conversion of values to ``linear`` or ``log`` space before interpolation.
Interpolating the log is usually preferred when dealing with large dynamic
ranges, such as resistivity.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
          use_defaults=False
    )
    app.space

- **Linear**: Values interpolated in linear-space.
- **Log**: Values converted to log-space before interpolation. The result is converted back to its original space (signed) after transfer. Useful for data values spanning multiple order of magnitude such as resistivity.


Horizontal Extent
^^^^^^^^^^^^^^^^^

Add limits to the horizontal extrapolation of the data.

Object hull
"""""""""""

Use radial distance from an object vertices or centroids.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    from ipywidgets import VBox
    app = DataInterpolation(
          use_defaults=False
    )
    app.xy_extent


Max distance
""""""""""""

Set the maximum extrapolation distance (m).

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    from ipywidgets import VBox
    app = DataInterpolation(
          use_defaults=False
    )
    app.max_distance


Vertical Extent
^^^^^^^^^^^^^^^

Add limits to the vertical extrapolation of the data.

Topography
""""""""""

Define the upper boundary from topography (see :ref:`Topography <topo_widget>`)

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    from ipywidgets import VBox
    app = DataInterpolation(
          use_defaults=False
    )
    app.topography.widget



Max depth
"""""""""

Set the maximum depth (vertical distance below ``Source``) to extrapolate data.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    from ipywidgets import VBox
    app = DataInterpolation(
          use_defaults=False
    )
    app.max_depth


Output
------

See :ref:`Trigger panel<trigger_panel>` base applications.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import DataInterpolation
    app = DataInterpolation(
        h5file=r"../assets/FlinFlon_light.geoh5",
    )
    app.trigger_panel
