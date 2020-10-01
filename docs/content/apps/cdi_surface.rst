:orphan:

.. _cdi_surface:

CDI to 3D surface
=================

With this application, users can convert CDI ``Curve`` to 3D ``Surface`` objects
for visualization and modeling.


.. figure:: ./images/cdi_surface_app.png
        :align: center
        :alt: inv_app



.. note:: The latest version of applications can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.

          The following list of interactive widgets are for documentation and demonstration purposes only.

Input Parameters
----------------

Project
^^^^^^^

See :ref:`Project panel <workspaceselection>`


.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    app = CDICurve2Surface(
        h5file=r"../assets/FlinFlon.geoh5",
    )
    app.project_panel

Object and data
^^^^^^^^^^^^^^^

List of objects with corresponding data groups available for transfer to the
new ``Surface`` object.

See :ref:`Object, data selection <objectdataselection>`

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    from ipywidgets import HBox
    app = CDICurve2Surface(
          h5file=r"../assets/FlinFlon.geoh5",
    )
    HBox([app.objects, app.data])


Z options
^^^^^^^^^

Elevation
"""""""""

Assign z-coordinates based on ``Elevation`` (m) field provided by the
``Curve`` object.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    from ipywidgets import HBox
    app = CDICurve2Surface(
          h5file=r"../assets/FlinFlon.geoh5",

    )
    app.z_option.disabled = True
    app.depth_panel

Depth
"""""

Assign z-coordinates based on a ``Depth`` field provided by the ``Curve``
object

.. jupyter-execute::
        :hide-code:

        from geoapps.processing import CDICurve2Surface
        from ipywidgets import HBox
        app = CDICurve2Surface(
              h5file=r"../assets/FlinFlon.geoh5",

        )
        app.z_option.value = "depth"
        app.z_option.disabled = True
        HBox([app.z_option, app.elevations.data])

The final elevation is assigned relative to either:

  - A topography ``Object`` with elevation

    .. jupyter-execute::
        :hide-code:

        from geoapps.processing import CDICurve2Surface
        from ipywidgets import HBox
        app = CDICurve2Surface(
              h5file=r"../assets/FlinFlon.geoh5",

        )
        app.z_option.value = "depth"
        app.topography.objects.value = "Topography"
        app.topography.data.value = "Z"
        app.z_option.disabled = True
        app.topography.options.disabled = True
        app.topography.widget

  - A constant offset value ``Relative to Sensor`` (below curve vertices)

    .. jupyter-execute::
        :hide-code:

        from geoapps.processing import CDICurve2Surface
        from ipywidgets import HBox
        app = CDICurve2Surface(
              h5file=r"../assets/FlinFlon.geoh5",

        )
        app.z_option.value = "depth"
        app.topography.options.value = "Relative to Sensor"
        app.topography.options.disabled = True
        app.topography.widget

  - A ``Constant`` elevation

    .. jupyter-execute::
        :hide-code:

        from geoapps.processing import CDICurve2Surface
        from ipywidgets import HBox
        app = CDICurve2Surface(
              h5file=r"../assets/FlinFlon.geoh5",

        )
        app.z_option.value = "depth"
        app.topography.options.value = "Constant"
        app.topography.options.disabled = True
        app.topography.widget



Line
^^^^

Select ``Line`` field identifier to brake up the sections.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    from ipywidgets import HBox
    app = CDICurve2Surface(
          h5file=r"../assets/FlinFlon.geoh5",

    )
    app.lines.data


Triangulation
^^^^^^^^^^^^^

Maximum triangulation distance allowed during the ``Surface`` creation.

Useful option for CDI curves with missing values.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    from ipywidgets import HBox
    app = CDICurve2Surface(
          h5file=r"../assets/FlinFlon.geoh5",

    )
    app.max_distance


Output Parameters
-----------------

String value used to name the new ``Surface`` object.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    app = CDICurve2Surface(
        h5file=r"../assets/FlinFlon.geoh5",
    )
    app.export_as

See :ref:`Trigger panel<trigger_panel>` base applications.

.. jupyter-execute::
    :hide-code:

    from geoapps.processing import CDICurve2Surface
    app = CDICurve2Surface(
        h5file=r"../assets/FlinFlon.geoh5",
    )
    app.trigger_panel
