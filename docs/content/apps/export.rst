:orphan:

.. _export:

Export
======

This application lets users export objects and data from a ``geoh5`` to
various open file formats.


.. figure:: ./images/export_app.png
        :align: center
        :alt: export



.. note:: Active widgets on this page are for demonstration only.

          The latest version of the application can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Worksapce
---------

Select target ``geoh5`` file (see :ref:`Project panel <workspaceselection>`)

.. jupyter-execute::
    :hide-code:

    from geoapps.export import Export
    app = Export(
        h5file=r"../assets/FlinFlon.geoh5"
    )
    app.project_panel



Input data
----------

Select from the list of objects and data available for export (see :ref:`Object, data selection <objectdataselection>`)

.. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            HBox([app.objects, app.data])



Output Parameters
-----------------

List of file formats currently supported.

.. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.file_type



ESRI Shapefile (``Points``, ``Curve``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Export option to **.shp**. Z-coodinate is ignored.

``EPSG code``: Coordinate system assigned to the shapefile (`Spatial Reference <https://spatialreference.org/ref/epsg/>`_).


.. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.file_type.value = "ESRI shapefile"
            app.file_type.disabled = True
            app.type_widget


Column value (All)
^^^^^^^^^^^^^^^^^^

Export option to **csv**. The x, y and z coordinates of every nodes/cells are appended to the list of data by default.


.. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.file_type.value = "csv"
            app.file_type.disabled = True
            app.type_widget


Geotiff (``Grid2D``)
^^^^^^^^^^^^^^^^^^^^

Export option to **.geotiff**.
        - ``EPSG code``: Coordinate system assigned to the shapefile (`Spatial Reference <https://spatialreference.org/ref/epsg/>`_).
        - ``Type``: Type of geotiff exported
           - ``Float``: Single-band image containing the float value of selected data.
           - ``RGB``: 3-band image containing the RGB color displayed in ANALYST.


.. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.file_type.value = "geotiff"
            app.file_type.disabled = True
            app.type_widget



UBC format (``BlockModel``, ``Octree``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Export option to UBC mesh (**.msh**) and model (**.mod**) format.


.. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.file_type.value = "UBC format"
            app.file_type.disabled = True
            app.type_widget


Output
------

Set export directory and trigger export

.. jupyter-execute::
    :hide-code:

    from geoapps.export import Export
    from ipywidgets import VBox

    app = Export(
        h5file=r"../assets/FlinFlon.geoh5"
    )
    VBox([
      app.trigger
      app.export_directory
    ])
