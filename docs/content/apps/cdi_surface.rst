:orphan:

.. _cdi_surface:

CDI to 3D surface
=================

With this application, users can convert CDI curve objects to surfaces for 3D
visualization.


.. figure:: ./images/cdi_surface_app.png
        :align: center
        :alt: inv_app



.. note:: The latest version of applications can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Input data
----------

The following list of interactive widgets are for documentation and demonstration purposes only.


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

   * - See :ref:`Project panel <workspaceselection>`

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

       The selected all data made available to populate the scatter plot axes options

       See :ref:`Object, data selection <objectdataselection>`


Output Parameters
-----------------

.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.processing import CDICurve2Surface
            app = CDICurve2Surface(
                h5file=r"../assets/FlinFlon.geoh5",
            )
            app.export_as
   * - String value used as name of contours added to the ``geoh5`` project.

.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.processing import CDICurve2Surface
            app = CDICurve2Surface(
                h5file=r"../assets/FlinFlon.geoh5",
            )
            app.trigger_panel
   * - See :ref:`Trigger panel<trigger_panel>` base applications.
