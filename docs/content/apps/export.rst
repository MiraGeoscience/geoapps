:orphan:

.. _export:

Export
======

This application exports objects and data to various file formats.


.. figure:: ./images/export_app.png
        :align: center
        :alt: export



.. note:: The latest version of applications can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Input data
----------

The following list of interactive widgets are for documentation and demonstration purposes only.


.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            app = Export(
                h5file=r"../assets/FlinFlon.geoh5
            )
            app.project_panel

   * - See :ref:`Project panel <workspaceselection>`

.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.objects

   * - List of objects and data available for export.



Output Parameters
-----------------

.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            from ipywidgets import HBox
            app = Export(
                  h5file=r"../assets/FlinFlon.geoh5"
            )
            app.file_type

   * - List of file formats currently supported.


.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            app = Export(
                h5file=r"../assets/FlinFlon.geoh5"
            )
            app.trigger
   * - Export trigger button.

.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.export import Export
            app = Export(
                h5file=r"../assets/FlinFlon.geoh5"
            )
            app.live_link_path
   * - Set export directory.
