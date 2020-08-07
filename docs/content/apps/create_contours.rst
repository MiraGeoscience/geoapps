Create contours
===============

With this application, users can create contours from data using the `matplotlib <https://scikit-image.org/>`_ open-source package.

- The contouring can be done on points, curves, surfaces and grids.
- Contours can be exported to `Geoscience ANALYST <https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_, either as 2D or 3D curves.

`Video tutorial available on Youtube <https://youtu.be/sjaQzZlm8qQ>`_


.. figure:: ./images/Contouring_app.png
        :align: center
        :alt: inv_app


.. jupyter-execute::
    :hide-code:

    from shutil import copyfile
    from geoapps.functions.processing import contour_values_widget

    my_h5file = r"../geoapps/assets/FlinFlon.geoh5"

    # Lets create a working copy
    new_file = my_h5file[:-6] + "_work.geoh5"
    copyfile(my_h5file, new_file)

    # Generate contours on pre-set values for this demo
    app = contour_values_widget(
        new_file,
        objects="Gravity_Magnetics",
        data='Airborne_TMI',
        contours="-400:100:2000, -240"
    )

    app.children[1]

**Input parameters**

.. list-table::
   :header-rows: 1

   * - **1- Input Data**
     - **Object and data fields selection**
   * - .. jupyter-execute::
        :hide-code:

        from ipywidgets.widgets import Dropdown
        Dropdown(description="Objects", options=['Gravity_Magnetics'])
     - List of objects present in the target ``geoh5`` file.
   * -

     - Select data contained by the *object* selected above
   * -
     - Numerical values or range of values to draw contours. For examples:

        *-400:2000:50*: Draws contours between -400 to 2000 for every 50 increment

        *240*: Draws contours at a discrete data values of 240

        *-400:2000:50, -240*: Any combination of the above, in any order...
   * -
     - String value used as name of contours added to the ``geoh5`` project. Defaults to the data name.
   * -
     - Contours will be exported with Z (elevation) based on value of contours. Otherwise, contours are linearly draped on the object vertices.
   * -
     - Triggers write to ``geoh5``
