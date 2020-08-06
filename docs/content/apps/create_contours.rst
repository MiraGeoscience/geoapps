Create contours
===============

With this application, users can create contours from data using the `matplotlib <https://scikit-image.org/>`_ open-source package.
The contouring can be done on points, curves, surfaces and grids.

**Notes**
    - Only the X and Y coordinates are used to generate contours in plan view. The result might not be satisfactory when applied to 3D geometries.
    - Video tutorial `available on Youtube <https://youtu.be/sjaQzZlm8qQ>`_

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

Input options
-------------

-   Select an **object** from the list registered in the target geoh5

    .. jupyter-execute::
        :hide-code:

        from ipywidgets.widgets import Dropdown

        Dropdown(description="Objects", options=['Gravity_Magnetics'])


 - **Data**: Select data contained by the *object* selected above
 - **Contours**: Numerical values or range of values to draw contours. For examples:
     - *-400:50:2000*
         - Draws contours between -400 to 2000 for every 50 increment
     - *-240*
         - Draws contours at a discrete data values of 240
     - *-400:50:2000, -240*
        - Or any combination of the above, in any order: Contours between -400 to 2000 for every 50 increment, plus a contour at -240, and so on..
 - **Save as**: String value used as name of contours added to the geoh5 project. Defauts to the data.name + contours
 - **Assign Z from values**: Contours will be exported with Z (elevation) based on value of contours. Otherwise, contours are linearly draped on the object vertices.

 - **Export to GA**: Triggers write to geoh5

.. figure:: ./images/Contouring_app.png
        :align: center
        :width: 600
