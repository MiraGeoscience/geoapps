Instructions
============

This section provides general information about running the applications.

Jupyter Notebook
----------------

All applications are currently built with `Jupyter Notebooks <https://jupyter-notebook.readthedocs.io/en/stable/>`_
running locally from the user's default web browser.
It is recommended to use either Chrome, Firefox or Microsoft Edge Chromium.

Cell execution
^^^^^^^^^^^^^^

Applications generally consist of two Cells:

	.. figure:: ../images/application_cells.png
	    :align: center
	    :width: 100%

- (Top) Text block that describe what the application is about with useful links.
- (Bottom) A Code block that imports the necessary library and launches the main `Jupyter Widgets <https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Basics.html>`_.

To start the application, the Code cell must be executed. This can be done either through the
Run Button

	.. figure:: ../images/run_button.png
	    :align: center
	    :width: 50%

or by selecting the Code cell and pressing **Shift+Enter** on your keyboard.

Two indicators let the user know that code is currently being executed:
A bracket with a start next to the cell

	.. figure:: ../images/cell_execution_bottom.png
	    :align: center
	    :width: 50%

and a dark circle at the top right corner of the page.

	.. figure:: ../images/cell_execution_top.png
	    :align: center
	    :width: 25%

Once completed, both indicators will turn back white and the result will be displayed on screen.
The initial execution of the application can take several seconds due to the import of several
libraries and the creation of widgets.

FAQ
---

To be continued ...

.. ./../_examples/Automated_Lineament.ipynb
.. ./../_examples/Coordinate_Transformation.ipynb
.. ./../_examples/Create_contours.ipynb
.. ./../_examples/Export_to.ipynb
.. ./../_examples/Geophysical_Inversion_app.ipynb
.. ./../_examples/Grav_Mag_Block_Simulation.ipynb
.. ./../_examples/Object_to_object_interpolation.ipynb
