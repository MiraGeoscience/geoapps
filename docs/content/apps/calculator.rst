:orphan:

.. _calculator:

Calculator
==========

With this application, users can add or edit data from existing ``Objects`` using simple Python
operations or any function from the `Numpy
<https://numpy.org/doc/stable/reference/index.html>`_ library.


.. figure:: ./images/calculator_app.png
        :align: center
        :alt: calc



.. note:: The latest version of applications can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Input data
----------

The following list of interactive widgets are for documentation and demonstration purposes only.


.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            app = Calculator(
                h5file=r"../assets/FlinFlon.geoh5",
                static=True
            )
            app.project_panel

   * - See :ref:`Project panel <workspaceselection>`

.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            from ipywidgets import HBox
            app = Calculator(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app.objects

   * - List of objects available to pull data from.


.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            from ipywidgets import HBox
            app = Calculator(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            HBox([app.use, app.data])

   * - Add the selected data as a variable and append to the scripting window.


.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            from ipywidgets import HBox
            app = Calculator(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            HBox([app.add, app.channel])

   * - Add the name as data to the current object. Values of zeros are assigned to the new data by default.


Equation
^^^^^^^^

.. list-table::
   :header-rows: 0

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            from ipywidgets import HBox
            app = Calculator(
                  h5file=r"../assets/FlinFlon.geoh5",
                  static=True
            )
            app.equation.value = "numpy.log10(var['geochem.Al2O3']) / (var['geochem.CaO']/2 + var['geochem.Cu']**3.)"
            app.equation
   * - Scripting window used to compute values.

       All core Python element-wise operators are accepted: add (+), subtract (-), multiply
       (*), divide (/), power (**). Line breaks between operations can be used
       for clarity but must be surrounded by parentheses ().

       `Numpy <https://numpy.org/doc/stable/reference/index.html>`_ operations can also be used e.g.: numpy.log10() (log base 10)


Output Parameters
-----------------

.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            app = Calculator(
                h5file=r"../assets/FlinFlon.geoh5",
            )
            app.store.data
   * - Assign the result to the specified data.

.. list-table::
   :header-rows: 0

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            app = Calculator(
                h5file=r"../assets/FlinFlon.geoh5",
            )
            app.trigger

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.processing import Calculator
            app = Calculator(
                h5file=r"../assets/FlinFlon.geoh5",
            )
            app.live_link_panel
   * - See :ref:`Trigger panel<trigger_panel>` base applications.
