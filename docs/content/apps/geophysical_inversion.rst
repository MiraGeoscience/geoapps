:orphan:

.. _inversionApp:

Geophysical inversion (SimPEG)
==============================

This application provides an interface to geophysical inversion using the `SimPEG <https://simpeg.xyz/>`_ open-source algorithms. The application currently supports

 - Electromagnetic (time or frequency) data using a Laterally Constrained 1D approach
 - Gravity and magnetics (field and/or tensor) data using an octree mesh tiling approach.

.. note:: For gravity and magnetics data, it is recommended to run the
          inversion from a solid-state drive, as
          sensitivities are stored in chunks and accessed in parallel using
          the `Dask <https://dask.org/>`_ library.

.. figure:: ./images/Geophysical_inversion_app.png
        :align: center
        :alt: inv_app


.. note:: The latest version of applications can be `downloaded here <https://github.com/MiraGeoscience/geoapps/archive/develop.zip>`_.

          See the :ref:`Installation page <getting_started>` to get started.


Input data
----------

The following list of interactive widgets are for documentation and demonstration purposes only.


.. list-table::
   :header-rows: 1

   * - **1- Object and data fields selection**
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.selection import ObjectDataSelection
            ObjectDataSelection(
            select_multiple=True, add_groups=True,
                 h5file=r"../assets/FlinFlon.geoh5",
                 objects="Data_FEM_pseudo3D",
            ).widget
   * - List of objects with corresponding data and data groups.
       The selected data are used to populate **2- Data Components**

       See :ref:`Object, data selection <objectdataselection>`
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.plotting import PlotSelection2D
            app = PlotSelection2D(
              h5file=r"../assets/FlinFlon.geoh5",
            )
            app.widget
   * - See :ref:`Plot and select data <plotselectiondata>`


Data channels options
---------------------

.. list-table::
   :header-rows: 1

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.utils import geophysical_systems
            from ipywidgets.widgets import Dropdown
            Dropdown(
              options=["Magnetics", "Gravity"] + list(geophysical_systems.parameters().keys()),
              description="Survey Type: ",
            )

   * - List of available survey types.
       The application will attempt to assign the *Survey Type* based on
       known *Groups/Data* fields (e.g. *CPI* => *DIGHEM*).
   * -  .. jupyter-execute::
            :hide-code:

            from ipywidgets.widgets import Text
            Text(
              description="Inducing Field [Amp, Inc, Dec]",
              value="60000, 79, 11"
            )

        *(Magnetics only)*
   * - Inducing field parameters
       *[Amplitude (nT), Inclination (dd.dd), Declination (dd.dd)]*
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import ChannelOptions
            app = ChannelOptions("DIGHEM", "Frequency (Hz)")
            app.active.value=True
            app.active
   * - Checked if the channel is to be used in the inversion
   * -  *(EM only)*

        .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import ChannelOptions
            app = ChannelOptions("DIGHEM", "Frequency (Hz)")
            app.label.value = "900"
            app.label

   * - The frequency or time gate for this channel
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import ChannelOptions
            app = ChannelOptions("DIGHEM", "Frequency (Hz)")
            app.channel_selection.options = ["CPI56k", "CPI7000", "CPI900", "CPQ56k", "CPQ7000", "CPQ900"]
            app.channel_selection.value  = "CPI900"
            app.channel_selection
   * - The list of available data channels expected by the *Survey Type*.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import ChannelOptions
            app = ChannelOptions("DIGHEM", "Frequency (Hz)")
            app.uncertainties.value="0, 4"
            app.uncertainties
   * - Uncertainties applied to this channel: *% x abs(data) + floor*
   * - *(EM only)*

       .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import ChannelOptions
            app = ChannelOptions("DIGHEM", "Frequency (Hz)")
            app.offsets.value="8, 0, 0"
            app.offsets

   * - Offsets (m) between the receiver with respect to the transmitter center location.


Spatial information
-------------------

Topography
^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Defines the discrete air/ground interface.
   * -  .. jupyter-execute::
            :hide-code:

            Dropdown(
              options=["Topography", "Receivers", "Line ID (EM)"],
            )
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import TopographyOptions
            app = TopographyOptions(
                h5file=r"../assets/FlinFlon.geoh5",
                objects="Topography", value="Vertices"
            )
            app.options.value="Object"
            app.options.disabled=True
            app.widget

   * - Topography defined by an object x,y location and data z-data value.

       The option *Vertices* refers to the nodes of a ``Points``, ``Curve`` or ``Surface`` object.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import TopographyOptions
            app = TopographyOptions(
                h5file=r"../assets/FlinFlon.geoh5",
                objects="Topography", value="Vertices"
            )
            app.options.value="Relative to Sensor"
            app.options.disabled=True
            app.offset.value = -40
            app.widget

   * - Topography defined by the ``Receiver`` [x, y, z] locations and z-drape value (-below).
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import TopographyOptions
            app = TopographyOptions(
                h5file=r"../assets/FlinFlon.geoh5",
                objects="Topography", value="Vertices"
            )
            app.options.value="Constant"
            app.options.disabled=True
            app.widget

   * - Topography defined by the ``Receiver`` [x, y] locations at constant elevation (m).

Sensors
^^^^^^^

.. list-table::
   :header-rows: 1

   * - Defines the sensor position in 3D space.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import SensorOptions
            h5file = r"../assets/FlinFlon.geoh5"
            app = SensorOptions(h5file=h5file, objects="Data_FEM_pseudo3D")
            app.options.value="sensor location + (dx, dy, dz)"
            app.options.disabled=True
            app.widget
   * - Receiver locations defined by a constant offset from the
       ``Receiver`` [x, y, z] locations.

       Typically used for towed system where the GPS receiver is on the aircraft.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import SensorOptions
            h5file = r"../assets/FlinFlon.geoh5"
            app = SensorOptions(h5file=h5file, objects="Data_FEM_pseudo3D")
            app.options.value="topo + radar + (dx, dy, dz)"
            app.data.options = list(app.data.options) + ["radar"]
            app.data.value = 'radar'
            app.options.disabled=True
            app.widget
   * - Receiver locations defined by the ``Receiver`` [x, y] locations

       and z value interpolated from topography + clearance height.

       Typically used for gridded data with constant draped height

       or for airborne survey with inaccurate GPS elevation (radar height).

Line ID *(EM only)*
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Select data by survey lines.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import LineOptions
            h5file = r"../assets/FlinFlon.geoh5"
            app = LineOptions(h5file=h5file, objects="Data_FEM_pseudo3D")
            app.widget
   * - Select a data channel containing the line IDs and chose lines to be inverted.


Inversion Options
-----------------

List of parameters controlling the inversion.

.. list-table::
   :header-rows: 1

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.option_choices
   * - Output name

        .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.output_name
   * - Name given to the inversion group added to the ANALYST project.
   * - Target misfit

        .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.chi_factor
   * - Target data misfit where 1 = number of data
   * - Uncertainty mode

        .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.uncert_mode
   * - *Estimated*: Calculate uncertainty floor values based on the fields of the

        reference model.

       or

       *User Input*: Apply uncertainties as set in **2- Data Components**

Starting model
^^^^^^^^^^^^^^

Initial model used to begin the inversion.

.. list-table::
   :header-rows: 1

   * -
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.starting_model.options.value = "Model"
            widgets.starting_model.options.disabled = True
            widgets.starting_model.objects.value = "O2O_Interp_25m"
            widgets.starting_model.data.value = "VTEM_model"
            widgets.starting_model.widget
   * - Model object and values selected from any Surface, BlockModel or Octree object

       Values are interpolated onto the inversion mesh using a nearest neighbor algorithm.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.starting_model.options.value = "Value"
            widgets.starting_model.value.value = 1e-4
            widgets.starting_model.options.disabled = True
            widgets.starting_model.widget
   * - Constant half-space value

Susceptibility model *(FEM Only)*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Susceptibility values used in the forward calculations only.

.. list-table::
   :header-rows: 1

   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.susceptibility_model.options.value = "Model"
            widgets.susceptibility_model.options.disabled = True
            widgets.susceptibility_model.objects.value = "O2O_Interp_25m"
            widgets.susceptibility_model.data.value = "VTEM_model"
            widgets.susceptibility_model.widget
   * - Model values selected from any Surface, BlockModel or Octree object

       Values are interpolated onto the inversion mesh using a nearest neighbor algorithm.
   * -  .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.susceptibility_model.options.value = "Value"
            widgets.susceptibility_model.value.value = 1e-4
            widgets.susceptibility_model.options.disabled = True
            widgets.susceptibility_model.widget
   * - Constant half-space value

Regularization
^^^^^^^^^^^^^^

Parameters controlling the regularization function.

.. list-table::
   :header-rows: 1

   * - Reference model

       .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.reference_model.options.value = "None"
            widgets.reference_model.options.disabled = True
            widgets.reference_model.widget

       *(Gravity/Magnetics only)*
   * - No reference value.
   * - *(EM only)*

       .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.reference_model.options.value = "Best-fitting halfspace"
            widgets.reference_model.options.disabled = True
            widgets.reference_model.widget

   * - Preliminary inversion to determine a best-fitting halfspace at each station
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.reference_model.options.value = "Model"
            widgets.reference_model.options.disabled = True
            widgets.reference_model.objects.value = "O2O_Interp_25m"
            widgets.reference_model.data.value = "VTEM_model"
            widgets.reference_model.widget
   * - Model values selected from any Surface, BlockModel or Octree object

       Values are interpolated onto the inversion mesh using a nearest neighbor algorithm.
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.reference_model.options.value = "Value"
            widgets.reference_model.options.disabled = True
            widgets.reference_model.value.value = "1e-4"
            widgets.reference_model.widget
   * - Constant half-space value
   * - :math:`\alpha`-Scaling

       .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.alphas

   * - Scaling between the components of the regularization function.
   * - :math:`l_p`-norms

       .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.norms

   * - Norms applied to the components of the regularization :math:`p_s, p_x, p_y, p_z`

Mesh parameters
^^^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - Octree mesh (Gravity/Magnetics)
   * - .. figure:: ./images/Octree_refinement.png
        :scale: 50%
        :align: left
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import MeshOctreeOptions
            widgets = MeshOctreeOptions()
            widgets.core_cell_size
   * - Dimensions (x,y,z) of the smallest octree cells.
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import MeshOctreeOptions
            widgets = MeshOctreeOptions()
            widgets.octree_levels_topo
   * - Number of layers of cells at each octree level below the topography surface.
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import MeshOctreeOptions
            widgets = MeshOctreeOptions()
            widgets.octree_levels_obs
   * - Number of layers of cells at each octree level below the observation points.
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import MeshOctreeOptions
            widgets = MeshOctreeOptions()
            widgets.depth_core
   * - Minimum depth (m) of the mesh, rounded up to the next power of 2.
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import MeshOctreeOptions
            widgets = MeshOctreeOptions()
            widgets.padding_distance
   * - Additional padding distance (m) along West, East, North, South, Down and Up.
   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import MeshOctreeOptions
            widgets = MeshOctreeOptions()
            widgets.max_distance
   * - Maximum interpolation distance between the observation points.

       Cell sizes are allowed to increase to the next levels beyond this distance.

Bounds
^^^^^^
.. list-table::
   :header-rows: 1

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.lower_bound.value = "1e-5"
            widgets.upper_bound.value = "1e-1"
            widgets.inversion_options["upper-lower bounds"]
   * - Upper and lower bound constraints applied on model values.

       Leave boxes empty to remove bounds


Ignore values
^^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - .. jupyter-execute::
            :hide-code:

            from geoapps.inversion import InversionOptions
            h5file = r"../assets/FlinFlon.geoh5"
            widgets = InversionOptions(h5file=h5file)
            widgets.ignore_values
   * - Ignore data points with dummy values OR outside a threshold value.

       e.g. "<0" will ignore all negative data values.

Optimization
^^^^^^^^^^^^
.. list-table::
   :header-rows: 1

   * - .. jupyter-execute::
          :hide-code:

          from geoapps.inversion import InversionOptions
          h5file = r"../assets/FlinFlon.geoh5"
          widgets = InversionOptions(h5file=h5file)
          widgets.max_iterations
   * - Maximum number of :math:`\beta`-iterations allowed.

       Note that when applying sparse norms, the inversion may require >20 iterations to converge.
   * - .. jupyter-execute::
          :hide-code:

          from geoapps.inversion import InversionOptions
          h5file = r"../assets/FlinFlon.geoh5"
          widgets = InversionOptions(h5file=h5file)
          widgets.chi_factor
   * - Target data misfit where :math:`\chi=1` corresponds to :math:`\phi_d=N` (number of data).
