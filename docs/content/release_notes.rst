Release Notes
=============

Release 0.6.0 - 2021/11/08
--------------------------

**(Major Release)**

**New Application** - Direct-current and IP 3D inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The application provides an interface to the open-source `SimPEG <https://simpeg.xyz/>`_ package for the inversion of direct-current (DC) and induced polarization (IP) data using a Laterally Constrained 1D approach.

 - Direct-current (potential) data inversion for the recovery of conductivity (S/m).
 - Apparent chargeability (V/V) data for the recovery of chargeability (SI).

All inversion are performed on a 3D octree mesh.

Application Updates
^^^^^^^^^^^^^^^^^^^

- All 3D inversions have been updated to SimPEG v0.5.1.
  The transition also includes several updates to parallelization allowing computations on distributed systems.
- Magnetic and gravity inversions now relies on the ui.json input file format.
  - Inversion parameters can be re-imported from existing ui.json files.
  - Alternatively, the ui.json can be loaded in Geoscience ANALYST as a custom UI.
- Update to Octree Mesh Creation and Peak Finder applications to the ui.json format.
- Added unittests
- Bug fixes


Installation Updates
^^^^^^^^^^^^^^^^^^^^

Some changes have been made on the installation routine and dependencies.
Please visit the `Getting Started <https://geoapps.readthedocs.io/en/latest/content/installation.html) page for details>`_.



Release 0.5.1 - 2021/09/01
--------------------------

**(Hot fix)**

- Fix inversion application topography/receiver location from field.
- Fix typos in docs
- Bump requirement version geoh5py=0.4.1
- Add unit test


Release 0.5.0 - 2021/07/15
--------------------------

**(Major Release)**

**New Application** - Octree Mesh Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New application for the creation of octree meshes with arbitrary refinements around entities.

Application Updates
^^^^^^^^^^^^^^^^^^^

- Major update to the Peak Finder algorithm and application. New selection panel for the query of channel groups. Faster plotting.
- Implementation of the *ui.json* for Peak Finder and Octree Mesh Creation. Parameters can be re-loaded in the Notebook app from the Project Panel.
- Object and Data selection widgets now use the entities uuid as reference.
  The name of Objects is displayed with the parent group to facilitate the sorting/selection.
- Zonge (8 Hz) added to the list of airborne EM systems.

.. note::
    Upcoming with Geoscience ANALYST Pro (v3.3.1), the ui.json will be used to launch
    the application directly from a live workspace with drag+drop to the viewport.

        .. image:: applications/images/GA_pro_octree.gif


Release 0.4.1 - 2021/04/07
--------------------------

- Add unit tests on utils
- Hot fixes for docs


Release 0.4.0 - 2021/03/10
--------------------------

**New Application** - Isosurface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New application for the creation of 3D isosurface values around any type of object.


Dependencies
^^^^^^^^^^^^

In order to protect the applications from breaking between releases, we will now fix the version
of most third-party packages. Users will be notified if an update of the requirements is needed.


Application Updates
^^^^^^^^^^^^^^^^^^^

- Fix numpy warnings for deprecated conversion to numpy.float
- Fix issues with gdal and osr imports
- Stability updates to the Peak Finder app.
- Data selection by line ID now accepts ReferencedData type.
- Add base run test for all apps


Release 0.3.0 - 2021/02/11
--------------------------

(Major Release)

**New Application** - Peak Finder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Corporate Sponsor: BHP (Jonathan Lowe)

Application designed for the detection and grouping of time-domain
electromagnetic (TEM) anomalies measured along flight lines.

While initially designed for TEM data, the same application can be used for
the characterization of anomalies of mixed data types
(e.g. magnetics, gravity, topography, etc.).

Documentation Updates
^^^^^^^^^^^^^^^^^^^^^

Major re-work of the documentation to solve limitations with ReadTheDocs.


Application Updates
^^^^^^^^^^^^^^^^^^^

- Coordinate Transformation app now supports (and relies) on Well-Known-Text strings. ESRI and EPSG codes are also allowed.
- New option for Surface Creation of horizons (2.5D surfaces)
- New plotting utilities for Surfaces, Points and BlockModel objects using Plotly
- New EM systems added: Hummingbird, GEOTEM 75 Hz, SkyTEM 306 (HM/LM), QUESTEM

Previous Releases
-----------------

Release 0.2.10 - 2021/01/28
^^^^^^^^^^^^^^^^^^^^^^^^^^^

(Hot fix)

-  Broken dependencies (thanks Joel)


Release 0.2.9 - 2021/01/19
^^^^^^^^^^^^^^^^^^^^^^^^^^

(Minor Release)

- Allow integer data types
- Update data dependencies for ezdxf
- Begin adding skeleton for unit tests (0% coverage)



Release 0.2.6 - 2020/12/14
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Update KMeans clustering application for reference data.


Release 0.2.5
^^^^^^^^^^^^^

- Upper/lower bound values added to the KMeans clustering application.
- Fix for documentation
