Release Notes
=============

Release 0.12.0 (2025-04-14)
---------------------------

Major release

- GEOPY-378: Allow balancing of octree mesh.
- GEOPY-440: Fix reference angles in MVI
- GEOPY-998: Migrate Peak Finder to **peak-finder-app** repository.
- GEOPY-1072: Implement line refinement for octree mesh.
- GEOPY-1224: Migrate inversion code to **simpeg-drivers** repository.
- GEOPY-1484: Migrate edge detection to **curve-apps** repository.
- General maintenance and bug fixes.


Release 0.11.1 (2024-03-04)
---------------------------

Hotfix release for issues encountered since 0.11.0.

- GEOPY-1265: Log applied twice on joint survey inversion with EM methods
- GEOPY-1275: Fix crash on DCIP 2D inversion with line ID on nodes. Line ID is now on M-N pairs.
- GEOPY-1325: Inducing field strength not applied to the MVI inversion. Issue introduced since updating SimPEG to v0.19.0.
- GEOPY-1331: Apply fixes to coterminal angle calculation for MVI inversion.


Release 0.11.0 (2023-10-04)
---------------------------

**(Major Release)**

New features
^^^^^^^^^^^^

- GEOPY-715: Add Joint survey inversion: multi-surveys, single physical property.
- GEOPY-750: Add frequency-domain 3D inversion.
- GEOPY-787: Add Joint Cross Gradient inversion: up to 3 physical properties.
- GEOPY-871: Add B-field receivers to TEM inversion
- GEOPY-995: Migrate utility functions to lightweight package: https://pypi.org/project/geoapps-utils/
- GEOPY-1100: Add octree refinement on triangulated 3D Surfaces.
- GEOPY-1059 : Add octree refinement along curve path.


Inversion updates
^^^^^^^^^^^^^^^^^

- GEOPY-74: Update SimPEG from v0.15.0 to v0.19.0.
- GEOPY-267: Save command log to inversion group.
- GEOPY-328: Replace 1D inversion models from surface to DrapeModel object.
- GEOPY-401: Remove clipping of topography on data extent.
- GEOPY-564, 720: Save uijson to inversion group and update metadata.
- GEOPY-607: Migrate from jupyter widgets to Dash application.
- GEOPY-189, 217, 483, 507: Updates to behaviour of applications.
- GEOPY-632: Remove detrend options in ui.json.
- GEOPY-927: Allow to run dcip-2D inversion from existing mesh.
- GEOPY-1021: Remove data windowing from ui.json.

General Feature updates
^^^^^^^^^^^^^^^^^^^^^^^

- GEOPY-767, 994: Migrate and refactor Peak Finder to separate repository.
- GEOPY-961: Use Qt web window for Dash applications.
- GEOPY-1100, 1020: Fix deprecation warnings.
- GEOPY-1059: Re-implementation of radial and surface refinement.

UI.json features
^^^^^^^^^^^^^^^^

- GEOPY-830: Use of pathlib.Path for file paths.
- GEOPY-875: Add geoapps 'version' identifier to all applications.


Release 0.10.0 (2023-03-20)
---------------------------

**(Major Release)**

- GEOPY-738: Add Airborne Time-Domain EM (ATEM) inversion to the inversion suite.
- GEOPY-829, 727: Bug fixes


Release 0.9.2 (2023-01-17)
--------------------------

Hotfix release for issues encountered since 0.9.1.

- GEOPY-835: Fix iso-surface creation crash after multiple runs of marching cube.
- GEOPY-734, 827, 828, 829, 833: Improve installation and fix SSL error encountered by some users.
- GEOPY-814: Update copyright year
- GEOPY-732: Fix crash on ui.json execution of non-inversion apps from ANALYST.
- GEOPY-729: Add version information to main documentation page.


Release 0.9.1 (2022-12-13)
--------------------------

This release fixes some of the issues encountered since releasing v0.9.0.

- GEOPY-697, 694, 685: Better handling of Curve entity in inversions UI.
- GEOPY-690: Re-implementation of the Z from topo option for DC-IP and MT inversions. Source and receivers are no longer draped onto the active (discretized) topography. To reduce numerical artifacts, the active set is instead augmented to include cells intercepted by receivers locations. The drape on top remains optional as for all other methods.
- GEOPY-397: Re-simulation of tensor gravity and magnetics in the Flin Flon demo project.

Also included are SimPEG specific changes:

- Fix error in the IRLS weights for MVI using the "total" gradient option.
- Fix error in the stashed regularization operator introduced in v0.9.0


Release 0.9.0 (2022-10-28)
--------------------------

**(Major Release)**

This release focuses on SimPEG DC/IP 2D inversion as well as a revamp of all inversion UIs.

- GEOPY-604-606: Create inversion UI and mechanics for SimPEG DC/IP 2D inversion.
- GEOPY-657: Standardization and update of all inversion UIs (Grav, MVI, DCIP, Natural Sources)
    - Removal of octree mesh creation within the inversion UI. Creation must be done prior to running the inversion.
    - Reference, starting and bound models referenced to the input mesh. Interpolation most be done prior to running the inversion.
    - General UX improvements to layout.
- GEOPY-645: Add beta cooling rate and cooling factor option.
- GEOPY-641: Add option to store sensitivities on RAM or SSD (default).
- GEOPY-613: Allow for TEM survey entities as input for SimPEG EM1D inversions.

New or revamped applications:

- GEOPY-579: New BlockModel creation application.
- GEOPY-592: Conversion of Clustering to Dash (Plotly) app with ui.json implementation.
- GEOPY-588: Conversion of Scatter Plot to Dash (Plotly) app with ui.json implementation.
- GEOPY-534: Conversion of Edge Detection to Dash (Plotly) app with ui.json implementation.
- GEOPY-456: Conversion of Contouring to Dash (Plotly) app with ui.json implementation.



Release 0.8.1 (2022-09-15)
--------------------------

**(Hot fix)**

- Fix corruption of geoh5 file for inversions from ANALYST runs.
- Fix issues with iso-surface introduced by geoh5py v0.4.0 update.
- Fix re-load of jupyter apps (Peak Finder, 3D inversions)  from ui.json.
- PEP8 (pylint) compliance code update.


Release 0.8.0 (2022-07-06)
--------------------------

**(Major Release)**

This release focuses on updates to be compatible with ``geoh5 v2.0`` and ``Geoscience ANALYST v4.0``:
 - Compatibility update for ``geoh5py v0.3.0``.
 - Make Jupyter apps access data in read-only.

New UI.json implementations:
 - Iso-surface creation

Inversion updates:
 - Use of the ``SimPEGGroup`` for storing inputs, log file and results.
 - Inversion Directive compliance with geoh5 open/close mechanism.

General maintenance and bug fixes.


Release 0.7.1 (2022-05-03)
--------------------------

**(Hot fix)**

Fix dependency on geoana v0.1.3


Release 0.7.0 (2022-04-25)
--------------------------

**(Major Release)**

Changes to core functionalities:
 - Migration of ui.json read/write and validations to geoh5py implementation
 - Make soft dependencies for gdal/fiona
 - Resolve conflicts with geosoft/anaconda installation.
 - Run command for all inversion standardized to ``geoapps.inversion.driver``
 - Update dependency to ``geoh5py 0.2.0``

New development focused on natural source data inversion:
 - Implement impedance (MT) inversion with run test
 - Implement tipper (ZTEM) inversion with run test\
 - Improved spatial tiling

Bug fixes:
 - Bad residual calculations on gz, gxz and gyz
 - Remove air cells from DC starting model
 - Allow Points and Curve entities for starting/ref model input.
 - Wrong padding direction on Data transfer app mesh creation.


Release 0.6.3 (2022-02-09)
--------------------------

**(Hot fixes)**

 - Fix limitations for unrecognized ui.json keys
 - Fix Mag/grav inversion crash for:
    - Selection of multi-components grav/mag data
    - Constant topography value option
    - Min/max values for inducing field angles
    - Update ui.json for default bool index


Release 0.6.2 (2022-01-05)
--------------------------

**(Hot fixes)**

 - Fix Block Model origin issue in Data Transfer App
 - Fix optional tem checkbox in PeakFinder App
 - Fix issue with sorting of data in dropdowns
 - Fix issue with reference MVI model
 - Fix FEM-1D crash when using susceptibility model
 - Fix crash on geoh5 change for Octree Mesh App
 - Docs maintenance


Release 0.6.1 (2021-12-09)
--------------------------

**(Minor Release)**

This release mostly addresses issues encountered since release of v0.6.0.

 - Remove json warnings from jupyter-notebook apps.
 - Optimization and bug fixes for Peak Finder
 - Fix crash at the end of multi-component inversions
 - Fix update upper/lower bound dropdowns on geoh5 change.
 - Remove 'Z' options from dropdown channels. Leave empty if vertices are to be used.
 - Remove redundant checkboxes for data channel selection in inversion ui.json files
 - General API maintenance and unit tests


Release 0.6.0 (2021-11-08)
--------------------------

**(Major Release)**

**New Application** - Direct current and induced polarization 3D inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The application provides an interface to the open-source `SimPEG <https://simpeg.xyz/>`_ package for 3D inversion of direct current (DC) and induced polarization (IP) data.

 - Direct current data (V/A) inversion for the recovery of conductivity (S/m).
 - Apparent chargeability data (V/V) for the recovery of chargeability (SI).

All inversions are performed on a 3D octree mesh.

**Application Updates**

- All 3D inversions have been updated to SimPEG v0.15.1.
    - The transition also includes several updates to parallelization allowing computations on distributed systems.
- Magnetic and gravity inversions now relies on the ui.json input file format.
    - Inversion parameters can be re-imported from existing ui.json files.
    - Alternatively, the ui.json can be loaded in Geoscience ANALYST as a custom UI.
- Magnetic vector inversions can now be run with starting and reference models consisting of amplitude, inclination, and declination components.
- Inversion apps now include a detrending option to remove an nth order polynomial using either all the data or just the perimeter points.
- Octree Mesh Creation and Peak Finder applications also now rely on the ui.json format.
- Added unit tests
- Bug fixes
- This release will be accompanied by a Geoscience ANALYST release (v3.4) that exposes geoapps applications to Pro Geophysics users via dropdown menu.
  Follow the release link (`Geoscience ANALYST v3.4 <https://mirageoscience.com/geoscience-analyst-v3-4/>`_) to learn more and find out what else is included.

Installation Updates
^^^^^^^^^^^^^^^^^^^^

Some changes have been made on the installation routine and dependencies.
Please visit the `Getting Started <https://geoapps.readthedocs.io/en/latest/content/installation.html) page for details>`_.



Release 0.5.1 (2021-09-01)
--------------------------

**(Hot fix)**

- Fix inversion application topography/receiver location from field.
- Fix typos in docs
- Bump requirement version geoh5py=0.4.1
- Add unit test


Release 0.5.0 (2021-07-15)
--------------------------

**(Major Release)**

**New Application** - Octree Mesh Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New application for the creation of octree meshes with arbitrary refinements around entities.

**Application Updates**

- Major update to the Peak Finder algorithm and application. New selection panel for the query of channel groups. Faster plotting.
- Implementation of the *ui.json* for Peak Finder and Octree Mesh Creation. Parameters can be re-loaded in the Notebook app from the Project Panel.
- Object and Data selection widgets now use the entities uuid as reference.
  The name of Objects is displayed with the parent group to facilitate the sorting/selection.
- Zonge (8 Hz) added to the list of airborne EM systems.

.. note::
    Upcoming with Geoscience ANALYST Pro (v3.3.1), the ui.json will be used to launch
    the application directly from a live workspace with drag+drop to the viewport.

        .. image:: applications/images/GA_pro_octree.gif


Release 0.4.1 (2021-04-07)
--------------------------

- Add unit tests on utils
- Hot fixes for docs


Release 0.4.0 (2021-03-10)
--------------------------

**New Application** - Isosurface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New application for the creation of 3D isosurface values around any type of object.


Dependencies
^^^^^^^^^^^^

In order to protect the applications from breaking between releases, we will now fix the version
of most third-party packages. Users will be notified if an update of the requirements is needed.


**Application Updates**

- Fix numpy warnings for deprecated conversion to numpy.float
- Fix issues with gdal and osr imports
- Stability updates to the Peak Finder app.
- Data selection by line ID now accepts ReferencedData type.
- Add base run test for all apps


Release 0.3.0 (2021-02-11)
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


**Application Updates**

- Coordinate Transformation app now supports (and relies) on Well-Known-Text strings. ESRI and EPSG codes are also allowed.
- New option for Surface Creation of horizons (2.5D surfaces)
- New plotting utilities for Surfaces, Points and BlockModel objects using Plotly
- New EM systems added: Hummingbird, GEOTEM 75 Hz, SkyTEM 306 (HM/LM), QUESTEM

Previous Releases
-----------------

Release 0.2.10 (2021-01-28)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

(Hot fix)

-  Broken dependencies (thanks Joel)


Release 0.2.9 (2021-01-19)
^^^^^^^^^^^^^^^^^^^^^^^^^^

(Minor Release)

- Allow integer data types
- Update data dependencies for ezdxf
- Begin adding skeleton for unit tests (0% coverage)



Release 0.2.6 (2020-12-14)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Update KMeans clustering application for reference data.


Release 0.2.5
^^^^^^^^^^^^^

- Upper/lower bound values added to the KMeans clustering application.
- Fix for documentation
