Release Notes
=============

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
