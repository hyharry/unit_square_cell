.. unit_square_cell_docu documentation master file, created by
   sphinx-quickstart on Sun Jan 24 22:37:03 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Unit Cell Module's documentation!
=================================================

This Unit Cell Module consists of three files, *cell_geom.py*, *cell_material.py*, and *cell_computation.py*. The main computation part of Unit Cell is done in *cell_computation.py*, while the other two make up the assisting part of the computation, where materal and geometry (mesh, inclusions) are defined within these two files.

*cell_geom.py*
  Mesh and inclusions are generated through this file. It is needed for the modelling of composites. The inclusions defined in this files are circle (sphere) and rectangular (brick).

*cell_material.py*
  In this file a generic creation of material is covered. In the current version of the module the material concerning about plasticity and viscosity is not realized.

*cell_computation.py*
  After specifying geometry and materials in the modelling, the unit cell in micro state is analyzed in the context of this file. Main functionalities of this file ranges from solving fluctuation, post-processing and visualization of the results.

**Contents:**

.. toctree::
   :maxdepth: 2
   
   Manual on cell_geom.py
   Manual on cell_material.py
   Manual on cell_computation.py
   source file


Indices and tables
==================
   
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
