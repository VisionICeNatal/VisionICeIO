VisionICeIO
============

I/O utilities for the ICe Vision Lab LabView recording system.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc

      Full reference of all public functions and the ``Experiment`` class.

   .. grid-item-card:: Developer Guide
      :link: developer
      :link-type: doc

      Set up your environment, run tests, lint, and build documentation.

   .. grid-item-card:: Binary Format Specification
      :link: data_format
      :link-type: doc

      Detailed layout of the DLTG container and the new headerless format.

   .. grid-item-card:: Changelog
      :link: CHANGELOG
      :link-type: doc

      Version history and release notes.

.. mermaid::

   flowchart TB
       FILES["Binary files<br/>.spike .swave .stim<br/>.analog .info"]
       EXP["Experiment<br/>load_from_dir()"]
       DS["xr.Dataset<br/>waveforms, spike_times,<br/>n_spikes, stim_label, lfp"]
       ZARR[".zarr store"]

       FILES -->|"parse binary files"| EXP
       EXP -->|"assemble into labelled arrays"| DS
       DS -->|"persist to disk"| ZARR

       style FILES fill:#f5f5f5,stroke:#999
       style EXP fill:#4a90d9,stroke:#2c5f8a,color:#fff
       style DS fill:#2ecc71,stroke:#1a9c54,color:#fff
       style ZARR fill:#f5f5f5,stroke:#999

.. toctree::
   :maxdepth: 2
   :hidden:

   api
   developer
   data_format
   CHANGELOG
