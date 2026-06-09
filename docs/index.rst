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

Institutions & Funding
----------------------

.. raw:: html

   <p style="margin:1.5rem 0;text-align:center;">
     <img src="_static/logo_ice.png" alt="Brain Institute (ICe), UFRN" style="height:55px;width:auto;margin:0 1.25rem;vertical-align:middle;" />
     <a href="https://uni-goettingen.de/en/608362.html" target="_blank"><img src="_static/logo_cidbn.jpg" alt="CIDBN / University of Göttingen" style="height:55px;width:auto;margin:0 1.25rem;vertical-align:middle;" /></a>
     <a href="https://www.gov.br/cnpq/" target="_blank"><img src="_static/logo_cnpq.jpg" alt="CNPq — Conselho Nacional de Desenvolvimento Científico e Tecnológico" style="height:55px;width:auto;margin:0 1.25rem;vertical-align:middle;" /></a>
     <img src="_static/logo_inct_neurocomp.png" alt="INCT-NeuroComp" style="height:55px;width:auto;margin:0 1.25rem;vertical-align:middle;" />
     <a href="https://www.gov.br/capes/" target="_blank"><img src="_static/logo_capes.png" alt="CAPES" style="height:55px;width:auto;margin:0 1.25rem;vertical-align:middle;" /></a>
   </p>

This software is co-developed by the Schmidt Lab at the Brain Institute of the Federal University of Rio Grande do Norte (UFRN, Natal, Brazil) and the Göttingen Campus Institute for Dynamics of Biological Networks (`CIDBN <https://uni-goettingen.de/en/608362.html>`_).
This work is partially supported by the
Brazilian National Council for Scientific and Technological Development (`CNPq <https://www.gov.br/cnpq/>`_; grants 445096/2024-1 and 408389/2024-9, INCT-NeuroComp)
and by the Coordination for the Improvement of Higher Education Personnel (`CAPES <https://www.gov.br/capes/>`_)
through the PROBRAL programme (grant 88881.986124/2024-01).
