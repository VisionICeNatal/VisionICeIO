Developer Guide
===============

Setting up a Development Environment
-------------------------------------

Clone the repository and install in editable mode::

    git clone https://github.com/VisionICeNatal/VisionICeIO.git
    cd VisionICeIO
    python -m venv .venv
    source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -e ".[test,docs,dev]"

Running the Test Suite
----------------------

Run all tests::

    pytest

Run with coverage::

    pytest --cov=visioniceio --cov-report=term-missing

Testing without real data files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most I/O functions require binary DLTG files which are not included in the
repository. For unit tests, create synthetic DLTG files programmatically::

    import struct, tempfile, numpy as np

    def make_dltg_file(suffix=".swa", n_datasets=1, data_bytes=b""):
        """Create a minimal valid DLTG file for testing."""
        header = b"DTLG" + b"\x00\x00\x00\x01"
        header += struct.pack(">I", n_datasets)
        desc = suffix.lstrip(".").encode("ascii")
        header += struct.pack(">I", 12 + 4 + len(desc))
        header += struct.pack(">H", len(desc)) + desc
        offset_table = struct.pack(">I", len(header) + 512)
        offset_table += b"\x00" * (127 * 4)
        dim = struct.pack(">i", len(data_bytes))
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        f.write(header + offset_table + dim + data_bytes)
        f.close()
        return f.name

Linting and Formatting
----------------------

The project uses `ruff <https://docs.astral.sh/ruff/>`_::

    ruff check .              # lint
    ruff check --fix .        # auto-fix
    ruff format .             # format
    ruff format --check .     # check only

Building the Documentation
--------------------------

Build locally::

    cd docs
    make html

Output is in ``docs/_build/html/``. Docs are deployed to GitHub Pages
via CI on pushes to ``main``.

DLTG Binary Format Notes
-------------------------

When adding support for new DLTG file types, refer to
``docs/data_format.md`` for the container layout. Key points:

- 4-byte magic ``DTLG`` (big-endian).
- Offset tables hold 128 entries each; the last entry chains to the next table.
- String datasets store a 4-byte signed length prefix followed by raw bytes.
- Numeric datasets store a 4-byte dimension prefix followed by data.
- All multi-byte integers are big-endian.

Release Checklist
-----------------

Prerequisites (one-time setup)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a `PyPI <https://pypi.org/>`_ account (and optionally a
   `TestPyPI <https://test.pypi.org/>`_ account).
2. Create an API token on PyPI (or TestPyPI) under *Account Settings →
   API tokens*.
3. Store the token so ``twine`` can use it.  The recommended way is a
   ``~/.pypirc`` file::

       [pypi]
       username = __token__
       password = pypi-<your-token>

       [testpypi]
       repository = https://test.pypi.org/legacy/
       username = __token__
       password = pypi-<your-test-token>

   Alternatively you can pass ``--username __token__ --password <token>``
   to ``twine upload`` directly.

Publishing a new version
^^^^^^^^^^^^^^^^^^^^^^^^

1. Update the version string in ``pyproject.toml`` (``version``).
   ``visioniceio.__version__`` and ``docs/conf.py release`` are
   derived from it automatically — no other files need editing.

2. Make sure the test suite passes::

       pytest

3. Remove any previous build artefacts and build the sdist + wheel::

       rm -rf dist/
       python -m build

4. Validate the package metadata::

       twine check dist/*

5. *(Recommended)* Upload to **TestPyPI** first to catch rendering or
   metadata issues before publishing for real::

       twine upload --repository testpypi dist/*

   Then verify the package page at
   ``https://test.pypi.org/project/visioniceio/`` and optionally test
   installation in a fresh virtual environment::

       pip install --index-url https://test.pypi.org/simple/ visioniceio

6. Upload to **PyPI**::

       twine upload dist/*

7. Tag the release and push the tag::

       git tag v0.x.y
       git push origin v0.x.y

8. Verify the published package at
   ``https://pypi.org/project/visioniceio/`` and confirm installation::

       pip install visioniceio
