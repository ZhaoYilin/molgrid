=========
MolGrid
=========

**MolGrid** is a Python library for constructing and manipulating molecular integration grids. It provides tools for generating atomic grids and combining them into molecular grids with support for Becke partitioning schemes.

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg


   
Key Features
=============

- **Flexible Grid Generation**: Generate atomic and molecular integration grids using state-of-the-art quadrature rules
- **Becke Partitioning**: Apply Becke weight partitioning schemes for accurate molecular grid weight distributions
- **Multiple Quadrature Rules**: Support for Lebedev (angular) and Gauss-Chebyshev (radial) quadrature
- **Pure Python**: Easy to install and use with standard scientific Python tools

Quick Start
===========

.. code-block:: python

    from molgrid import Molecule, AtomicGrid, MolecularGrid
    import numpy as np

    # Create a molecule
    atoms = [
    Atom('H', [0.0, 0.0, 0.0]),
    Atom('O', [1.0, 0.0, 0.00])
        ]
    mol = Molecule(atoms)

    # Combine into molecular grid with Becke partitioning
    mol_grid = MolecularGrid(mol, nshells=75, nangpts=110, partition_method='becke')

    # Access grid data
    grid_points = mol_grid.coords  # Shape: (N, 3)
    weights = mol_grid.weights   # Shape: (N,)

About
======

MolGrid was developed to provide researchers with an efficient, flexible tool for constructing integration grids in quantum chemistry calculations. It emphasizes code clarity and usability while maintaining computational efficiency.

Contents
=========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   notebook

.. toctree::
   :maxdepth: 2
   :caption: API
   :glob:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
