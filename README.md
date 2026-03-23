# MolGrid

MolGrid: A Python library for generating real-space finite grids for DFT calculations.

## Overview

MolGrid is a lightweight Python library designed to generate high-quality real-space integration grids for density functional theory (DFT) calculations. It provides a flexible framework for creating atomic and molecular grids, with support for Becke grid partition to optimize computational efficiency.

## Features

- **Atomic Grids**: Generate radial and angular grids for individual atoms
- **Molecular Grids**: Combine atomic grids to form molecular integration grids
- **Becke Partition**: Optimize grid points using Becke atomic partition weights
- **Flexible Parameters**: Customize radial shells and angular points for each atom
- **Integration Support**: Calculate electron density and other properties on the grid
- **PySCF Integration**: Compatible with PySCF for electronic structure calculations

## Installation

### From Source

```bash
git clone https://github.com/yourusername/molgrid.git
cd molgrid
pip install -e .
```

### Dependencies

- NumPy
- PySCF (optional, for electron density calculations)

## Basic Usage

### Creating a Molecular Grid

```python
from molgrid import Atom, Molecule, MolecularGrid

# Create atoms
oxygen = Atom('O', [0.0, 0.0, 0.0])
hydrogen1 = Atom('H', [0.0, -0.757, 0.587])
hydrogen2 = Atom('H', [0.0, 0.757, 0.587])

# Create molecule
water = Molecule([oxygen, hydrogen1, hydrogen2])

# Create molecular grid with Becke partition
mol_grid = MolecularGrid(water, nshells=32, nangpts=110, partition_method='becke')

print(f"Molecular grid points: {len(mol_grid)}")
print(f"Coordinates shape: {mol_grid.coords.shape}")
print(f"Weights shape: {mol_grid.weights.shape}")
```

### Calculating Electron Density

```python
from pyscf import gto, dft
from pyscf.dft import numint

# Create PySCF molecule
mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='6-31g', verbose=0)

# Run DFT calculation
mf = dft.RKS(mol)
mf.kernel()
rdm1 = mf.make_rdm1()

# Evaluate molecular orbitals on the grid
ao = numint.eval_ao(mol, mol_grid.coords, deriv=0)

# Calculate electron density
rho = numint.eval_rho(mol, ao, rdm1, xctype='LDA')

# Integrate electron density
number_of_electrons = np.sum(rho * mol_grid.weights)
print(f"Number of electrons: {number_of_electrons:.6f}")
```

## Examples

The `examples` directory contains several example scripts:

- `01_basic_usage.py`: Basic usage of the library
- `02_electron_density.py`: Electron density calculation and integration

To run the examples:

```bash
python examples/01_basic_usage.py
python examples/02_electron_density.py
```

## Testing

The library includes a comprehensive test suite in the `test` directory. To run the tests:

```bash
pytest test/ -v
```

## Documentation

Full documentation is available at [https://zhaoyilin.github.io/molgrid/](https://zhaoyilin.github.io/molgrid/).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.