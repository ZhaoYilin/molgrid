"""Basic usage example for molgrid library."""

import numpy as np
from molgrid import Atom, Molecule, AtomicGrid, MolecularGrid

# Create atoms
oxygen = Atom('O', [0.0, 0.0, 0.0])
hydrogen1 = Atom('H', [0.0, -0.757, 0.587])
hydrogen2 = Atom('H', [0.0, 0.757, 0.587])

# Create molecule
water = Molecule([oxygen, hydrogen1, hydrogen2])
print(f"Created molecule: {water}")
print(f"Number of atoms: {len(water)}")

# Create atomic grid for oxygen
oxygen_grid = AtomicGrid(oxygen, nshells=10, nangpts=110)
print(f"\nOxygen atomic grid:")
print(f"Number of points: {len(oxygen_grid)}")
print(f"Coordinates shape: {oxygen_grid.coords.shape}")
print(f"Weights shape: {oxygen_grid.weights.shape}")

# Create molecular grid
mol_grid = MolecularGrid(water, nshells=10, nangpts=110)
print(f"\nMolecular grid:")
print(f"Number of points: {len(mol_grid)}")
print(f"Coordinates shape: {mol_grid.coords.shape}")
print(f"Weights shape: {mol_grid.weights.shape}")

# Iterate through atomic grids
print("\nAtomic grids in molecular grid:")
for i, atomic_grid in enumerate(mol_grid):
    print(f"Atom {i} ({atomic_grid.atom.symbol}): {len(atomic_grid)} points")

print("\nBasic usage example completed successfully!")