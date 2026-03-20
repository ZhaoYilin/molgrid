"""Grid pruning example for molgrid library."""

import numpy as np
from molgrid import Atom, Molecule, MolecularGrid

# Create water molecule
atoms = [
    Atom('O', [0.0, 0.0, 0.0]),
    Atom('H', [0.0, -0.757, 0.587]),
    Atom('H', [0.0, 0.757, 0.587])
]
water = Molecule(atoms)

# Create molecular grid without pruning
mol_grid_no_prune = MolecularGrid(water, nshells=20, nangpts=110, prune_method=None)
print(f"Molecular grid without pruning:")
print(f"Number of points: {len(mol_grid_no_prune)}")

# Print atomic grid sizes without pruning
print("\nAtomic grid sizes without pruning:")
for i, atomic_grid in enumerate(mol_grid_no_prune):
    print(f"Atom {i} ({atomic_grid.atom.symbol}): {len(atomic_grid)} points")

# Create molecular grid with Becke pruning
print("\nCreating grid with Becke pruning...")
mol_grid_pruned = MolecularGrid(water, nshells=20, nangpts=110, prune_method='becke', prune_threshold=1e-3)
print(f"Molecular grid with pruning:")
print(f"Number of points: {len(mol_grid_pruned)}")

# Print atomic grid sizes with pruning
print("\nAtomic grid sizes with pruning:")
for i, atomic_grid in enumerate(mol_grid_pruned):
    print(f"Atom {i} ({atomic_grid.atom.symbol}): {len(atomic_grid)} points")

# Calculate pruning efficiency
original_size = 3 * 20 * 110  # 3 atoms * 20 shells * 110 angular points
pruned_size = len(mol_grid_pruned)
efficiency = (1 - pruned_size / original_size) * 100
print(f"\nPruning efficiency: {efficiency:.2f}%")
print("Grid pruning example completed successfully!")