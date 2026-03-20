"""Electron density integration example for molgrid library."""

import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint
from molgrid import Atom, Molecule, MolecularGrid

# Create water molecule using PySCF
mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='6-31g', verbose=0)

# Run DFT calculation
mf = dft.RKS(mol)
mf.kernel()
rdm1 = mf.make_rdm1()

# Create molecule for molgrid
atoms = [
    Atom('O', [0.0, 0.0, 0.0]),
    Atom('H', [0.0, -0.757, 0.587]),
    Atom('H', [0.0, 0.757, 0.587])
]
water = Molecule(atoms)

# Create molecular grid with pruning
mol_grid = MolecularGrid(water, nshells=32, nangpts=110)

print(f"Molecular grid:")
print(f"Number of points: {len(mol_grid)}")

# Evaluate molecular orbitals on the grid
ao = numint.eval_ao(mol, mol_grid.coords, deriv=0)

# Calculate electron density
rho = numint.eval_rho(mol, ao, rdm1, xctype='LDA')

# Integrate electron density
number_of_electrons = np.sum(rho * mol_grid.weights)

print(f"\nIntegration results:")
print(f"Calculated number of electrons: {number_of_electrons:.6f}")
print(f"Expected number of electrons: 10.000000")
print(f"Error: {abs(number_of_electrons - 10.0):.6f}")

# Calculate electron integral for each atomic grid
print("\nElectron integrals for each atomic grid:")
total_electrons = 0
for i, atomic_grid in enumerate(mol_grid):
    # Evaluate AO on atomic grid
    ao_atom = numint.eval_ao(mol, atomic_grid.coords, deriv=0)
    # Calculate density on atomic grid
    rho_atom = numint.eval_rho(mol, ao_atom, rdm1, xctype='LDA')
    # Integrate electron density on atomic grid
    atom_electrons = np.sum(rho_atom * atomic_grid.weights)
    total_electrons += atom_electrons
    print(f"Atom {i} ({atomic_grid.atom.symbol}): {atom_electrons:.6f} electrons")

print(f"\nTotal electrons from atomic grids: {total_electrons:.6f}")
print(f"Expected number of electrons: 10.000000")
print(f"Error: {abs(total_electrons - 10.0):.6f}")

print("\nElectron density integration example completed successfully!")