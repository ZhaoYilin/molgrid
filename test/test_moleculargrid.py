import pytest
import numpy as np
from molgrid.molecule import Atom
from molgrid.moleculargrid import MolecularGrid


def test_moleculargrid_without_prune():
    """Test MolecularGrid class without prune"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    
    # Create molecular grid without pruning
    grid = MolecularGrid(atoms, nshells=10, nangpts=110, prune_method=None)
    original_size = len(grid)
    # Ensure grid size matches expected value
    assert original_size == 3 * 10 * 110
    
def test_moleculargrid_with_prune():
    """Test MolecularGrid class with prune"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    
    # Create molecular grid
    grid = MolecularGrid(atoms, nshells=10, nangpts=110, prune_method='becke')
   
    # Test pruning reduces the number of grid points
    assert len(grid) <= 3 * 10 * 110  # 3 atoms * 10 shells * 110 angular points
    assert len(grid) > 0  # Ensure there is at least one grid point

def test_moleculargrid_iteration():
    """Test MolecularGrid class iteration"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    
    # Create molecular grid
    grid = MolecularGrid(atoms, nshells=10, nangpts=110)
    
    # Test iteration
    atomic_grids = list(grid)
    assert len(atomic_grids) == 3
    assert atomic_grids[0].atom.symbol == 'O'
    assert atomic_grids[1].atom.symbol == 'H'
    assert atomic_grids[2].atom.symbol == 'H'

def test_moleculargrid_with_different_shells():
    """Test MolecularGrid with different radial shells"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    
    # Specify different number of radial shells for each atom
    nshells = [10, 5, 5]
    nangpts = 110
    
    # Create molecular grid
    grid = MolecularGrid(atoms, nshells=nshells, nangpts=nangpts, prune_method=None)
    
    # Calculate expected grid size
    expected_size = 10 * 110 + 5 * 110 + 5 * 110
    assert len(grid) == expected_size