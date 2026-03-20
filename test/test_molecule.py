import pytest
import numpy as np
from molgrid.molecule import Atom, Molecule


def test_atom_creation():
    """Test Atom class creation and basic properties"""
    # Test default parameters
    atom = Atom('O', [0.0, 0.0, 0.0])
    assert atom.symbol == 'O'
    assert np.array_equal(atom.coordinate, np.array([0.0, 0.0, 0.0]))
    
    # Test custom coordinates
    atom = Atom('H', [1.0, 2.0, 3.0])
    assert atom.symbol == 'H'
    assert np.array_equal(atom.coordinate, np.array([1.0, 2.0, 3.0]))

def test_atom_properties():
    """Test Atom class properties"""
    atom = Atom('O', [0.0, 0.0, 0.0])
    assert hasattr(atom, 'number')  # Atomic number
    assert hasattr(atom, 'cov_radius_slater')  # Slater covalent radius
    assert hasattr(atom, 'mass')  # Atomic mass

def test_molecule_creation():
    """Test Molecule class creation"""
    # Create atom list
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    
    # Create molecule
    molecule = Molecule(atoms)
    assert len(molecule) == 3
    assert molecule[0].symbol == 'O'
    assert molecule[1].symbol == 'H'
    assert molecule[2].symbol == 'H'

def test_molecule_iteration():
    """Test Molecule class iteration"""
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    
    symbols = [atom.symbol for atom in molecule]
    assert symbols == ['O', 'H']