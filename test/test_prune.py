import pytest
import numpy as np
from molgrid.molecule import Atom, Molecule
from molgrid.prune import Becke


def test_becke_creation():
    """Test Becke class creation"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    
    # Create Becke object
    becke = Becke(molecule)
    assert becke.molecule == molecule

def test_becke_weight_function():
    """Test Becke class _weight_function method"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test point near oxygen atom
    point_near_o = [0.1, 0.1, 0.1]
    weights = becke._weight_function(point_near_o)
    assert len(weights) == 3
    assert weights[0] > weights[1]  # Oxygen should have the highest weight
    assert weights[0] > weights[2]
    assert np.isclose(sum(weights), 1.0)  # Weights should sum to 1
    
    # Test point near hydrogen atom
    point_near_h1 = [0.0, -0.7, 0.5]
    weights = becke._weight_function(point_near_h1)
    assert len(weights) == 3
    assert weights[1] > weights[0]  # First hydrogen should have the highest weight
    assert weights[1] > weights[2]
    assert np.isclose(sum(weights), 1.0)

def test_voronoi_polyhedron():
    """Test Becke class _voronoi_polyhedron method"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test Voronoi polyhedron function for oxygen atom
    point = [0.1, 0.1, 0.1]
    weight = becke._voronoi_polyhedron(atoms[0], point)
    assert weight > 0

def test_smoothing_function():
    """Test Becke class _smoothing_function method"""
    # Create water molecule
    atoms = [Atom('O', [0.0, 0.0, 0.0])]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test smoothing function with different input values
    assert becke._smoothing_function(1.0) == 0.0
    assert becke._smoothing_function(-1.0) == 1.0
    assert 0.0 < becke._smoothing_function(0.0) < 1.0

def test_polynomial_fk():
    """Test Becke class _polynomial_fk method"""
    # Create water molecule
    atoms = [Atom('O', [0.0, 0.0, 0.0])]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test polynomial function
    assert becke._polynomial_fk(1.0) == 1.0
    assert becke._polynomial_fk(-1.0) == -1.0
    assert becke._polynomial_fk(0.0) == 0.0

def test_polynomial_p():
    """Test Becke class _polynomial_p method"""
    # Create water molecule
    atoms = [Atom('O', [0.0, 0.0, 0.0])]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test polynomial p
    assert becke._polynomial_p(1.0) == 1.0
    assert becke._polynomial_p(-1.0) == -1.0
    assert becke._polynomial_p(0.0) == 0.0