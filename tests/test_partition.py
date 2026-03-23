import pytest
import numpy as np
from molgrid.molecule import Atom, Molecule
from molgrid.partition import Becke


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
    """Test Becke class weight_function method"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test single point near oxygen atom (must be 2D array)
    point_near_o = np.array([[0.1, 0.1, 0.1]])
    weights = becke.weight_function(point_near_o)
    assert weights.shape == (1, 3)  # 1 point, 3 atoms
    assert weights[0, 0] > weights[0, 1]  # Oxygen should have the highest weight
    assert weights[0, 0] > weights[0, 2]
    assert np.isclose(np.sum(weights), 1.0)  # Weights should sum to 1
    
    # Test point near hydrogen atom
    point_near_h1 = np.array([[0.0, -0.7, 0.5]])
    weights = becke.weight_function(point_near_h1)
    assert weights.shape == (1, 3)
    assert weights[0, 1] > weights[0, 0]  # First hydrogen should have the highest weight
    assert weights[0, 1] > weights[0, 2]
    assert np.isclose(np.sum(weights), 1.0)

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
    
    # Test Voronoi polyhedron function with single point
    point = np.array([[0.1, 0.1, 0.1]])
    P = becke._voronoi_polyhedron(point)
    assert P.shape == (1, 3)  # 1 point, 3 atoms
    assert np.all(P >= 0)  # All weights should be non-negative

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

def test_compute_weights():
    """Test Becke class compute_weights method"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test single point near oxygen
    coords = np.array([[0.1, 0.1, 0.1]])
    weights = becke.weight_function(coords)
    
    assert weights.shape == (1, 3)  # 1 point, 3 atoms
    assert np.allclose(np.sum(weights, axis=1), 1.0)  # Weights should sum to 1
    assert weights[0, 0] > weights[0, 1]  # Oxygen should have highest weight
    assert weights[0, 0] > weights[0, 2]
    
    # Test multiple points
    coords = np.array([
        [0.1, 0.1, 0.1],  # Near oxygen
        [0.0, -0.7, 0.5],  # Near first hydrogen
        [0.0, 0.7, 0.5]    # Near second hydrogen
    ])
    weights = becke.weight_function(coords)
    
    assert weights.shape == (3, 3)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert weights[0, 0] > weights[0, 1]  # Point 0: oxygen dominant
    assert weights[1, 1] > weights[1, 0]  # Point 1: hydrogen 1 dominant
    assert weights[2, 2] > weights[2, 0]  # Point 2: hydrogen 2 dominant

def test_voronoi_polyhedron_vectorized():
    """Test Becke class _voronoi_polyhedron method with vectorized input"""
    # Create water molecule
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587]),
        Atom('H', [0.0, 0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test with multiple points
    coords = np.array([
        [0.1, 0.1, 0.1],
        [0.0, -0.7, 0.5],
        [0.0, 0.7, 0.5]
    ])
    
    P = becke._voronoi_polyhedron(coords)
    assert P.shape == (3, 3)  # 3 points, 3 atoms
    assert np.all(P >= 0)  # All weights should be non-negative

def test_hetero_correction():
    """Test Becke class _hetero_correction method"""
    # Create water molecule (hetero-nuclear)
    atoms = [
        Atom('O', [0.0, 0.0, 0.0]),
        Atom('H', [0.0, -0.757, 0.587])
    ]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    aij = becke._hetero_matrix()
    assert aij.shape == (2, 2)
    assert aij[0, 0] == 0.0  # Diagonal should be zero
    assert aij[1, 1] == 0.0
    
    # Test homo-nuclear molecule
    atoms = [Atom('O', [0.0, 0.0, 0.0]), Atom('O', [1.0, 0.0, 0.0])]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    aij = becke._hetero_matrix()
    assert np.all(aij == 0.0)  # Homo-nuclear should have zero correction

def test_smoothing_function_vectorized():
    """Test Becke class _smoothing_function method with vectorized input"""
    # Create water molecule
    atoms = [Atom('O', [0.0, 0.0, 0.0])]
    molecule = Molecule(atoms)
    becke = Becke(molecule)
    
    # Test with array input
    mu_values = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    s_values = becke._smoothing_function(mu_values)
    
    assert s_values.shape == (5,)
    assert s_values[0] == 1.0  # mu = -1.0 -> s = 1.0
    assert s_values[-1] == 0.0  # mu = 1.0 -> s = 0.0
    assert np.all(s_values >= 0) and np.all(s_values <= 1.0)