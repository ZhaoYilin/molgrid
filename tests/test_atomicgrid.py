import pytest
import numpy as np
from molgrid.molecule import Atom
from molgrid.atomicgrid import AtomicGrid


def test_atomicgrid_creation():
    """Test AtomicGrid class creation"""
    atom = Atom('O', [0.0, 0.0, 0.0])
    grid = AtomicGrid(atom, nshells=10, nangpts=110)
    
    assert grid.atom == atom
    assert grid.nshells == 10
    assert grid.nangpts == 110

def test_atomicgrid_build():
    """Test AtomicGrid class build method"""
    atom = Atom('O', [0.0, 0.0, 0.0])
    grid = AtomicGrid(atom, nshells=10, nangpts=110)
    
    # Test shape of coordinates and weights
    assert grid.coords.shape == (10 * 110, 3)
    assert grid.weights.shape == (10 * 110,)
    
    # Test that weights sum to a reasonable value (should be close to atomic number)
    weights_sum = np.sum(grid.weights)
    assert weights_sum > 0

def test_atomicgrid_properties():
    """Test AtomicGrid class properties"""
    atom = Atom('O', [1.0, 2.0, 3.0])
    grid = AtomicGrid(atom, nshells=5, nangpts=26)
    
    assert np.array_equal(grid.center, np.array([1.0, 2.0, 3.0]))
    assert grid.radial_coords.shape == (5,)
    assert grid.radial_weights.shape == (5,)
    assert grid.angular_coords.shape[1] == 3
    assert grid.angular_weights.shape[0] == 26

def test_atomicgrid_len():
    """Test AtomicGrid class __len__ method"""
    atom = Atom('O', [0.0, 0.0, 0.0])
    grid = AtomicGrid(atom, nshells=10, nangpts=110)
    assert len(grid) == 10 * 110