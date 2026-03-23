import pytest
import numpy as np
from molgrid.quadrature import Lebedev, GaussChebychev


def test_lebedev_creation():
    """Test Lebedev class creation"""
    lebedev = Lebedev()
    assert isinstance(lebedev, Lebedev)

def test_lebedev_get_npoints_list():
    """Test Lebedev class get_npoints_list method"""
    lebedev = Lebedev()
    npoints_list = lebedev.get_npoints_list()
    assert isinstance(npoints_list, list)
    assert len(npoints_list) > 0
    assert all(isinstance(n, int) for n in npoints_list)

def test_lebedev_get_degrees_list():
    """Test Lebedev class get_degrees_list method"""
    lebedev = Lebedev()
    degrees_list = lebedev.get_degrees_list()
    assert isinstance(degrees_list, list)
    assert len(degrees_list) > 0
    assert all(isinstance(d, int) for d in degrees_list)

def test_lebedev_get():
    """Test Lebedev class get method"""
    lebedev = Lebedev()
    
    # Test getting Lebedev points for different degrees
    for degree in [3, 5, 7, 9]:
        coords, weights = lebedev.get(degree, coord='cartesian')
        assert coords.shape[1] == 3  # Cartesian coordinates are 3-dimensional
        assert len(coords) == len(weights)
        assert np.isclose(np.sum(weights), 4*np.pi)  # Weights should sum to 4\pi

def test_gauss_chebychev_creation():
    """Test GaussChebychev class creation"""
    chebychev = GaussChebychev()
    assert isinstance(chebychev, GaussChebychev)

def test_gauss_chebychev_semi_infinite():
    """Test GaussChebychev class semi_infinite method"""
    chebychev = GaussChebychev()
    
    # Test generating Gauss-Chebyshev points for semi-infinite interval
    r_scale = 1.0
    npoints = 10
    coords, weights = chebychev.semi_infinite(r_scale, npoints)
    
    assert len(coords) == npoints
    assert len(weights) == npoints
    assert all(c > 0 for c in coords)  # Points in semi-infinite interval should be positive
    assert all(w > 0 for w in weights)  # Weights should be positive

def test_gauss_chebychev_finite():
    """Test GaussChebychev class finite method"""
    chebychev = GaussChebychev()
    
    # Test generating Gauss-Chebyshev points for finite interval
    npoints = 10
    coords, weights = chebychev.finite(npoints)
    
    assert len(coords) == npoints
    assert len(weights) == npoints
    assert all(w > 0 for w in weights)  # Weights should be positive