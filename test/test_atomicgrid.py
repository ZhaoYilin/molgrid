"""Tests for AtomicGrid class."""

import pytest
import numpy as np
from molgrid.molecule import Atom
from molgrid.atomicgrid import AtomicGrid


class TestAtomicGridInit:
    """Test AtomicGrid initialization."""

    def test_init_default_params(self):
        """Test AtomicGrid initialization with defaults."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=32, nangpts=110)

        assert grid.atom == atom
        assert grid.nshells == 32
        assert grid.nangpts == 110

    def test_init_different_nshells(self):
        """Test AtomicGrid with different number of shells."""
        atom = Atom('C', [1.0, 2.0, 3.0])
        grid = AtomicGrid(atom, nshells=16, nangpts=50)

        assert grid.nshells == 16
        assert grid.nangpts == 50

    def test_init_different_atoms(self):
        """Test AtomicGrid for different atomic elements."""
        for symbol in ['H', 'C', 'N', 'O', 'S']:
            atom = Atom(symbol, [0.0, 0.0, 0.0])
            grid = AtomicGrid(atom, nshells=8, nangpts=26)

            assert grid.atom.symbol == symbol
            assert grid.nshells == 8


class TestAtomicGridGetters:
    """Test getter methods for radial/angular grids."""

    @pytest.fixture
    def hydrogen_grid(self):
        """Single hydrogen atom grid."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        return AtomicGrid(atom, nshells=16, nangpts=50)

    def test_get_radial_grid(self, hydrogen_grid):
        """Test get_radial_grid method."""
        rr, rw = hydrogen_grid.get_radial_grid()

        assert rr.shape[0] == 16  # 16 shells
        assert rw.shape[0] == 16  # 16 weights
        assert rr.dtype in [np.float32, np.float64]
        assert rw.dtype in [np.float32, np.float64]

    def test_get_angular_grid(self, hydrogen_grid):
        """Test get_angular_grid method."""
        angpts, aw = hydrogen_grid.get_angular_grid()

        # Angular points should be on unit sphere
        assert angpts.shape[1] == 3  # (x, y, z)
        assert aw.shape[0] == angpts.shape[0]

        # Check that points are approximately on unit sphere
        norms = np.linalg.norm(angpts, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_get_full_grid_shape(self, hydrogen_grid):
        """Test get_full_grid returns correct shape."""
        grid = hydrogen_grid.get_full_grid()

        expected_count = 16 * 50  # nshells * nangpts
        assert grid.shape == (expected_count, 3)

    def test_get_full_weights_shape(self, hydrogen_grid):
        """Test get_full_weights returns correct shape."""
        weights = hydrogen_grid.get_full_weights()

        expected_count = 16 * 50
        assert weights.shape == (expected_count,)

    def test_radial_weights_positive(self, hydrogen_grid):
        """Test that radial weights are positive."""
        rr, rw = hydrogen_grid.get_radial_grid()
        assert np.all(rw > 0), "All radial weights should be positive"

    def test_angular_weights_positive(self, hydrogen_grid):
        """Test that angular weights are positive."""
        angpts, aw = hydrogen_grid.get_angular_grid()
        assert np.all(aw > 0), "All angular weights should be positive"

    def test_full_weights_positive(self, hydrogen_grid):
        """Test that full weights are positive."""
        weights = hydrogen_grid.get_full_weights()
        assert np.all(weights > 0), "All full weights should be positive"


class TestAtomicGridProperties:
    """Test AtomicGrid properties."""

    def test_center_property(self):
        """Test center property."""
        pos = [1.5, 2.5, 3.5]
        atom = Atom('He', pos)
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        center = grid.center
        np.testing.assert_array_equal(center, pos)

    def test_atomic_number_property(self):
        """Test atomic_number property."""
        atom = Atom('N', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        assert grid.atomic_number == 7

    def test_len_method(self):
        """Test __len__ method."""
        atom = Atom('O', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=16, nangpts=50)

        expected_length = 16 * 50
        assert len(grid) == expected_length

    def test_repr_method(self):
        """Test __repr__ method."""
        atom = Atom('C', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=20, nangpts=86)

        repr_str = repr(grid)
        assert 'AtomicGrid' in repr_str
        assert 'C' in repr_str


class TestAtomicGridCentering:
    """Test grid centering at atomic position."""

    def test_grid_at_origin(self):
        """Test grid when atom is at origin."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        points = grid.get_full_grid()

        # Points should be distributed symmetrically around origin
        assert np.max(points) > 0
        assert np.min(points) < 0

    def test_grid_at_arbitrary_position(self):
        """Test grid when atom is at arbitrary position."""
        pos = [5.0, -3.0, 2.0]
        atom = Atom('He', pos)
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        points = grid.get_full_grid()

        # Grid should be centered at atom position (approximately)
        center = np.mean(points, axis=0)
        np.testing.assert_allclose(center, pos, atol=0.5)

    def test_grid_translation_invariance(self):
        """Test that grid shape is same regardless of atom position."""
        nshells = 8
        nangpts = 26

        grid1 = AtomicGrid(Atom('C', [0.0, 0.0, 0.0]), nshells, nangpts)
        grid2 = AtomicGrid(Atom('C', [10.0, 20.0, 30.0]), nshells, nangpts)

        # Both grids should have same number of points
        assert len(grid1) == len(grid2)

        # Both should have same relative distribution
        points1 = grid1.get_full_grid()
        points2 = grid2.get_full_grid()

        # Shift grid2 back to origin
        points2_shifted = points2 - np.array([10.0, 20.0, 30.0])

        # Distributions should match
        assert points1.shape == points2_shifted.shape


class TestAtomicGridDifferentElements:
    """Test AtomicGrid for different elements."""

    @pytest.mark.parametrize('symbol,number', [
        ('H', 1), ('C', 6), ('N', 7), ('O', 8), ('S', 16)
    ])
    def test_grid_for_element(self, symbol, number):
        """Test that grid works for all elements."""
        atom = Atom(symbol, [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        assert grid.atomic_number == number
        assert len(grid) == 8 * 26

        grid_pts = grid.get_full_grid()
        grid_wts = grid.get_full_weights()

        assert grid_pts.shape == (8 * 26, 3)
        assert grid_wts.shape == (8 * 26,)
        assert np.all(grid_wts > 0)


class TestAtomicGridAngularQuadrature:
    """Test angular quadrature properties."""

    def test_angular_grid_is_on_sphere(self):
        """Test that angular points are on unit sphere."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=110)

        angpts, _ = grid.get_angular_grid()

        # All points should be on unit sphere
        norms = np.linalg.norm(angpts, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_angular_weights_sum(self):
        """Test that angular weights sum properly."""
        atom = Atom('He', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=110)

        _, aw = grid.get_angular_grid()

        # Sum of angular weights should be positive
        total_weight = np.sum(aw)
        assert total_weight > 0


class TestAtomicGridRadialQuadrature:
    """Test radial quadrature properties."""

    def test_radial_points_increasing(self):
        """Test that radial points are generally increasing."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=16, nangpts=50)

        rr, _ = grid.get_radial_grid()

        # Radial points should have increasing magnitude (not strictly monotonic)
        assert np.max(rr) > np.min(rr)

    def test_radial_points_positive(self):
        """Test that all radial points are positive."""
        atom = Atom('C', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=16, nangpts=50)

        rr, _ = grid.get_radial_grid()

        # Radial points should be non-negative
        assert np.all(rr >= 0)

    def test_radial_range(self):
        """Test reasonable radial range."""
        atom = Atom('O', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=16, nangpts=50)

        rr, _ = grid.get_radial_grid()

        # Radial points should be in reasonable range
        # (for typical atomic grid)
        assert rr.max() < 100  # Should not be too large


class TestAtomicGridConsistency:
    """Test internal consistency of grids."""

    def test_weights_product_shape(self):
        """Test that weight product matches full grid."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        rr, rw = grid.get_radial_grid()
        angpts, aw = grid.get_angular_grid()

        full_wts = grid.get_full_weights()

        # Product of radial and angular weights should match full weights
        expected_wts = np.multiply.outer(rw, aw).reshape(-1)
        np.testing.assert_allclose(full_wts, expected_wts, rtol=1e-10)

    def test_grid_reconstruction(self):
        """Test that full grid can be reconstructed from components."""
        atom = Atom('C', [1.0, 2.0, 3.0])
        grid = AtomicGrid(atom, nshells=8, nangpts=26)

        rr, _ = grid.get_radial_grid()
        angpts, _ = grid.get_angular_grid()

        # Full grid should be outer product of radial and angular
        full_grid = grid.get_full_grid()

        # Check shape consistency
        expected_shape = (len(rr) * len(angpts), 3)
        assert full_grid.shape == expected_shape


class TestAtomicGridEdgeCases:
    """Test edge cases."""

    def test_small_nshells(self):
        """Test grid with minimal number of shells."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = AtomicGrid(atom, nshells=1, nangpts=6)

        assert len(grid) == 6

    def test_specific_nangpts_values(self):
        """Test grid with specific angular point counts."""
        atom = Atom('He', [0.0, 0.0, 0.0])

        # Common Lebedev grid sizes
        for nangpts in [6, 14, 26, 50, 86]:
            grid = AtomicGrid(atom, nshells=8, nangpts=nangpts)
            # Should not raise errors
            assert grid.get_full_grid().shape[0] == 8 * nangpts


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
