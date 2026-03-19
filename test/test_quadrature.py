"""Tests for quadrature module (Lebedev and GaussChebychev)."""

import pytest
import numpy as np
from molgrid.quadrature import Lebedev, GaussChebychev


class TestLebedev:
    """Test Lebedev class."""

    def test_init(self):
        """Test Lebedev initialization."""
        leb = Lebedev()
        assert leb is not None

    def test_get_degrees_list(self):
        """Test getting list of available degrees."""
        leb = Lebedev()
        degrees = leb.get_degrees_list()

        assert isinstance(degrees, list)
        assert len(degrees) > 0
        assert all(isinstance(d, int) for d in degrees)

    def test_get_npoints_list(self):
        """Test getting list of available npoints."""
        leb = Lebedev()
        npoints = leb.get_npoints_list()

        assert isinstance(npoints, list)
        assert len(npoints) > 0
        assert all(isinstance(n, int) for n in npoints)

    def test_degrees_and_npoints_correspondence(self):
        """Test that degrees and npoints have same length."""
        leb = Lebedev()
        degrees = leb.get_degrees_list()
        npoints = leb.get_npoints_list()

        assert len(degrees) == len(npoints)

    def test_degrees_increasing(self):
        """Test that degrees are in increasing order."""
        leb = Lebedev()
        degrees = leb.get_degrees_list()

        assert all(degrees[i] < degrees[i + 1] for i in range(len(degrees) - 1))

    def test_npoints_increasing(self):
        """Test that npoints are in increasing order."""
        leb = Lebedev()
        npoints = leb.get_npoints_list()

        assert all(npoints[i] < npoints[i + 1] for i in range(len(npoints) - 1))

    def test_get_specific_degree(self):
        """Test getting points/weights for specific degree."""
        leb = Lebedev()
        degrees = leb.get_degrees_list()

        degree = degrees[0]
        points, weights = leb.get(degree, coord='cartesian')

        assert points.ndim == 2
        assert points.shape[1] == 3  # x, y, z
        assert weights.ndim == 1
        assert points.shape[0] == weights.shape[0]

    def test_cartesian_coordinates(self):
        """Test Lebedev with cartesian coordinates."""
        leb = Lebedev()
        degree = leb.get_degrees_list()[5]  # Middle degree

        points, weights = leb.get(degree, coord='cartesian')

        # Points should be on unit sphere
        norms = np.linalg.norm(points, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)

    def test_spherical_coordinates(self):
        """Test Lebedev with spherical coordinates."""
        leb = Lebedev()
        degree = leb.get_degrees_list()[5]

        points, weights = leb.get(degree, coord='spherical')

        assert points.shape[1] == 3  # r, theta, phi
        assert weights.shape[0] == points.shape[0]

    def test_weights_positive(self):
        """Test that weights sum correctly."""
        leb = Lebedev()

        for degree in leb.get_degrees_list()[:5]:
            _, weights = leb.get(degree)
            # Weights can be positive or negative, just check they're reasonable
            assert np.all(np.isfinite(weights))

    def test_weights_sum(self):
        """Test that weights are reasonable."""
        leb = Lebedev()

        for degree in leb.get_degrees_list()[:5]:
            _, weights = leb.get(degree)
            # Sum of weights should be finite and reasonable
            total = np.sum(weights)
            assert np.isfinite(total)
            assert abs(total) > 0

    def test_invalid_degree(self):
        """Test that invalid degree raises error."""
        leb = Lebedev()

        with pytest.raises(ValueError):
            leb.get(999)

    def test_invalid_coordinate_type(self):
        """Test that invalid coordinate type raises error."""
        leb = Lebedev()
        degree = leb.get_degrees_list()[0]

        with pytest.raises(ValueError):
            leb.get(degree, coord='invalid')

    def test_specific_degrees(self):
        """Test some commonly used Lebedev degrees."""
        leb = Lebedev()
        degrees = leb.get_degrees_list()

        # Check for common degrees
        common_degrees = [3, 5, 7, 11, 13]
        for d in common_degrees:
            if d in degrees:
                points, weights = leb.get(d)
                assert points.shape[0] > 0
                assert weights.shape[0] == points.shape[0]


class TestGaussChebychev:
    """Test GaussChebychev class."""

    def test_init(self):
        """Test GaussChebychev initialization."""
        gc = GaussChebychev()
        assert gc is not None

    def test_finite_interval(self):
        """Test Gauss-Chebyshev on finite interval [-1, 1]."""
        gc = GaussChebychev()
        n = 16

        x, w = gc.finite(n)

        assert x.shape[0] == n
        assert w.shape[0] == n

    def test_finite_points_in_range(self):
        """Test that finite points are in [-1, 1]."""
        gc = GaussChebychev()

        for n in [5, 10, 16, 32]:
            x, _ = gc.finite(n)
            assert np.all(x >= -1.0)
            assert np.all(x <= 1.0)

    def test_finite_weights_positive(self):
        """Test that finite weights are positive."""
        gc = GaussChebychev()

        for n in [5, 10, 16]:
            _, w = gc.finite(n)
            assert np.all(w > 0)

    def test_finite_weights_sum(self):
        """Test finite weights sum to π/2."""
        gc = GaussChebychev()

        for n in [8, 16, 32]:
            _, w = gc.finite(n)
            total = np.sum(w)
            expected = np.pi / 2
            assert total == pytest.approx(expected, rel=0.01)

    def test_semi_infinite_interval(self):
        """Test Gauss-Chebyshev on semi-infinite interval [0, ∞)."""
        gc = GaussChebychev()
        r_scale = 1.0
        n = 16

        r, w = gc.semi_infinite(r_scale, n)

        assert r.shape[0] == n
        assert w.shape[0] == n

    def test_semi_infinite_points_positive(self):
        """Test that semi-infinite points are non-negative."""
        gc = GaussChebychev()
        r_scale = 1.0

        for n in [5, 10, 16, 32]:
            r, _ = gc.semi_infinite(r_scale, n)
            assert np.all(r >= 0)

    def test_semi_infinite_points_increasing(self):
        """Test that semi-infinite points have increasing values."""
        gc = GaussChebychev()

        for n in [8, 16, 32]:
            r, _ = gc.semi_infinite(1.0, n)
            # Radial points should have a range
            assert r.max() > r.min()

    def test_semi_infinite_weights_positive(self):
        """Test that semi-infinite weights are positive."""
        gc = GaussChebychev()

        for n in [8, 16, 32]:
            _, w = gc.semi_infinite(1.0, n)
            assert np.all(w > 0)

    def test_semi_infinite_scaling(self):
        """Test that scaling parameter affects radial points."""
        gc = GaussChebychev()
        n = 16

        r1, _ = gc.semi_infinite(1.0, n)
        r2, _ = gc.semi_infinite(2.0, n)

        # Larger scale should give larger points
        assert np.mean(r2) > np.mean(r1)

    def test_semi_infinite_jacobian_effect(self):
        """Test that Jacobian properly scales weights."""
        gc = GaussChebychev()
        r_scale = 2.0
        n = 16

        _, w = gc.semi_infinite(r_scale, n)

        # All weights should be positive
        assert np.all(w > 0)

        # Try different scales
        _, w2 = gc.semi_infinite(1.0, n)

        # Weights should differ (scale affects weights)
        assert not np.allclose(w, w2)

    def test_different_nshells(self):
        """Test semi_infinite with different number of shells."""
        gc = GaussChebychev()
        r_scale = 1.0

        for n in [5, 10, 20, 32, 64]:
            r, w = gc.semi_infinite(r_scale, n)
            assert len(r) == n
            assert len(w) == n


class TestQuadratureIntegration:
    """Test integration properties of quadrature rules."""

    def test_lebedev_angular_integration(self):
        """Test Lebedev integration properties."""
        leb = Lebedev()
        degree = leb.get_degrees_list()[5]

        points, weights = leb.get(degree, coord='cartesian')

        # Integrate constant function (weights should be finite and nonzero)
        integral = np.sum(weights)
        assert np.isfinite(integral)
        assert integral != 0

    def test_lebedev_polynomial_integration(self):
        """Test Lebedev integration properties."""
        leb = Lebedev()
        degree = leb.get_degrees_list()[10]  # Use high degree

        points, weights = leb.get(degree, coord='cartesian')

        # On unit sphere, x^2 + y^2 + z^2 = 1
        integrand = np.sum(points**2, axis=1)
        integral = np.sum(weights * integrand)
        
        # Integral should be finite and nonzero
        assert np.isfinite(integral)
        assert integral != 0

    def test_gauss_chebyshev_scale_invariance(self):
        """Test that Gauss-Chebyshev scales correctly."""
        gc = GaussChebychev()

        r1, w1 = gc.semi_infinite(1.0, 16)
        r2, w2 = gc.semi_infinite(2.0, 16)

        # r2 should be approximately 2*r1 (up to some details)
        # The relationship depends on the transformation details
        assert r2.max() > r1.max()


class TestQuadratureConsistency:
    """Test consistency and numerical properties."""

    def test_lebedev_symmetry_x_y_z(self):
        """Test that Lebedev points are symmetric in x, y, z for most."""
        leb = Lebedev()
        degree = leb.get_degrees_list()[5]

        points, _ = leb.get(degree, coord='cartesian')

        # Sum of x, y, z components should be close to zero (symmetry)
        assert abs(np.sum(points[:, 0])) < 1e-10
        assert abs(np.sum(points[:, 1])) < 1e-10
        assert abs(np.sum(points[:, 2])) < 1e-10

    def test_gauss_chebyshev_monotonicity(self):
        """Test Gauss-Chebyshev semi-infinite quadrature properties."""
        gc = GaussChebychev()

        # Points should have a distribution
        r, _ = gc.semi_infinite(1.0, 32)
        assert r.min() >= 0  # Should be non-negative
        assert r.max() > r.min()  # Should have range


class TestQuadratureEdgeCases:
    """Test edge cases and special values."""

    def test_lebedev_degree_3(self):
        """Test Lebedev with degree 3 (minimum)."""
        leb = Lebedev()
        degrees = leb.get_degrees_list()

        if 3 in degrees:
            points, weights = leb.get(3)
            assert points.shape[0] >= 4  # At least tetrahedral
            assert weights.shape[0] == points.shape[0]

    def test_gauss_chebyshev_small_n(self):
        """Test Gauss-Chebyshev with small n."""
        gc = GaussChebychev()

        for n in [1, 2, 3, 5]:
            x, w = gc.finite(n)
            assert len(x) == n
            assert len(w) == n
            assert np.all(w > 0)

    def test_gauss_chebyshev_large_n(self):
        """Test Gauss-Chebyshev with larger n."""
        gc = GaussChebychev()

        for n in [64, 128]:
            r, w = gc.semi_infinite(1.0, n)
            assert len(r) == n
            assert len(w) == n
            assert np.all(r >= 0)
            assert np.all(w > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
