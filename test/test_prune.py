"""Tests for Becke class in prune module."""

import pytest
import numpy as np
from molgrid.molecule import Atom, Molecule
from molgrid.prune import Becke


class TestBeckeInit:
    """Test Becke class initialization."""

    def test_init_with_molecule(self):
        """Test Becke initialization with Molecule."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [1.0, 0.0, 0.0]),
        ])
        becke = Becke(mol)

        assert becke.molecule == mol
        assert len(becke.atoms) == 2

    def test_init_atoms_extraction(self):
        """Test that Becke extracts atoms from molecule."""
        atoms = [
            Atom('C', [0.0, 0.0, 0.0]),
            Atom('O', [1.0, 0.0, 0.0]),
            Atom('H', [0.0, 1.0, 0.0]),
        ]
        mol = Molecule(atoms=atoms)
        becke = Becke(mol)

        assert len(becke.atoms) == 3
        assert becke.atoms[0].symbol == 'C'
        assert becke.atoms[1].symbol == 'O'
        assert becke.atoms[2].symbol == 'H'


class TestBeckeWeighting:
    """Test Becke weight function."""

    @pytest.fixture
    def h2_becke(self):
        """H2 molecule with Becke."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [1.4, 0.0, 0.0]),
        ])
        return Becke(mol)

    @pytest.fixture
    def single_atom_becke(self):
        """Single atom Becke (no weighting)."""
        mol = Molecule([Atom('H', [0.0, 0.0, 0.0])])
        return Becke(mol)

    def test_weight_function_output_shape(self, h2_becke):
        """Test that weight function returns list with correct length."""
        point = np.array([0.5, 0.0, 0.0])
        weights = h2_becke._weight_function(point)

        assert isinstance(weights, list)
        assert len(weights) == 2  # Two atoms

    def test_weight_function_single_atom(self, single_atom_becke):
        """Test weight function for single atom."""
        point = np.array([0.0, 0.0, 0.0])
        weights = single_atom_becke._weight_function(point)

        # For single atom, weight should be 1.0 (normalized)
        assert len(weights) == 1
        assert weights[0] == pytest.approx(1.0)

    def test_weights_normalize_to_one(self, h2_becke):
        """Test that weights sum to 1 (normalization)."""
        points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.7, 0.0, 0.0]),
            np.array([1.4, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.5]),
        ]

        for point in points:
            weights = h2_becke._weight_function(point)
            total = sum(weights)
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_weights_positive(self, h2_becke):
        """Test that all weights are non-negative."""
        for x in np.linspace(-2, 2, 5):
            point = np.array([x, 0.0, 0.0])
            weights = h2_becke._weight_function(point)
            assert all(w >= 0 for w in weights)

    def test_midpoint_weights(self):
        """Test weights at midpoint between atoms."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [2.0, 0.0, 0.0]),
        ])
        becke = Becke(mol)

        # At midpoint, weights should be roughly equal
        midpoint = np.array([1.0, 0.0, 0.0])
        weights = becke._weight_function(midpoint)

        assert weights[0] == pytest.approx(weights[1], rel=1e-10)

    def test_heteroatomic_flag(self, h2_becke):
        """Test heteroatomic correction parameter."""
        point = np.array([0.7, 0.0, 0.0])

        # With heteroatomic correction
        weights_hetero = h2_becke._weight_function(point, do_becke_hetero=True)

        # Without heteroatomic correction  
        weights_no_hetero = h2_becke._weight_function(point, do_becke_hetero=False)

        # Both should be valid (non-negative and sum to 1)
        assert all(w >= 0 for w in weights_hetero)
        assert all(w >= 0 for w in weights_no_hetero)
        assert sum(weights_hetero) == pytest.approx(1.0)
        assert sum(weights_no_hetero) == pytest.approx(1.0)


class TestVoronoiPolyhedron:
    """Test Voronoi polyhedron construction."""

    @pytest.fixture
    def binary_molecule_becke(self):
        """Binary molecule for testing."""
        mol = Molecule([
            Atom('C', [0.0, 0.0, 0.0]),
            Atom('O', [1.2, 0.0, 0.0]),
        ])
        return Becke(mol)

    def test_voronoi_output_shape(self, binary_molecule_becke):
        """Test that voronoi function returns scalar."""
        point = np.array([0.6, 0.0, 0.0])
        atom = binary_molecule_becke.atoms[0]

        result = binary_molecule_becke._voronoi_polyhedron(atom, point)

        assert isinstance(result, (float, np.floating, int, np.integer))

    def test_voronoi_positive(self, binary_molecule_becke):
        """Test that Voronoi weights are non-negative."""
        atom = binary_molecule_becke.atoms[0]

        for x in np.linspace(-2, 2, 5):
            point = np.array([x, 0.0, 0.0])
            result = binary_molecule_becke._voronoi_polyhedron(atom, point)
            assert result >= 0

    def test_voronoi_heteroatomic_parameter(self, binary_molecule_becke):
        """Test heteroatomic correction in Voronoi."""
        point = np.array([0.6, 0.0, 0.0])
        atom = binary_molecule_becke.atoms[0]

        result_hetero = binary_molecule_becke._voronoi_polyhedron(
            atom, point, do_becke_hetero=True
        )
        result_no_hetero = binary_molecule_becke._voronoi_polyhedron(
            atom, point, do_becke_hetero=False
        )

        # Both should be non-negative
        assert result_hetero >= 0
        assert result_no_hetero >= 0


class TestSmoothingFunction:
    """Test Becke smoothing function."""

    def test_smoothing_range(self):
        """Test that smoothing function is in valid range."""
        mol = Molecule([Atom('H', [0.0, 0.0, 0.0])])
        becke = Becke(mol)

        for mu in np.linspace(-1, 1, 11):
            result = becke._smoothing_function(mu)
            assert 0 <= result <= 1

    def test_smoothing_boundary(self):
        """Test smoothing function at boundaries."""
        mol = Molecule([Atom('H', [0.0, 0.0, 0.0])])
        becke = Becke(mol)

        # At mu = -1, should be close to 1
        result_minus_one = becke._smoothing_function(-1.0)
        assert result_minus_one > 0.9

        # At mu = 1, should be close to 0
        result_plus_one = becke._smoothing_function(1.0)
        assert result_plus_one < 0.1


class TestPolynomialFunctions:
    """Test polynomial functions used in smoothing."""

    def test_polynomial_p(self):
        """Test polynomial p function."""
        mol = Molecule([Atom('H', [0.0, 0.0, 0.0])])
        becke = Becke(mol)

        # Test some values
        assert becke._polynomial_p(0.0) == pytest.approx(0.0)
        assert becke._polynomial_p(1.0) == pytest.approx(1.0)

    def test_polynomial_fk(self):
        """Test polynomial fk function."""
        mol = Molecule([Atom('H', [0.0, 0.0, 0.0])])
        becke = Becke(mol)

        # Test that fk iteration returns reasonable values
        for mu in [-0.5, 0.0, 0.5]:
            result = becke._polynomial_fk(mu, k=3)
            assert -1.5 <= result <= 1.5


class TestBeckeConsistency:
    """Test consistency of Becke partitioning."""

    def test_partition_sum_unity(self):
        """Test that Becke partitioning is globally normalized."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('F', [1.0, 0.0, 0.0]),
        ])
        becke = Becke(mol)

        # Sample many points
        np.random.seed(42)
        for _ in range(10):
            point = np.random.randn(3) * 2  # Random point
            weights = becke._weight_function(point)
            assert sum(weights) == pytest.approx(1.0, abs=1e-10)

    def test_reproducibility(self):
        """Test that same molecule gives same weights."""
        atoms = [Atom('C', [0.0, 0.0, 0.0]), Atom('N', [1.0, 0.0, 0.0])]

        becke1 = Becke(Molecule(atoms=atoms))
        becke2 = Becke(Molecule(atoms=atoms))

        point = np.array([0.5, 0.5, 0.5])
        weights1 = becke1._weight_function(point)
        weights2 = becke2._weight_function(point)

        np.testing.assert_allclose(weights1, weights2)


class TestBeckeForMultiAtoms:
    """Test Becke partitioning for molecules with >2 atoms."""

    def test_three_atoms(self):
        """Test Becke for 3-atom molecule."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('O', [1.0, 0.0, 0.0]),
            Atom('H', [0.0, 1.0, 0.0]),
        ])
        becke = Becke(mol)

        point = np.array([0.3, 0.3, 0.0])
        weights = becke._weight_function(point)

        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights)

    def test_many_atoms(self):
        """Test Becke for molecule with many atoms."""
        atoms = [Atom('H', [float(i), 0.0, 0.0]) for i in range(5)]
        mol = Molecule(atoms=atoms)
        becke = Becke(mol)

        point = np.array([2.0, 0.5, 0.0])
        weights = becke._weight_function(point)

        assert len(weights) == 5
        assert sum(weights) == pytest.approx(1.0)
        assert all(w >= 0 for w in weights)


class TestBeckeNumerical:
    """Test numerical properties of Becke."""

    def test_weight_convergence(self):
        """Test that weights converge to correct values."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [5.0, 0.0, 0.0]),  # Far apart
        ])
        becke = Becke(mol)

        # Very close to first atom
        point_near_first = np.array([0.01, 0.0, 0.0])
        weights = becke._weight_function(point_near_first)
        assert weights[0] > 0.9  # Should be mostly on first atom

        # Very close to second atom
        point_near_second = np.array([4.99, 0.0, 0.0])
        weights = becke._weight_function(point_near_second)
        assert weights[1] > 0.9  # Should be mostly on second atom

    def test_weight_symmetry(self):
        """Test weight symmetry for symmetric molecules."""
        mol = Molecule([
            Atom('H', [-1.0, 0.0, 0.0]),
            Atom('H', [1.0, 0.0, 0.0]),
        ])
        becke = Becke(mol)

        # At origin, should be symmetric
        point = np.array([0.0, 0.0, 0.0])
        weights = becke._weight_function(point)
        assert weights[0] == pytest.approx(weights[1])

        # Off-axis should maintain relative symmetry
        point = np.array([0.0, 0.5, 0.5])
        weights = becke._weight_function(point)
        assert weights[0] == pytest.approx(weights[1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
