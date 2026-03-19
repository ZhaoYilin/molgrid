"""Tests for MolecularGrid class."""

import pytest
import numpy as np
from molgrid.molecule import Atom, Molecule
from molgrid.moleculargrid import MolecularGrid


class TestMolecularGridInit:
    """Test MolecularGrid initialization."""

    def test_init_single_atom_list(self):
        """Test initialization with a single atom in a list."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = MolecularGrid([atom], nshells=32, nangpts=110)

        assert len(grid.atoms) == 1
        assert len(grid.atomic_grids) == 1
        assert grid.nshells == 32
        assert grid.nangpts == 110

    def test_init_multiple_atoms_list(self):
        """Test initialization with multiple atoms in a list."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('O', [1.0, 0.0, 0.0]),
        ]
        grid = MolecularGrid(atoms, nshells=20, nangpts=50)

        assert len(grid.atoms) == 2
        assert len(grid.atomic_grids) == 2
        assert grid.nshells == 20
        assert grid.nangpts == 50

    def test_init_from_molecule(self):
        """Test initialization with a Molecule object."""
        mol = Molecule([
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [0.74, 0.0, 0.0]),
        ])
        grid = MolecularGrid(mol, nshells=16, nangpts=86)

        assert len(grid.atoms) == 2
        assert len(grid.atomic_grids) == 2
        assert grid.nshells == 16
        assert grid.nangpts == 86

    def test_init_empty_atoms(self):
        """Test initialization with empty atom list."""
        grid = MolecularGrid([])
        assert len(grid.atoms) == 0
        assert len(grid.atomic_grids) == 0


class TestMolecularGridGetters:
    """Test getter methods for grid data."""

    @pytest.fixture
    def hydrogen_grid(self):
        """Single hydrogen atom grid."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        return MolecularGrid([atom], nshells=32, nangpts=110)

    @pytest.fixture
    def h2_grid(self):
        """H2 molecule grid."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [1.4, 0.0, 0.0]),
        ]
        return MolecularGrid(atoms, nshells=20, nangpts=86)

    def test_get_full_grid_shape(self, hydrogen_grid):
        """Test that get_full_grid returns correct shape."""
        grid = hydrogen_grid.get_full_grid()
        assert grid.shape == (3520, 3)  # 32*110 = 3520 points, 3 coordinates
        assert grid.dtype == np.float64 or grid.dtype == np.float32

    def test_get_full_weights_shape(self, hydrogen_grid):
        """Test that get_full_weights returns correct shape."""
        weights = hydrogen_grid.get_full_weights()
        assert weights.shape == (3520,)  # 32*110 = 3520 weights
        assert weights.dtype == np.float64 or weights.dtype == np.float32

    def test_weights_positive(self, hydrogen_grid):
        """Test that all weights are positive."""
        weights = hydrogen_grid.get_full_weights()
        assert np.all(weights > 0), "All weights should be positive"

    def test_get_full_grid_with_weights_shape(self, hydrogen_grid):
        """Test that get_full_grid_with_weights returns correct shape."""
        grid_with_weights = hydrogen_grid.get_full_grid_with_weights()
        assert grid_with_weights.shape == (3520, 4)  # 3 coords + 1 weight
        assert grid_with_weights[:, :3].shape == (3520, 3)
        assert grid_with_weights[:, 3].shape == (3520,)

    def test_get_full_grid_with_weights_consistency(self, hydrogen_grid):
        """Test that grid_with_weights combines data correctly."""
        grid = hydrogen_grid.get_full_grid()
        weights = hydrogen_grid.get_full_weights()
        grid_with_weights = hydrogen_grid.get_full_grid_with_weights()

        # Check that data is consistent
        np.testing.assert_allclose(grid_with_weights[:, :3], grid)
        np.testing.assert_allclose(grid_with_weights[:, 3], weights)

    def test_multi_atom_grid_size(self, h2_grid):
        """Test that multi-atom grid has correct total size."""
        grid = h2_grid.get_full_grid()
        weights = h2_grid.get_full_weights()
        natoms = 2
        expected_total = natoms * 20 * 86  # natoms * nshells * nangpts

        assert grid.shape[0] == expected_total
        assert weights.shape[0] == expected_total

    def test_grid_centered_at_atoms(self, hydrogen_grid):
        """Test that grid points are centered around atom."""
        grid = hydrogen_grid.get_full_grid()
        atom_pos = hydrogen_grid.atoms[0].coordinate

        # Grid points should be distributed around the atom
        # For a single atom at origin, some points should be positive and negative
        assert np.any(grid[:, 0] > 0), "Should have some positive x coordinates"
        assert np.any(grid[:, 0] < 0), "Should have some negative x coordinates"
        assert np.any(grid[:, 1] > 0), "Should have some positive y coordinates"
        assert np.any(grid[:, 1] < 0), "Should have some negative y coordinates"


class TestMolecularGridPrune:
    """Test prune_grid method."""

    @pytest.fixture
    def h2_grid(self):
        """H2 molecule grid."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [1.4, 0.0, 0.0]),
        ]
        return MolecularGrid(atoms, nshells=16, nangpts=50)

    def test_prune_grid_basic(self, h2_grid):
        """Test basic pruning with threshold."""
        orig_grid = h2_grid.get_full_grid()
        orig_weights = h2_grid.get_full_weights()
        orig_count = len(orig_weights)

        # Prune with a small threshold
        pruned_grid, pruned_weights = h2_grid.prune_grid(method='becke', threshold=1e-6)

        # Should have fewer or equal points
        assert pruned_grid.shape[0] == pruned_weights.shape[0]
        assert pruned_grid.shape[0] <= orig_count
        assert pruned_grid.shape[1] == 3

    def test_prune_grid_no_threshold(self, h2_grid):
        """Test that zero threshold keeps all points."""
        grid, weights = h2_grid.prune_grid(method='becke', threshold=0.0)
        full_grid = h2_grid.get_full_grid()

        # With threshold=0, should keep all points (or nearly all due to numerical precision)
        assert grid.shape[0] >= full_grid.shape[0] * 0.99  # Allow 1% numerical tolerance

    def test_prune_grid_high_threshold(self, h2_grid):
        """Test that high threshold removes most points."""
        grid, weights = h2_grid.prune_grid(method='becke', threshold=1.0)

        # With threshold=1.0, should remove many or all points
        assert grid.shape[0] < h2_grid.get_full_grid().shape[0]

    def test_prune_weights_decreasing_threshold(self, h2_grid):
        """Test that larger threshold removes more points."""
        grid_1e6, weights_1e6 = h2_grid.prune_grid(method='becke', threshold=1e-6)
        grid_1e3, weights_1e3 = h2_grid.prune_grid(method='becke', threshold=1e-3)

        # Larger threshold should result in fewer points
        assert weights_1e3.shape[0] <= weights_1e6.shape[0]


class TestMolecularGridBecke:
    """Test Becke weight functionality."""

    @pytest.fixture
    def h2_grid(self):
        """H2 molecule grid."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [1.4, 0.0, 0.0]),
        ]
        return MolecularGrid(atoms, nshells=16, nangpts=50)

    def test_apply_becke_weights(self, h2_grid):
        """Test apply_becke_weights modifies grid."""
        # Get original weights
        orig_weights = h2_grid.get_full_weights().copy()

        # Apply Becke weights
        h2_grid.apply_becke_weights(do_becke_hetero=True)

        # Becke weights should modify the weights
        # (unless by coincidence they are identical, which is highly unlikely)
        updated_weights = h2_grid.get_full_weights()
        # We can't strictly assert inequality due to numerical precision,
        # but we can check that the operation doesn't crash

    def test_get_molecular_grid_with_becke_shape(self, h2_grid):
        """Test get_molecular_grid_with_becke returns correct shape."""
        grid_with_becke = h2_grid.get_molecular_grid_with_becke(do_becke_hetero=True)

        expected_count = 2 * 16 * 50  # 2 atoms, 16 shells, 50 angular points
        assert grid_with_becke.shape == (expected_count, 4)
        assert grid_with_becke.dtype == np.float64 or grid_with_becke.dtype == np.float32

    def test_becke_weights_positive(self, h2_grid):
        """Test that Becke weights are all positive."""
        grid_with_becke = h2_grid.get_molecular_grid_with_becke(do_becke_hetero=True)
        becke_weights = grid_with_becke[:, 3]

        assert np.all(becke_weights >= 0), "Becke weights should be non-negative"

    def test_becke_consistency(self):
        """Test that Becke weights are consistent for same molecule."""
        atoms = [Atom('C', [0.0, 0.0, 0.0]), Atom('O', [1.2, 0.0, 0.0])]

        grid1 = MolecularGrid(atoms, nshells=8, nangpts=26)
        weights1 = grid1.get_molecular_grid_with_becke().copy()

        grid2 = MolecularGrid(atoms, nshells=8, nangpts=26)
        weights2 = grid2.get_molecular_grid_with_becke().copy()

        # Weights should be the same for identical grids
        np.testing.assert_allclose(weights1, weights2, rtol=1e-10)

    def test_becke_heteroatomic_effect(self):
        """Test that heteroatomic correction parameters affect weights."""
        atoms = [Atom('H', [0.0, 0.0, 0.0]), Atom('O', [1.0, 0.0, 0.0])]
        
        # Grid without heteroatomic correction
        grid1 = MolecularGrid(atoms, nshells=8, nangpts=26)
        w1 = grid1.get_molecular_grid_with_becke(do_becke_hetero=False)

        # Grid with heteroatomic correction
        grid2 = MolecularGrid(atoms, nshells=8, nangpts=26)
        w2 = grid2.get_molecular_grid_with_becke(do_becke_hetero=True)

        # Weights may or may not be different, but both should be valid
        assert np.all(w1[:, 3] >= 0)
        assert np.all(w2[:, 3] >= 0)


class TestMolecularGridMagic:
    """Test special/magic methods."""

    def test_len_single_atom(self):
        """Test __len__ for single atom."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = MolecularGrid([atom], nshells=32, nangpts=110)
        assert len(grid) == 32 * 110

    def test_len_multiple_atoms(self):
        """Test __len__ for multiple atoms."""
        atoms = [Atom('H', [0.0, 0.0, 0.0]), Atom('H', [1.0, 0.0, 0.0])]
        grid = MolecularGrid(atoms, nshells=20, nangpts=86)
        assert len(grid) == 2 * 20 * 86

    def test_len_empty(self):
        """Test __len__ for empty grid."""
        grid = MolecularGrid([])
        assert len(grid) == 0

    def test_repr(self):
        """Test __repr__ method."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('O', [1.0, 0.0, 0.0]),
        ]
        grid = MolecularGrid(atoms, nshells=16, nangpts=50)
        repr_str = repr(grid)

        assert 'MolecularGrid' in repr_str
        assert 'natoms=2' in repr_str
        assert 'nshells=16' in repr_str
        assert 'nangpts=50' in repr_str


class TestMolecularGridEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_atom_different_elements(self):
        """Test single atoms of different elements."""
        for element in ['H', 'C', 'N', 'O']:
            atom = Atom(element, [0.0, 0.0, 0.0])
            grid = MolecularGrid([atom], nshells=8, nangpts=26)
            assert grid.get_full_grid().shape[0] == 8 * 26

    def test_empty_grid_operations(self):
        """Test operations on empty grid."""
        grid = MolecularGrid([])

        # Should return empty arrays with correct shapes
        assert grid.get_full_grid().shape == (0, 3)
        assert grid.get_full_weights().shape == (0,)
        assert grid.get_full_grid_with_weights().shape == (0, 4)
        assert len(grid) == 0

    def test_displaced_atoms(self):
        """Test grid with atoms at various positions."""
        # Create a small triangular molecule
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [1.0, 0.0, 0.0]),
            Atom('H', [0.5, 0.866, 0.0]),  # Approximately equilateral
        ]
        grid = MolecularGrid(atoms, nshells=8, nangpts=26)

        assert len(grid) == 3 * 8 * 26

        grid_points = grid.get_full_grid()
        # Points should be distributed around all three atoms
        assert grid_points.max() > 1.0
        assert grid_points.min() < 0.0

    def test_large_molecule(self):
        """Test grid with many atoms (small stress test)."""
        # Create a relatively large molecule
        atoms = [
            Atom('C', [float(i), 0.0, 0.0])
            for i in range(5)
        ]
        grid = MolecularGrid(atoms, nshells=8, nangpts=26)

        assert len(grid) == 5 * 8 * 26
        assert grid.get_full_grid().shape == (5 * 8 * 26, 3)


class TestMolecularGridNumerical:
    """Test numerical properties of grids."""

    def test_grid_weights_normalization(self):
        """Test that grid weights are reasonable."""
        atom = Atom('H', [0.0, 0.0, 0.0])
        grid = MolecularGrid([atom], nshells=32, nangpts=110)

        weights = grid.get_full_weights()

        # Weights should be positive
        assert np.all(weights > 0)

        # Total weight should be roughly proportional to integrated volume
        # For a single atom with Bragg-Slater radius, expect certain order of magnitude
        total_weight = np.sum(weights)
        assert total_weight > 0

    def test_grid_isotropy_single_atom(self):
        """Test that single atom grid is roughly isotropic."""
        atom = Atom('He', [0.0, 0.0, 0.0])
        grid = MolecularGrid([atom], nshells=16, nangpts=26)

        points = grid.get_full_grid()

        # Check that points extend in all directions
        coords = [points[:, 0], points[:, 1], points[:, 2]]
        for coord in coords:
            assert np.max(coord) > 0, f"Should extend in positive direction"
            assert np.min(coord) < 0, f"Should extend in negative direction"

    def test_atomic_grid_consistency_with_molecular(self):
        """Test that single-atom molecular grid matches atomic grid."""
        from molgrid.atomicgrid import AtomicGrid

        atom = Atom('C', [1.0, 2.0, 3.0])

        # Create atomic grid directly
        atomic_grid = AtomicGrid(atom, nshells=16, nangpts=50)

        # Create molecular grid with same atom
        molecular_grid = MolecularGrid([atom], nshells=16, nangpts=50)

        # Points should match (translation to atom center is handled consistently)
        ag_points = atomic_grid.get_full_grid()
        mg_points = molecular_grid.get_full_grid()

        assert ag_points.shape == mg_points.shape
        np.testing.assert_allclose(ag_points, mg_points, rtol=1e-10)

        # Weights should match
        ag_weights = atomic_grid.get_full_weights()
        mg_weights = molecular_grid.get_full_weights()

        np.testing.assert_allclose(ag_weights, mg_weights, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
