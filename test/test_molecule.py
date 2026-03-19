"""Tests for molecule module (Atom, Element, Molecule classes)."""

import pytest
import numpy as np
from molgrid.molecule import Element, Atom, Molecule


class TestElement:
    """Test Element class."""

    def test_init_by_symbol(self):
        """Test Element initialization by symbol."""
        h = Element(symbol='H')
        assert h.symbol == 'H'
        assert h.number == 1
        assert h.mass == pytest.approx(1.007975, rel=1e-4)

    def test_init_by_number(self):
        """Test Element initialization by atomic number."""
        c = Element(number=6)
        assert c.symbol == 'C'
        assert c.number == 6
        assert c.mass == pytest.approx(12.0106, rel=1e-4)

    def test_element_properties(self):
        """Test that Element has expected periodic table properties."""
        o = Element(symbol='O')
        assert hasattr(o, 'symbol')
        assert hasattr(o, 'number')
        assert hasattr(o, 'mass')
        assert hasattr(o, 'cov_radius_slater')

    def test_element_invalid_symbol(self):
        """Test that invalid symbol raises error."""
        with pytest.raises(ValueError):
            Element(symbol='Xx')

    def test_element_invalid_number(self):
        """Test that invalid element number raises error."""
        with pytest.raises(ValueError):
            Element(number=200)

    def test_element_missing_both_args(self):
        """Test that missing both symbol and number raises error."""
        with pytest.raises(ValueError):
            Element()

    def test_element_repr(self):
        """Test Element __repr__."""
        n = Element(symbol='N')
        repr_str = repr(n)
        assert 'Element' in repr_str
        assert 'N' in repr_str


class TestAtom:
    """Test Atom class."""

    def test_init_with_element_object(self):
        """Test Atom initialization with Element object."""
        elem = Element(symbol='H')
        atom = Atom(elem, [0.0, 0.0, 0.0])
        assert atom.symbol == 'H'
        assert atom.number == 1

    def test_init_with_symbol(self):
        """Test Atom initialization with element symbol."""
        atom = Atom('C', [1.0, 2.0, 3.0])
        assert atom.symbol == 'C'
        assert atom.number == 6
        np.testing.assert_array_equal(atom.coordinate, [1.0, 2.0, 3.0])

    def test_init_with_atomic_number(self):
        """Test Atom initialization with atomic number."""
        atom = Atom(8, [-1.0, 0.5, 2.3])
        assert atom.symbol == 'O'
        assert atom.number == 8

    def test_coordinate_assignment(self):
        """Test assign_coordinate method."""
        atom = Atom('He', [0.0, 0.0, 0.0])
        atom.assign_coordinate([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(atom.coordinate, [1.0, 1.0, 1.0])

    def test_coordinate_invalid_type(self):
        """Test that non-list coordinate raises error."""
        atom = Atom('He', [0.0, 0.0, 0.0])
        with pytest.raises(TypeError):
            atom.assign_coordinate(np.array([1, 2, 3]))

    def test_coordinate_invalid_length(self):
        """Test that non-3-element coordinate raises error."""
        atom = Atom('He', [0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            atom.assign_coordinate([1.0, 2.0])

    def test_atom_equality(self):
        """Test atom equality check."""
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('H', [0.0, 0.0, 0.0])
        assert atom1 == atom2

    def test_atom_inequality_different_element(self):
        """Test that atoms with different elements are not equal."""
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('C', [0.0, 0.0, 0.0])
        assert atom1 != atom2

    def test_atom_inequality_different_position(self):
        """Test that atoms at different positions are not equal."""
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('H', [1.0, 0.0, 0.0])
        assert atom1 != atom2

    def test_atom_distance_to(self):
        """Test distance_to method."""
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('H', [3.0, 4.0, 0.0])
        distance = atom1.distance_to(atom2)
        assert distance == pytest.approx(5.0)

    def test_atom_repr(self):
        """Test Atom __repr__."""
        atom = Atom('N', [1.0, 2.0, 3.0])
        repr_str = repr(atom)
        assert 'Atom' in repr_str
        assert 'N' in repr_str

    def test_atom_various_elements(self):
        """Test atoms of different elements."""
        for symbol in ['H', 'C', 'N', 'O', 'S', 'P']:
            atom = Atom(symbol, [0.0, 0.0, 0.0])
            assert atom.symbol == symbol


class TestMolecule:
    """Test Molecule class."""

    def test_init_empty(self):
        """Test Molecule initialization without atoms."""
        mol = Molecule()
        assert len(mol) == 0
        assert len(mol.atoms) == 0

    def test_init_with_name(self):
        """Test Molecule initialization with name."""
        mol = Molecule(name='Water')
        assert mol.name == 'Water'

    def test_init_with_charge_and_multiplicity(self):
        """Test Molecule initialization with charge and multiplicity."""
        mol = Molecule(charge=1, multiplicity=2)
        assert mol.charge == 1
        assert mol.multiplicity == 2

    def test_add_atom(self):
        """Test adding atoms to molecule."""
        mol = Molecule()
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('H', [0.74, 0.0, 0.0])

        mol.add_atom(atom1)
        mol.add_atom(atom2)

        assert len(mol) == 2
        assert mol.atoms[0] == atom1
        assert mol.atoms[1] == atom2

    def test_add_invalid_atom(self):
        """Test that adding non-Atom raises error."""
        mol = Molecule()
        with pytest.raises(TypeError):
            mol.add_atom("not an atom")

    def test_remove_atom(self):
        """Test removing atoms from molecule."""
        mol = Molecule()
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('O', [1.0, 0.0, 0.0])

        mol.add_atom(atom1)
        mol.add_atom(atom2)

        removed = mol.remove_atom(0)
        assert removed == atom1
        assert len(mol) == 1

    def test_remove_atom_invalid_index(self):
        """Test that removing invalid index raises error."""
        mol = Molecule()
        atom = Atom('H', [0.0, 0.0, 0.0])
        mol.add_atom(atom)

        with pytest.raises(IndexError):
            mol.remove_atom(5)

    def test_get_atom(self):
        """Test getting atoms by index."""
        mol = Molecule()
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('O', [1.0, 0.0, 0.0])

        mol.add_atom(atom1)
        mol.add_atom(atom2)

        assert mol.get_atom(0) == atom1
        assert mol.get_atom(1) == atom2

    def test_get_coordinates(self):
        """Test getting all atomic coordinates."""
        mol = Molecule()
        mol.add_atom(Atom('H', [0.0, 0.0, 0.0]))
        mol.add_atom(Atom('O', [1.0, 1.0, 1.0]))

        coords = mol.get_coordinates()
        assert coords.shape == (2, 3)
        np.testing.assert_array_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(coords[1], [1.0, 1.0, 1.0])

    def test_get_symbols(self):
        """Test getting all atomic symbols."""
        mol = Molecule()
        mol.add_atom(Atom('C', [0.0, 0.0, 0.0]))
        mol.add_atom(Atom('O', [1.0, 0.0, 0.0]))
        mol.add_atom(Atom('H', [0.0, 1.0, 0.0]))

        symbols = mol.get_symbols()
        assert symbols == ['C', 'O', 'H']

    def test_get_atomic_numbers(self):
        """Test getting all atomic numbers."""
        mol = Molecule()
        mol.add_atom(Atom('H', [0.0, 0.0, 0.0]))
        mol.add_atom(Atom('O', [1.0, 0.0, 0.0]))

        numbers = mol.get_atomic_numbers()
        assert numbers == [1, 8]

    def test_get_masses(self):
        """Test getting all atomic masses."""
        mol = Molecule()
        mol.add_atom(Atom('H', [0.0, 0.0, 0.0]))
        mol.add_atom(Atom('C', [1.0, 0.0, 0.0]))

        masses = mol.get_masses()
        assert len(masses) == 2
        assert masses[0] == pytest.approx(1.007975, rel=1e-4)
        assert masses[1] == pytest.approx(12.0106, rel=1e-4)

    def test_molecule_indexing(self):
        """Test molecule indexing with __getitem__."""
        mol = Molecule()
        atom1 = Atom('H', [0.0, 0.0, 0.0])
        atom2 = Atom('O', [1.0, 0.0, 0.0])

        mol.add_atom(atom1)
        mol.add_atom(atom2)

        assert mol[0] == atom1
        assert mol[1] == atom2

    def test_molecule_iteration(self):
        """Test molecule iteration with __iter__."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('O', [1.0, 0.0, 0.0]),
            Atom('H', [2.0, 0.0, 0.0]),
        ]
        mol = Molecule(atoms=atoms)

        count = 0
        for atom in mol:
            assert atom in atoms
            count += 1

        assert count == 3

    def test_molecule_len(self):
        """Test molecule length."""
        mol = Molecule()
        assert len(mol) == 0

        mol.add_atom(Atom('H', [0.0, 0.0, 0.0]))
        assert len(mol) == 1

        mol.add_atom(Atom('O', [1.0, 0.0, 0.0]))
        assert len(mol) == 2

    def test_h2_molecule(self):
        """Test H2 molecule."""
        atoms = [
            Atom('H', [0.0, 0.0, 0.0]),
            Atom('H', [0.74, 0.0, 0.0]),
        ]
        mol = Molecule(atoms=atoms, name='H2')

        assert mol.name == 'H2'
        assert len(mol) == 2
        assert mol.get_symbols() == ['H', 'H']
        assert np.allclose(mol.atoms[0].distance_to(mol.atoms[1]), 0.74)

    def test_water_molecule(self):
        """Test water molecule."""
        atoms = [
            Atom('O', [0.0, 0.0, 0.0]),
            Atom('H', [0.96, 0.0, 0.0]),
            Atom('H', [-0.24, 0.93, 0.0]),
        ]
        mol = Molecule(atoms=atoms, name='H2O', charge=0, multiplicity=1)

        assert mol.name == 'H2O'
        assert len(mol) == 3
        assert mol.get_atomic_numbers() == [8, 1, 1]
        assert mol.charge == 0
        assert mol.multiplicity == 1


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_very_large_coordinates(self):
        """Test atoms with very large coordinates."""
        atom = Atom('He', [1e6, 1e6, 1e6])
        assert atom.coordinate[0] == pytest.approx(1e6)

    def test_negative_coordinates(self):
        """Test atoms with negative coordinates."""
        atom = Atom('Ne', [-1.5, -2.5, -3.5])
        np.testing.assert_array_equal(atom.coordinate, [-1.5, -2.5, -3.5])

    def test_zero_coordinates(self):
        """Test atoms at origin."""
        atom = Atom('Ar', [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(atom.coordinate, [0.0, 0.0, 0.0])

    def test_molecule_all_same_element(self):
        """Test molecule with all atoms of same element."""
        atoms = [Atom('H', [float(i), 0.0, 0.0]) for i in range(4)]
        mol = Molecule(atoms=atoms)

        assert len(mol) == 4
        assert all(s == 'H' for s in mol.get_symbols())

    def test_molecule_many_atoms(self):
        """Test molecule with many atoms (stress test)."""
        atoms = []
        symbols = ['H', 'C', 'N', 'O']
        for i in range(20):
            symbol = symbols[i % len(symbols)]
            atoms.append(Atom(symbol, [float(i), 0.0, 0.0]))

        mol = Molecule(atoms=atoms)
        assert len(mol) == 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
