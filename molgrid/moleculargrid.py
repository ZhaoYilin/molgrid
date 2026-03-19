import numpy as np
from molgrid.atomicgrid import AtomicGrid
from molgrid.prune import Becke


class MolecularGrid:
    """Build a molecular integration grid by combining atomic grids.

    Each atom produces a radial/angular grid via :class:`~molgrid.atomicgrid.AtomicGrid`.
    Becke partition weights can be applied to distribute grid contributions across
    atoms.
    """

    def __init__(self, atoms, nshells=32, nangpts=110):
        """Initialize the MolecularGrid.

        Parameters
        ----------
        atoms : list or Molecule
            List of :class:`~molgrid.molecule.Atom` objects or a :class:`~molgrid.molecule.Molecule`.
        nshells : int, optional
            Number of radial shells for each atom (default: 32)
        nangpts : int, optional
            Number of angular points for each atom (default: 110)
        """
        # If a Molecule is provided, extract its atom list.
        if hasattr(atoms, 'atoms'):
            self.atoms = atoms.atoms
        else:
            # Allow any iterable of Atoms
            self.atoms = list(atoms)

        self.nshells = nshells
        self.nangpts = nangpts
        self.atomic_grids = [AtomicGrid(atom, nshells, nangpts) for atom in self.atoms]
        
        # Store grid points with weights for each atomic grid
        for agrid in self.atomic_grids:
            # Add weight column to points array (assuming points are [x, y, z])
            # If AtomicGrid stores points differently, adjust accordingly
            if not hasattr(agrid, 'points_with_weights'):
                points = agrid.get_full_grid()
                weights = agrid.get_full_weights()
                # Store as [x, y, z, weight] for each point
                agrid.points_with_weights = np.column_stack([points, weights])
        
    def get_full_grid(self):
        """Combine atomic grids into a full molecular grid (points only)."""
        if not self.atomic_grids:
            return np.zeros((0, 3))
        return np.vstack([agrid.get_full_grid() for agrid in self.atomic_grids])

    def get_full_weights(self):
        """Combine atomic weights into a full molecular weight array."""
        if not self.atomic_grids:
            return np.zeros((0,))
        return np.hstack([agrid.get_full_weights() for agrid in self.atomic_grids])

    def get_full_grid_with_weights(self):
        """Combine atomic grids into a full molecular grid with weights [x, y, z, weight]."""
        if not self.atomic_grids:
            return np.zeros((0, 4))
        full_data = []
        for agrid in self.atomic_grids:
            points = agrid.get_full_grid()
            weights = agrid.get_full_weights()
            full_data.append(np.column_stack([points, weights]))
        return np.vstack(full_data)
    
    def prune_grid(self, method='becke', threshold=1e-6):
        """Prune grid points with weights below a certain threshold.

        Parameters
        ----------
        method : str, optional
            Method to use when determining weights. Supported values are:
            - 'becke': use Becke partition weights (default)
            - any other value: use raw atomic grid weights
        threshold : float, optional
            Weight threshold for pruning points

        Returns
        -------
        tuple
            (filtered_points, filtered_weights)
        """
        if method == 'becke':
            grid_with_weights = self.get_molecular_grid_with_becke()
            mask = grid_with_weights[:, 3] > threshold
            return grid_with_weights[mask, :3], grid_with_weights[mask, 3]

        full_grid = self.get_full_grid()
        full_weights = self.get_full_weights()
        mask = full_weights > threshold
        return full_grid[mask], full_weights[mask]
    
    def apply_becke_weights(self, do_becke_hetero=True):
        """Apply Becke partition weights to all atomic grids.

        This updates each atomic grid's `points_with_weights` array so that the
        fourth column contains weights scaled by the Becke partition function.

        Parameters
        ----------
        do_becke_hetero : bool, optional
            Whether to use Becke's heteroatomic correction (default: True)
        """
        if not self.atomic_grids:
            return

        becke = Becke(self)

        for iat, agrid in enumerate(self.atomic_grids):
            points = agrid.get_full_grid()
            weights = agrid.get_full_weights()

            # Compute Becke partition weights for each point and select the
            # contribution for this atomic grid.
            becke_weights = np.array(
                [becke._weight_function(p, do_becke_hetero=do_becke_hetero) for p in points]
            )

            agrid.points_with_weights[:, 3] = weights * becke_weights[:, iat]

    def get_molecular_grid_with_becke(self, do_becke_hetero=True):
        """
        Get molecular grid with Becke weights applied
        
        Parameters
        ----------
        do_becke_hetero : bool, optional
            Whether to use Becke's heteroatomic correction
            
        Returns
        -------
        ndarray
            Grid points with Becke weights [x, y, z, weight]
        """
        # Apply Becke weights
        self.apply_becke_weights(do_becke_hetero)

        # Collect all points with weights
        if not self.atomic_grids:
            return np.zeros((0, 4))

        all_points_with_weights = []
        for agrid in self.atomic_grids:
            all_points_with_weights.append(agrid.points_with_weights)

        return np.vstack(all_points_with_weights)
    
    def __len__(self):
        """Total number of grid points"""
        return sum(len(agrid) for agrid in self.atomic_grids)
    
    def __repr__(self):
        return (f"MolecularGrid(natoms={len(self.atoms)}, "
                f"nshells={self.nshells}, "
                f"nangpts={self.nangpts}, "
                f"npoints={len(self)})")
