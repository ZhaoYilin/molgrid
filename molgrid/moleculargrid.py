import numpy as np
from molgrid.atomicgrid import AtomicGrid
from molgrid.prune import Becke


class MolecularGrid:
    """Build a molecular integration grid by combining atomic grids.

    Each atom produces a radial/angular grid via :class:`~molgrid.atomicgrid.AtomicGrid`.
    Becke partition weights can be applied to distribute grid contributions across atoms.
    """

    def __init__(self, molecule, nshells=32, nangpts=110, prune_method='becke', prune_threshold=1e-6):
        """Initialize the MolecularGrid.

        Parameters
        ----------
        molecule : Molecule
            A Molecule object containing atomic information.
        nshells : int or [int], optional
            Number of radial shells for each atom (default: 32), If a single int is 
            provided, replicate it for all atoms. Otherwise, use the provided list.
        nangpts : int or [int], optional
            Number of angular points for each atom (default: 110), If a single int is 
            provided, replicate it for all atoms. Otherwise, use the provided list.
        prune_method : None or 'becke', optional
            Method for pruning grid points (default: 'becke'). None disables pruning, 
            'becke' uses Becke atomic partition weights for pruning. 
        prune_threshold : float, optional
            Threshold for pruning grid points (default: 1e-6). Points with weights
            below this threshold will be removed.
        """
        # Store molecule object
        self.molecule = molecule

        # Number of radial shells and angular points 
        if isinstance(nshells, int):
            nshells = [nshells] * len(self.molecule)
        if isinstance(nangpts, int):
            nangpts = [nangpts] * len(self.molecule)
        self.nshells = nshells
        self.nangpts = nangpts  
       
        # Store pruning method and threshold
        self.prune_method = prune_method
        self.prune_threshold = prune_threshold
         
        # Grid attributes
        self.__atomic_grids = None
        self.__coords = None
        self.__weights = None

        # Build the grid upon initialization
        self.build()

    @property
    def coords(self):
        """Cached full grid coordinates."""
        if self.__coords is None:
            self.build()
        return self.__coords

    @property
    def weights(self):
        """Cached full grid weights."""
        if self.__weights is None:
            self.build()
        return self.__weights

    def build(self):
        """Build each atomic grid and refresh cached molecular grid.
        
        This method constructs atomic grids for each atom in the molecule and
        optionally prunes the grid points based on the specified pruning method.
        
        Returns
        -------
        None
            Modifies the grid in place by building and optionally pruning the grid.
        
        Raises
        ------
        ValueError
            If an invalid prune method is specified.
        """
        # Build atomic grids for each atom in the molecule
        atomic_grids = [AtomicGrid(atom_i, self.nshells[i], self.nangpts[i])
            for i, atom_i in enumerate(self.molecule)]

        # No pruning method specified
        if self.prune_method is None:
            # Use all atomic grid coordinates and weights directly
            self.__atomic_grids = atomic_grids
            self.__coords = np.vstack([agrid.coords for agrid in atomic_grids])
            self.__weights = np.hstack([agrid.weights for agrid in atomic_grids])
            return 
        
        # Use Becke method for pruning
        elif self.prune_method == 'becke':
            # Create Becke object for calculating atomic partition weights
            becke = Becke(self.molecule)
            pruned_coords = []
            pruned_weights = []

            # Iterate over each atomic grid
            for i, grid_i in enumerate(atomic_grids):
                # Calculate Becke weights (atomic partition weights) for each point
                becke_weights = np.array([becke._weight_function(r)[i] for r in grid_i.coords])
                
                # Adjust original weights by Becke weights
                adjusted_weights = grid_i.weights * becke_weights

                # Keep points with Becke weights above threshold
                keep = becke_weights >= self.prune_threshold

                # Collect pruned coordinates and adjusted weights
                pruned_coords.append(grid_i.coords[keep])
                pruned_weights.append(adjusted_weights[keep])
                
                # Update atomic grid's internal coordinates and adjusted weights
                grid_i._AtomicGrid__coords = grid_i.coords[keep]
                grid_i._AtomicGrid__weights = adjusted_weights[keep]

            # Update molecular grid coordinates and weights
            self.__atomic_grids = atomic_grids
            self.__coords = np.vstack(pruned_coords)
            self.__weights = np.hstack(pruned_weights)
            return
        else:
            # Raise error for invalid pruning method
            raise ValueError("Invalid prune method. Supported: None or 'becke'.")

    def __len__(self):
        """Total number of grid points."""
        return len(self.__weights)

    def __iter__(self):
        """Iterate over underlying atomic grid objects."""
        return iter(self.__atomic_grids)