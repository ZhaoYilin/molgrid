import numpy as np
from molgrid.atomicgrid import AtomicGrid
from molgrid.partition import Becke


class MolecularGrid:
    """Build a molecular integration grid by combining atomic grids.

    Each atom produces a radial/angular grid via :class:`~molgrid.atomicgrid.AtomicGrid`.
    Becke partition weights can be applied to distribute grid contributions across atoms.
    """

    def __init__(self, molecule, nshells=32, nangpts=110, partition_method='becke'):
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
        partition_method : 'becke', optional
            Method for partitioning grid points (default: 'becke'). 'becke' uses Becke
            atomic partition weights for partitioning. 
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
        self.partition_method = partition_method
         
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
        if self.partition_method is None:
            # Use all atomic grid coordinates and weights directly
            self.__atomic_grids = atomic_grids
            self.__coords = np.vstack([agrid.coords for agrid in atomic_grids])
            self.__weights = np.hstack([agrid.weights for agrid in atomic_grids])
            return 
        elif isinstance(self.partition_method, str):
            self.partition(atomic_grids, self.partition_method)
        else:
            raise ValueError("Invalid partition method. Supported: 'becke'.")
        
    def partition(self, atomic_grids, partition_method='becke'):
        """Partition the grid weights based on the specified method.
        
        Parameters
        ----------
        atomic_grids : [AtomicGrid]
            List of atomic grids to partition.
        partition_method : str, optional
            Partition method to use. Default is 'becke'.
            
        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If an invalid partition method is specified.
        """
        if partition_method == 'becke':
            # Create Becke object for calculating atomic partition weights
            becke = Becke(self.molecule)
            partition_weights = []  

            # Iterate over each atomic grid
            for i, grid_i in enumerate(atomic_grids):
                # Calculate Becke weights (atomic partition weights) for each point
                W = becke.weight_function(grid_i.coords)  # (Ng, Na)
                becke_weights = W[:, i]
                
                # Adjust original weights by Becke weights
                adjusted_weights = grid_i.weights * becke_weights
                
                # Update atomic grid's internal coordinates and adjusted weights
                grid_i._AtomicGrid__weights = adjusted_weights
                partition_weights.append(adjusted_weights)
                
                
            # Update molecular grid coordinates and weights
            self.__atomic_grids = atomic_grids
            self.__coords = np.vstack([agrid.coords for agrid in atomic_grids])
            self.__weights = np.hstack(partition_weights)
            return
        else:
            raise ValueError("Invalid partition method. Supported: 'becke'.")        

    def __len__(self):
        """Total number of grid points."""
        return len(self.__weights)

    def __iter__(self):
        """Iterate over underlying atomic grid objects."""
        return iter(self.__atomic_grids)