import numpy as np
from molgrid.quadrature import Lebedev, GaussChebychev

class AtomicGrid:
    def __init__(self, atom, nshells: int = 32, nangpts: int = 110):
        """
        Initialize the AtomicGrid class
        
        Parameters
        ----------
        atom : Atom
            Atom object containing position and atomic properties
        nshells : int, optional
            Number of radial shells (default: 32)
        nangpts : int, optional
            Number of angular points (default: 110)
        """
        self.atom = atom
        self.nshells = nshells       # number of radial shells
        self.nangpts = nangpts       # number of angular points
        
        # Initialize quadrature objects
        self.__lebedev = Lebedev()
        self.__chebychev = GaussChebychev()
        
        # Grid attributes
        self.__radcoords = None  # radial grids coordinates
        self.__radweights = None  # radial grids weights
        self.__angcoords = None  # angular grids coordinates
        self.__angweights = None  # angular grids weights   
        self.__coords = None # full grids coordinates
        self.__weights = None # full grids weights
    
        # Build the grid upon initialization
        self.build()  
                
    def build(self):
        """
        Build and store the full atomic integration grid.

        The routine generates radial and angular sub-grids, combines them into
        3D Cartesian grid coordinates, and computes spherical volume weights:
        weight = r^2 * w_radial * 4π * w_angular.

        Notes
        -----
        - `self.__coords` and `self.__weights` are written in-place.
        - `self.__coords` shape is (nshells * nangpts, 3).
        - `self.__weights` shape is (nshells * nangpts,).

        Returns
        -------
        None
        """
        # Generate radial and angular grids
        self._generate_radial_grid()
        self._generate_angular_grid()
        
        
        radcoord, radial_weights = self.__radcoords, self.__radweights
        angcoords, angular_weights = self.__angcoords, self.__angweights
        
        # Outer product to generate full grid: r_i * angular_points_j
        coords = np.multiply.outer(radcoord, angcoords)
        
        # Reshape to (nshells * nangpts, 3)
        nshells = len(radcoord)
        nang = len(angcoords)
        coords = coords.reshape(nshells * nang, 3)
        
        # Add atomic center position
        center = getattr(self.atom, 'coordinate', np.array([0.0, 0.0, 0.0]))
        coords = coords + center
        
        # In spherical integration, dV = r^2 dr d\Omega.
        radial_weights = radial_weights * (radcoord ** 2)

        # Outer product of weights
        weights = np.multiply.outer(radial_weights, angular_weights)
        weights = weights.flatten()  # Flatten to 1D array

        self.__coords = coords
        self.__weights = weights
                
    def _generate_radial_grid(self):
        """
        Generate radial grid using Gauss-Chebyshev quadrature
        """
        # Get Bragg-Slater radius for the atom in Angstroms
        r_BS_angstrom = getattr(self.atom, 'cov_radius_slater')
        # Convert radius from Angstrom to Bohr
        r_BS_bohr = r_BS_angstrom * 1.889726124565062
        r_scale = r_BS_bohr * 0.5 
        
        # Generate radial points and weights
        self.__radcoords, self.__radweights = self.__chebychev.semi_infinite(r_scale, self.nshells)
        
    def _generate_angular_grid(self):
        """
        Generate angular grid using Lebedev quadrature
        """
        # Find the closest available Lebedev degree for requested nangpts
        available_npoints = self.__lebedev.get_npoints_list()
        
        # Find the degree that gives closest to requested number of points
        target = self.nangpts
        closest_npoints = min(available_npoints, key=lambda x: abs(x - target))
        
        # Get the degree for this number of points
        degrees = self.__lebedev.get_degrees_list()
        npoints_list = self.__lebedev.get_npoints_list()
        
        # Find matching degree
        idx = npoints_list.index(closest_npoints)
        degree = degrees[idx]
        
        # Get angular points (on unit sphere)
        self.__angcoords, self.__angweights = self.__lebedev.get(degree, coord='cartesian')
                   
    @property
    def center(self):
        """Atomic center position"""
        return getattr(self.atom, 'coordinate', np.array([0.0, 0.0, 0.0]))
 
    @property
    def coords(self):
        """Full grid coordinates (nshells * nangpts, 3)"""
        if self.__coords is None:
            self.build()
        return self.__coords
    
    @property
    def weights(self):
        """Full grid weights (nshells * nangpts,)"""
        if self.__weights is None:
            self.build()
        return self.__weights
    
    @property
    def radial_coords(self):
        """
        Return radial grid coordinates
        
        Returns
        -------
        ndarray             
            Radial coordinates (shape: (nshells, 3))
        """
        if self.__radcoords is None:
            self._generate_radial_grid()
        return self.__radcoords        

    @property
    def radial_weights(self):
        """
        Return radial grid weights

        Returns
        -------
        ndarray
            Radial weights (shape: (nshells,))
        """
        if self.__radweights is None:
            self._generate_radial_grid()
        return self.__radweights

    @property
    def angular_coords(self):
        """
        Return angular grid coordinates

        Returns
        -------
        ndarray            
            Angular coordinates (shape: (nangpts, 3))
        """
        if self.__angcoords is None:
            self._generate_angular_grid()
        return self.__angcoords
    
    @property
    def angular_weights(self):
        """
        Return angular grid weights

        Returns
        -------
        ndarray
            Angular weights (shape: (nangpts,))
        """
        if self.__angweights is None:
            self._generate_angular_grid()
        return self.__angweights
    
    def __len__(self):
        """Total number of atomic grid points."""
        if self.__coords is None:
            self.build()
        return len(self.__coords)
