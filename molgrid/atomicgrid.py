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
        self.__rr = None  # radial points
        self.__rw = None  # radial weights
        self.__angpts = None  # angular points
        self.__aw = None  # angular weights
        
        # Generate grids
        self._generate_radial_grid()
        self._generate_angular_grid()
        
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
        self.__rr, self.__rw = self.__chebychev.semi_infinite(r_scale, self.nshells)
        
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
        self.__angpts, self.__aw = self.__lebedev.get(degree, coord='cartesian')
        
    def get_radial_grid(self):
        """
        Return radial grid points and weights
        
        Returns
        -------
        tuple
            (radial_points, radial_weights)
        """
        if self.__rr is None or self.__rw is None:
            self._generate_radial_grid()
        return self.__rr, self.__rw

    def get_angular_grid(self):
        """
        Return angular grid points and weights
        
        Returns
        -------
        tuple
            (angular_points, angular_weights)
        """
        if self.__angpts is None or self.__aw is None:
            self._generate_angular_grid()
        return self.__angpts, self.__aw
           
    def get_full_grid(self):
        """
        Get all the grid points as a list of positions (N x 3) matrix
        
        Returns
        -------
        ndarray
            Full 3D grid points (nshells * nangpts, 3)
        """
        # Ensure grids are generated
        rr, _ = self.get_radial_grid()
        angpts, _ = self.get_angular_grid()
        
        # Outer product to generate full grid: r_i * angular_points_j
        full_grid = np.multiply.outer(rr, angpts)
        
        # Reshape to (nshells * nangpts, 3)
        nshells = len(rr)
        nang = len(angpts)
        full_grid = full_grid.reshape(nshells * nang, 3)
        
        # Add atomic center position
        center = getattr(self.atom, 'coordinate', np.array([0.0, 0.0, 0.0]))
        full_grid = full_grid + center
        
        return full_grid
    
    def get_full_weights(self):
        """
        Get weights for all grid points
        
        Returns
        -------
        ndarray
            Weights for each grid point
        """
        # Ensure grids are generated
        rr, rw = self.get_radial_grid()
        angpts, aw = self.get_angular_grid()
        
        # Weights are product of radial and angular weights
        nshells = len(rr)
        nang = len(angpts)
        
        # Outer product of weights
        weights = np.multiply.outer(rw, aw)
        
        # Reshape to 1D array
        return weights.reshape(nshells * nang)
    
    @property
    def center(self):
        """Atomic center position"""
        return getattr(self.atom, 'coordinate', np.array([0.0, 0.0, 0.0]))
    
    @property
    def atomic_number(self):
        """Atomic number"""
        return getattr(self.atom, 'number') 
    
    def __len__(self):
        """Total number of grid points"""
        rr, _ = self.get_radial_grid()
        angpts, _ = self.get_angular_grid()
        return len(rr) * len(angpts)
    
    def __repr__(self):
        return (f"AtomicGrid(atom={getattr(self.atom, 'symbol', 'Unknown')}, "
                f"nshells={self.nshells}, "
                f"nangpts={self.nangpts})")