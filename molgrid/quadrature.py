import numpy as np
import glob
import os

class Lebedev:
    def __init__(self):
        """Construct the Lebedev coefficient library"""
        self._data = []
        data_dir = os.path.join(os.path.dirname(__file__), 'data/lebedev')
        
        for f in glob.glob(os.path.join(data_dir, '*.csv')):
            name = os.path.basename(f)
            if name.startswith('lebedev_'):
                deg = name.split('_')[1]
                npoints = name.split('_')[2].split('.')[0] 
                self._data.append({'degree': deg, 'npoints': npoints, 'file': f})
    
    def get(self, degree, coord='cartesian'):
        """
        Get points and weights for given degree
        
        Args:
            degree: Quadrature degree
            coord: 'cartesian' (x,y,z) or 'spherical' (r,theta,phi)
            
        Returns:
            (points, weights)
        """
        deg_str = str(degree)
        if deg_str not in [d['degree'] for d in self._data]:
            avail = sorted([int(d['degree']) for d in self._data])
            raise ValueError(f"Degree {degree} not found. Available: {avail}")
        
        data = np.loadtxt([d['file'] for d in self._data if d['degree'] == deg_str][0], delimiter=',', skiprows=1)
        
        if coord == 'cartesian':
            return data[:, 3:6], data[:, 6]
        elif coord == 'spherical':
            return data[:, 0:3], data[:, 6]
        else:
            raise ValueError("coord must be 'cartesian' or 'spherical'")

    def get_degrees_list(self):
        """Get list of available degrees"""
        return sorted([int(d['degree']) for d in self._data])
    
    def get_npoints_list(self):
        """Get list of number of points for each degree"""
        return sorted([int(d['npoints']) for d in self._data])
    

class GaussChebychev:
    def __init__(self):
        """Construct the Gauss-Chebyshev quadrature library"""
        pass
        
    def finite(self, nshells):
        """
        Generate Gauss-Chebyshev quadrature on finite interval [-1, 1]
        
        Parameters
        ----------
        nshells : int
            Number of quadrature points
            
        Returns
        -------
        x : ndarray
            Quadrature points in [-1, 1]
        w : ndarray
            Quadrature weights (sum = π/2 for second-kind Chebyshev)
        """
        # Angular spacing: π/(n+1)
        f = np.pi / float(nshells + 1)
        
        # Indices: k = 1, 2, ..., n
        z = np.arange(1, nshells + 1)
        
        # Chebyshev points: x_k = cos(kπ/(n+1))
        x = np.cos(f * z)
        
        # Second-kind Chebyshev weights: ω_k = (π/(n+1)) sin²(kπ/(n+1))
        w = f * np.sin(f * z)**2
        
        return x, w

    def semi_infinite(self, r_scale, nshells):
        """
        Generate Gauss-Chebyshev quadrature on semi-infinite interval [0, ∞)
        
        Maps [-1, 1] → [0, ∞) using: r = r_scale * (1 + x) / (1 - x)
        
        Parameters
        ----------
        r_scale : float
            Radial scaling factor (e.g., Bragg-Slater radius)
        nshells : int
            Number of quadrature points
            
        Returns
        -------
        r : ndarray
            Radial quadrature points in [0, ∞)
        w : ndarray
            Radial quadrature weights for ∫ f(r) dr
        """
        # Get Gauss-Chebyshev (second kind) points and weights on [-1, 1]
        x, w_finite = self.finite(nshells)
        
        # Map from [-1, 1] to [0, ∞)
        # r = r_scale * (1 + x) / (1 - x)
        # x = -1 → r = 0, x = 1 → r = ∞
        r = r_scale * (1.0 + x) / (1.0 - x)
        
        # Jacobian of the transformation: dr/dx
        jacobian = 2.0 * r_scale / (1.0 - x)**2
        
        # Convert from weighted integral to ordinary integral
        # The final weights include the Jacobian and remove the weight function
        w = w_finite * jacobian / np.sqrt(1.0 - x**2)
        
        return r, w