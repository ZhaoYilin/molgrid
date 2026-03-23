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
        
        # Lebedev weights in data sum to 1, so scale by 4\pi. 
        if coord == 'cartesian':
            return data[:, 3:6], data[:, 6] * 4 * np.pi
        elif coord == 'spherical':
            return data[:, 0:3], data[:, 6] * 4 * np.pi
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
            Quadrature weights (sum = pi/2 for second-kind Chebyshev)
        """
        f = np.pi / float(nshells + 1)
        z = np.arange(1, nshells + 1)
        x = np.cos(f * z)
        
        # Second-kind Chebyshev weights: w_k = (π/(n+1)) sin^2(k*pi/(n+1))
        w = f * np.sin(f * z)**2
        
        return x, w

    def semi_infinite(self, r_scale, nshells):
        """
        Generate Gauss-Chebyshev quadrature on semi-infinite interval [0, \infty)
        
        Maps [-1, 1] to [0, \infty) using: r = r_scale * (1 + x) / (1 - x)
        
        Parameters
        ----------
        r_scale : float
            Radial scaling factor 
        nshells : int
            Number of quadrature points
            
        Returns
        -------
        r : ndarray
            Radial quadrature points in [0, \infty)
        w : ndarray
            Radial quadrature weights for \int f(r) dr
        """
        # Get Gauss-Chebyshev (second kind) points and weights on [-1, 1]
        x, w_finite = self.finite(nshells)
        
        # Map from [-1, 1] to [0, \infty)
        r = r_scale * (1.0 + x) / (1.0 - x)
        
        # Jacobian of the transformation: dr/dx
        jacobian = 2.0 * r_scale / (1.0 - x)**2
        
        # Convert from weighted integral to ordinary integral
        # The final weights include the Jacobian and remove the weight function
        w = w_finite * jacobian / np.sqrt(1.0 - x**2)
        
        return r, w