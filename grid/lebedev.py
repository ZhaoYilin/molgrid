import numpy as np
import glob
import os

class Lebedev:
    def __init__(self):
        """Construct the Lebedev coefficient library"""
        self._data = []
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
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
        return sorted([int(d['degree']) for d in self._data])
    
    def get_npoints_list(self):
        return sorted([int(d['npoints']) for d in self._data])