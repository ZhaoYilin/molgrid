import numpy as np
import glob
import os


class Lebedev:
    # Class variable for classmethod use
    _file_paths = {}  # degree -> file path
    
    def __init__(self):
        """Construct the Lebedev coefficient library"""
        # Load file paths to class variable (only runs once)
        if not self.__class__._file_paths:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            for f in glob.glob(os.path.join(data_dir, '*.csv')):
                name = os.path.basename(f)
                if name.startswith('lebedev_'):
                    deg = name.split('_')[1].split('.')[0]
                    self.__class__._file_paths[deg] = f
        
        # Instance variable references class variable
        self._data = self.__class__._file_paths
                
    @classmethod 
    def get(cls, degree, coord='cartesian'):
        """
        Get points and weights for given degree
        
        Args:
            degree: Quadrature degree
            coord: 'cartesian' (x,y,z) or 'spherical' (r,theta,phi)
            
        Returns:
            (points, weights)
        """
        deg_str = str(degree)
        if deg_str not in cls._file_paths:
            avail = sorted([int(d) for d in cls._file_paths])
            raise ValueError(f"Degree {degree} not found. Available: {avail}")
        
        # Load CSV: r,theta,phi,x,y,z,weight
        data = np.loadtxt(cls._file_paths[deg_str], delimiter=',', skiprows=1)
        
        if coord == 'cartesian':
            return data[:, 3:6], data[:, 6]  # x,y,z
        elif coord == 'spherical':
            return data[:, 0:3], data[:, 6]  # r,theta,phi
        else:
            raise ValueError("coord must be 'cartesian' or 'spherical'")