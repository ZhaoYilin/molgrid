import numpy as np

class Becke:
    def __init__(self, molecule):
        """Construct Becke grid for a molecule."""
        self.molecule = molecule

    def _weight_function(self, coord_r):
        """Compute the Becke weight function for a point in space.

        Parameters
        ----------
        coord_r : ndarray
            Point coordinates [x, y, z]

        Returns
        -------
        list of float
            List of weights for each atom
        """
        weights = []
        for atomA in self.molecule:
            weight = self._voronoi_polyhedron(atomA, coord_r)
            weights.append(weight)

        # Normalize weights so they sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        return weights
    
    def _voronoi_polyhedron(self, atomA, coord_r):
        """Voronoi polyhedron function constructed from the smoothing function.

        Parameters
        ----------
        atomA : Atom
            Reference atom
        coord_r : ndarray
            Point coordinates [x, y, z]

        Returns
        -------
        float
            Becke weight
        """
        rip = np.linalg.norm(atomA.coordinate - coord_r)
        sprod = 1.0

        for atomB in self.molecule:
            if atomA == atomB:
                continue

            rij = np.linalg.norm(atomA.coordinate - atomB.coordinate)
            rjp = np.linalg.norm(atomB.coordinate - coord_r)
            mu = (rip - rjp) / rij

            # Modify mu based on Becke hetero formulas
            if hasattr(atomA, 'number') and hasattr(atomB, 'number'):
                if atomA.number != atomB.number:
                    chi = atomA.cov_radius_slater / atomB.cov_radius_slater
                    u = (chi - 1.0) / (chi + 1.0)
                    a = u / (u**2 - 1)
                    a = min(a, 0.5)
                    a = max(a, -0.5)
                    mu += a * (1 - mu**2)

            sprod *= self._smoothing_function(mu)

        return sprod
    
    def _smoothing_function(self, mu, k=3):
        """
        Smoothing function - single iteration
        
        Parameters
        ----------
        mu : float
            Input value
        k : int, optional
            Number of iterations    
            
        Returns
        -------
        float
            Smoothed value
        """
        fk = self._polynomial_fk(mu, k)
        return 0.5 * (1.0 - fk)
    
    def _polynomial_fk(self, mu, k=3):
        """
        Polynomial fk obtained by the iteration formula
        
        Parameters
        ----------
        mu : float
            Input value
        k : int, optional
            Number of iterations
            
        Returns
        -------
        float
            Polynomial value
        """
        result = mu
        for _ in range(k):
            result = self._polynomial_p(result)
        return result

    def _polynomial_p(self, mu):
        """
        Polynomial p which is the first iteration of the polynomial fk.
        
        Parameters
        ----------
        mu : float
            Input value
            
        Returns
        -------
        float
            Polynomial value
        """
        return 1.5 * mu - 0.5 * mu**3