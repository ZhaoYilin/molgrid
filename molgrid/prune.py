import numpy as np

class Becke:
    def __init__(self, molecule):
        """Construct Becke grid for a molecule"""
        self.molecule = molecule
        self.atoms = molecule.atoms  # Added to store atoms from molecule

    def _weight_function(self, xyzp, do_becke_hetero=True):
        """Compute the Becke weight function for a point in space.

        Parameters
        ----------
        xyzp : ndarray
            Point coordinates [x, y, z]
        do_becke_hetero : bool, optional
            Whether to apply heteroatomic corrections (default: True)

        Returns
        -------
        list of float
            List of weights for each atom
        """
        weights = []
        for ati in self.atoms:
            weight = self._voronoi_polyhedron(ati, xyzp, do_becke_hetero=do_becke_hetero)
            weights.append(weight)

        # Normalize weights so they sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        return weights
    
    def _voronoi_polyhedron(self, ati, xyzp, do_becke_hetero=True):
        """Voronoi polyhedron function constructed from the smoothing function.

        Parameters
        ----------
        ati : Atom
            Reference atom
        xyzp : ndarray
            Point coordinates [x, y, z]
        do_becke_hetero : bool, optional
            Whether to apply heteroatomic corrections (default: True)

        Returns
        -------
        float
            Becke weight
        """
        rip = np.linalg.norm(ati.coordinate - xyzp)
        sprod = 1.0

        for atj in self.atoms:
            if ati == atj:
                continue

            rij = np.linalg.norm(ati.coordinate - atj.coordinate)
            rjp = np.linalg.norm(atj.coordinate - xyzp)
            mu = (rip - rjp) / rij

            # Modify mu based on Becke hetero formulas
            if do_becke_hetero and hasattr(ati, 'number') and hasattr(atj, 'number'):
                if ati.number != atj.number:
                    chi = ati.cov_radius_slater / atj.cov_radius_slater
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