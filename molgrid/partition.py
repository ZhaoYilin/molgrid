import numpy as np

class Becke:
    def __init__(self, molecule):
        """
        Construct Becke partition for a molecule.

        Parameters
        ----------
        molecule : Molecule
            A Molecule object containing atomic information.
        """
        self.molecule = molecule
        self.atom_coords = np.array([atom.coordinate for atom in molecule])  # (Na, 3)
        self.natom = len(molecule)

        # Precompute interatomic distance matrix (Na, Na)
        self.R_matrix = self._distance_matrix()

        # Precompute hetero-atomic correction matrix (Na, Na)
        self.a_matrix = self._hetero_matrix()

    def weight_function(self, coords, chunk_size=20000):
        """
        Compute normalized Becke weights w_A(r).

        Parameters
        ----------
        coords : ndarray of shape (Ng, 3)
            Grid point coordinates.
        chunk_size : int, optional
            Number of grid points per batch.

        Returns
        -------
        W : ndarray of shape (Ng, Na)
            Becke weights for each atom.
        """
        P = self._voronoi_polyhedron(coords, chunk_size)

        # Normalize weights
        W = P / np.sum(P, axis=1, keepdims=True)
        return W

    def _voronoi_polyhedron(self, coords, chunk_size=20000):
        """
        Compute unnormalized Becke cell function P_A(r).

        Parameters
        ----------
        coords : ndarray of shape (Ng, 3)
        chunk_size : int

        Returns
        -------
        P_all : ndarray of shape (Ng, Na)
        """
        coords = np.asarray(coords)
        Ng = coords.shape[0]
        Na = self.natom

        P_all = np.empty((Ng, Na))

        for start in range(0, Ng, chunk_size):
            end = min(start + chunk_size, Ng)
            r = coords[start:end]  # (Nc, 3)

            # Compute distances from grid points to atoms (Nc, Na)
            diff = r[:, None, :] - self.atom_coords[None, :, :]
            R = np.linalg.norm(diff, axis=2)

            # Build mu tensor (Nc, Na, Na)
            rip = R[:, :, None]
            rjp = R[:, None, :]
            rij = self.R_matrix[None, :, :]

            # Compute mu_ij
            mu = rip - rjp
            mu /= rij

            # Apply hetero-atomic correction
            mu += self.a_matrix[None, :, :] * (1.0 - mu**2)

            # Clip to avoid numerical overflow
            mu = np.clip(mu, -1.0, 1.0)

            # Apply Becke smoothing
            s = self._smoothing_function(mu)

            # Set diagonal terms (i == j) to 1.0 (no self-interaction)
            idx = np.arange(Na)
            s[:, idx, idx] = 1.0

            # Use log-product to improve numerical stability
            logP = np.sum(np.log(s + 1e-15), axis=2)
            logP -= np.max(logP, axis=1, keepdims=True)            
            P = np.exp(logP)

            P_all[start:end] = P

        return P_all

    def _smoothing_function(self, mu, k=3):
        """
        Apply Becke smoothing polynomial.

        Parameters
        ----------
        mu : ndarray
        k : int
            Number of recursive polynomial applications

        Returns
        -------
        ndarray
        """
        for _ in range(k):
            mu = 1.5 * mu - 0.5 * mu**3

        return 0.5 * (1.0 - mu)
    
    def _distance_matrix(self):
        """
        Compute interatomic distance matrix.

        Returns
        -------
        R_matrix : ndarray of shape (Na, Na)
        """
        diff = self.atom_coords[:, None, :] - self.atom_coords[None, :, :]
        R_matrix = np.linalg.norm(diff, axis=2)

        # Avoid division by zero
        R_matrix[R_matrix < 1e-12] = 1e-12

        return R_matrix

    def _hetero_matrix(self):
        """
        Compute Becke hetero-atomic correction parameters.

        Returns
        -------
        a_matrix : ndarray of shape (Na, Na)
        """
        a_matrix = np.zeros((self.natom, self.natom))

        for i, atomA in enumerate(self.molecule):
            for j, atomB in enumerate(self.molecule):
                if i == j:
                    continue

                if hasattr(atomA, 'number') and hasattr(atomB, 'number'):
                    if atomA.number != atomB.number:
                        chi = atomA.cov_radius_slater / atomB.cov_radius_slater
                        u = (chi - 1.0) / (chi + 1.0)

                        # Becke transformation
                        a = u / (u**2 - 1.0)
                        a = np.clip(a, -0.5, 0.5)

                        a_matrix[i, j] = a

        return a_matrix


    