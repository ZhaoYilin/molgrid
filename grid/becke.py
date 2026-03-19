import numpy as np

def becke_reweight_atoms(atoms,agrids,**kwargs):
    for iat,agrid in enumerate(agrids):
        for i in range(agrid.npts):
            xyzp = agrid.points[i,:3]
            Ps = [becke_atomic_grid_p(atj,xyzp,atoms,**kwargs) for atj in atoms]
            P = Ps[iat]/sum(Ps)
            agrid.points[i,3] = P*agrid.points[i,3]
    return

def becke_atomic_grid_p(ati,xyzp,atoms,**kwargs):
    from pyquante2.grid.data import Bragg
    do_becke_hetero = kwargs.get('do_becke_hetero',True)
    sprod = 1
    rip = np.linalg.norm(ati.r-xyzp)
    for atj in atoms:
        if ati == atj: continue
        rij = np.linalg.norm(ati.r-atj.r)
        rjp = np.linalg.norm(atj.r-xyzp)
        mu = (rip-rjp)/rij
        # Modify mu based on Becke hetero formulas (App A)
        if do_becke_hetero and ati.atno != atj.atno:
            chi = Bragg[ati.atno]/Bragg[atj.atno]
            u = (chi-1.)/(chi+1.)
            a = u/(u*u-1)
            a = min(a,0.5)
            a = max(a,-0.5)
            mu += a*(1-mu*mu)
        sprod *= sbecke(mu)
    return sprod

def fbecke(x,n=3):
    """ Voronoi polyhedron function constructed from the smoothing function

    Parameters
    ----------
    x : _type_
        _description_
    n : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    """    
    for i in range(n): x = pbecke(x)
    return x

def pbecke(x): 
    r""" The polynomial fk is obtained by the iteration formula

    Parameters
    ----------
    x : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """    
    return 1.5*x-0.5*pow(x,3)

def sbecke(x,n=3): 
    r"""smoothing function

    Parameters
    ----------
    x : _type_
        _description_
    n : int, optional
        _description_, by default 3

    Returns
    -------
    _type_
        _description_
    """    
    return 0.5*(1-fbecke(x,n))
