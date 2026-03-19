import numpy as np
from lebedev import lebedev

# Distance units
bohr2ang = 0.529177249  # Conversion of length from bohr to angstrom
ang2bohr = 1/bohr2ang

class atomic_grid(object):
    def __init__(self,atom,**kwargs):
        atno,x,y,z = atom.atuple()

        grid_params = LegendreGrid(atno,**kwargs)
        self.points = []

        for rrad,wrad,nang in grid_params:
            for xang,yang,zang,wang in lebedev[nang]:
                w = wrad*wang
                self.points.append((rrad*xang+x,rrad*yang+y,rrad*zang+z,w))
        self.points = np.array(self.points,dtype=float)
        self.npts = self.points.shape[0]

# The following two routines return [(ri,wi,nangi)] for nrad shells.
# The ri's are properly adjusted to go to the proper distances.
# The wi's are adjusted to only have to be multiplied by wrad from
# the lebedev shell

def LegendreGrid(Z,**kwargs):
    from constants import ang2bohr
    from data import Bragg
    from legendre import legendre
    Rmax = 0.5*Bragg[Z]*ang2bohr

    nrad = kwargs.get('nrad',20)
    fineness = kwargs.get('fineness',1)
    radial = legendre[nrad]
    grid = []
    for i in range(nrad):
        xrad,wrad = radial[i]
        rrad = BeckeRadMap(xrad,Rmax)
        dr = 2*Rmax/pow(1-xrad,2)
        vol = 4*np.pi*rrad*rrad*dr
        nangpts = 14
        #nangpts = ang_mesh(float(i+1)/nrad,fineness)
        print(nangpts)
        grid.append((rrad,wrad*vol,nangpts))
    return grid
    
def BeckeRadMap(x,Rmax):
    return Rmax*(1.0+x)/(1.0-x)

def ang_mesh(frac,fineness,alevs = None):
    """\
    Determine the number of points in the angular mesh based on
    the fraction of the total radial grid index frac c (0,1).

    You can optionally pass in the number of points for
    the 5 different regions
    """
    if not alevs:
        ang_levels = [
            [ 6, 14, 26, 26, 14], # Coarse
            [ 50, 50,110, 50, 26], # Medium
            [ 50,110,194,110, 50], # Fine
            [194,194,194,194,194]  # ultrafine
            ]
        alevs = ang_levels[fineness]
    nang = alevs[0]
    if frac > 0.4: nang = alevs[1]
    if frac > 0.5: nang = alevs[2]
    if frac > 0.7: nang = alevs[3]
    if frac > 0.8: nang = alevs[4]
    return nang
