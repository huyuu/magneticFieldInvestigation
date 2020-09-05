import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
from numpy import abs, sqrt, cos, sin, pi

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1


# Constants

mu0 = 4*nu.pi*1e-7
I = 100


# Model

def Aphi(I, coilRadius, coilZ, lo, z):
    squaredK = 4*coilRadius*lo/( (coilRadius+lo)**2 + (z-coilZ)**2 )
    k = sqrt(squaredK)
    Aphi =  mu0*I/pi * ( (sqrt((coilRadius+lo)**2+(z-coilZ)**2)/(2*lo) - coilRadius/sqrt((coilRadius+lo)**2+(z-coilZ)**2))*ellipk(squaredK) - ellipe(squaredK) )
    return Aphi


if __name__ == '__main__':
    coilRadius = 1.5e-2
    coilZ = 0
    points = 50
    Z0 = coilRadius

    los = nu.linspace(0.1*coilRadius, 0.9*coilRadius, points)
    zs = nu.linspace(0, Z0, points)
    As = nu.zeros((points, points))

    for i, lo in enumerate(los):
        for j, z in enumerate(zs):
            As[i, j] = Aphi(1.0, coilRadius, coilZ, lo, z)

    _los, _zs = nu.meshgrid(los, zs, indexing='ij')
    pl.contourf(_los, _zs, As, levels=50)
    pl.colorbar()
    pl.show()
