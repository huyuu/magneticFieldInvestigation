import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
from numpy import abs, sqrt, cos, sin, pi, arccos

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1


# Constants

mu0 = 4*nu.pi*1e-7


# Model

def Omega(I, r, theta, coilRadius):
    return 0.5*I*(1/r - cos(theta)/r**2)


def BpFromScalarPotential(I, r, theta, coilRadius):
    Bp_lo = -0.5*I*mu0*( -sin(theta)/r**2 + 3*sin(theta)*cos(theta)/r**3 )
    Bp_z = -0.5*I*mu0*( -cos(theta)/r**2 + (2*cos(theta)**2 - sin(theta)**2)/r**3 )
    return (Bp_lo, Bp_z)


def Aphi(I, lo, z, coilRadius):
    coilZ = 0
    squaredK = 4*coilRadius*lo/( (coilRadius+lo)**2 + (z-coilZ)**2 )
    k = sqrt(squaredK)
    Aphi =  mu0*I/pi * ( (sqrt((coilRadius+lo)**2+(z-coilZ)**2)/(2*lo) - coilRadius/sqrt((coilRadius+lo)**2+(z-coilZ)**2))*ellipk(squaredK) - ellipe(squaredK) )
    return Aphi


def BpFromVectorPotential(I, lo, z, coilRadius, coilZ=0):
    a = coilRadius
    z = z - coilZ
    beta = (a+lo)**2+z**2
    k = sqrt(4*a*lo/beta)
    squaredK = k**2
    dKdk = ellipe(squaredK)/(k*(1-k**2)) - ellipk(squaredK)/k
    dEdk = (ellipe(squaredK) - ellipk(squaredK))/k
    dkdz = -sqrt(4*a*lo)*beta**(-1.5)*z
    dkdlo = sqrt(a/lo/beta) - 2*(a+lo)*sqrt(a*lo)/beta**1.5

    Bp_lo = -mu0*I/(2*pi*lo) * (
        (z/sqrt(beta) + 2*a*lo*z/beta**1.5) * ellipk(squaredK) +\
        (sqrt(beta)-2*a*lo/sqrt(beta)) * dKdk * dkdz -\
        z/sqrt(beta) * ellipe(squaredK) -\
        sqrt(beta) * dEdk * dkdz
    )
    Bp_z = mu0*I/(2*pi*lo) * (
        ((a+lo)/sqrt(beta) - 2*a/sqrt(beta) + 2*a*lo*(a+lo)/beta**1.5) * ellipk(squaredK) +\
        (sqrt(beta) - 2*a*lo/sqrt(beta)) * dKdk * dkdlo -\
        (a+lo)/sqrt(beta) * ellipe(squaredK) -\
        sqrt(beta) * dEdk * dkdlo
    )
    return (Bp_lo, Bp_z)


if __name__ == '__main__':
    coilRadius = 1.5e-2
    coilZ = 0
    points = 100
    Z0 = coilRadius
    I = 1.0
    deltaT = 1e-2

    los = nu.linspace(0.2*coilRadius, 0.9*coilRadius, points)
    zs = nu.linspace(-2*Z0, 2*Z0, points)
    omegas = nu.zeros((points, points))
    aphis = nu.zeros((points, points))
    bs_lo = nu.zeros((points, points))
    bs_z = nu.zeros((points, points))
    ms_lo = nu.zeros((points, points))
    ms_z = nu.zeros((points, points))
    for i, lo in enumerate(los):
        for j, z in enumerate(zs):
            r = sqrt(lo**2 + z**2)
            theta = arccos(z/r)
            omegas[i, j] = Omega(I, r, theta, coilRadius)
            # bp = BpFromScalarPotential(I, r, theta, coilRadius)
            aphis[i, j] = Aphi(I, lo, z, coilRadius)
            bp = BpFromVectorPotential(I, lo, z, coilRadius)
            bs_lo[i, j] = bp[0]
            bs_z[i, j] = bp[1]
            ms_lo[i, j] = -bp[1]
            ms_z[i, j] = bp[0]

    _los, _zs = nu.meshgrid(los, zs, indexing='ij')
    pl.contourf(_los/coilRadius, _zs, omegas, levels=50)
    pl.contourf(_los/coilRadius, _zs, aphis, levels=50)
    pl.colorbar()
    pl.quiver(_los/coilRadius, _zs, bs_lo, bs_z)
    pl.show()

    pl.contourf(_los/coilRadius, _zs, omegas, levels=100)
    pl.colorbar()
    pl.quiver(_los/coilRadius, _zs, ms_lo, ms_z)
    pl.show()

    pl.contourf(_los/coilRadius, _zs, bs_z, levels=50)
    pl.colorbar()
    pl.show()
