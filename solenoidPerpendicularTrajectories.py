import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
import pickle
from numpy import abs, sqrt, cos, sin, pi, arccos

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1

from solenoidScalarPotentialDistribution import Omega, BpFromScalarPotential, BpFromVectorPotential, Aphi


# Constants

mu0 = 4*nu.pi*1e-7



# Models

def drawTrajectory(I, coilRadius, Z0, deltaT, x0_lo, x0_z):
    x = nu.array([x0_lo, x0_z])
    trajectory = []
    t = 0
    while 0.2 <= x[0]/coilRadius <= 0.9 and -2 <= x[1]/Z0 <= 2:
        if sqrt(x[0]**2 + x[1]**2) >= coilRadius/100:
            trajectory.append([x[0]/coilRadius, x[1]/Z0])
        bp = BpFromVectorPotential(I, x[0], x[1], coilRadius)
        m = nu.array([-bp[1], bp[0]])
        x += m * deltaT
        t += deltaT
    return nu.array(trajectory)


# Main

if __name__ == '__main__':
    coilRadius = 1.5e-2
    coilZ = 0
    points = 100
    Z0 = coilRadius
    I = 1.0
    deltaT = 1e-5

    los = nu.linspace(0.2*coilRadius, 0.9*coilRadius, points)
    zs = nu.linspace(-2*Z0, 2*Z0, points)
    omegas = nu.zeros((points, points))
    aphis = nu.zeros((points, points))
    bs_lo = nu.zeros((points, points))
    bs_z = nu.zeros((points, points))
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

    # compute trajectories
    args = []
    for z0 in nu.linspace(-1.5*Z0, 1.5*Z0, 21):
        args.append((I, coilRadius, Z0, deltaT, 0.9*coilRadius, z0))
    trajectories = []
    with mp.Pool(processes=min(mp.cpu_count()-1, 50)) as pool:
        trajectories = pool.starmap(drawTrajectory, args)
    # with open('trajectories.pickle', 'wb') as file:
    #     pickle.dump(trajectories, file)
    # plot bs
    _los, _zs = nu.meshgrid(los, zs, indexing='ij')
    pl.quiver(_los/coilRadius, _zs/Z0, bs_lo, bs_z)
    # plot trajectories
    for trajectory in trajectories:
        pl.plot(trajectory[:, 0], trajectory[:, 1], '--', c='gray')
    pl.show()
