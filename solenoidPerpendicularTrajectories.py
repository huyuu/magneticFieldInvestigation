import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits import mplot3d
import multiprocessing as mp
import pickle
import datetime as dt
import redis
from numpy import abs, sqrt, cos, sin, pi, arccos

from scipy.integrate import quadrature, dblquad, tplquad
from scipy.special import ellipk, ellipe, ellipkm1

from solenoidScalarPotentialDistribution import Omega, BpFromScalarPotential, BpFromVectorPotential, Aphi


# Constants

mu0 = 4*nu.pi*1e-7



# Models

class TrajectoryGenerator():
    def __init__(self):
        self.hostIP = '10.32.247.48'
        self.hostPort=6379

        self.I = 1.0
        self.coilRadius = 1.5e-2
        self.Z0 = self.coilRadius
        self.deltaT = 1e-5
        N = 21
        conductorWidth = 4e-3
        if N % 2 == 1:
            self.coilZs = nu.linspace(-(N//2) * conductorWidth, (N//2) * conductorWidth, N)
        else:
            self.coilZs = nu.linspace( -(N//2 - 0.5) * conductorWidth, (N//2 - 0.5) * conductorWidth, N)
        # initial points [x0]
        self.z0s = nu.linspace(-1.5*self.Z0, 1.5*self.Z0, 21)


    def run(self):
        # get mode string
        modeString = sys.args[1]
        assert modeString != None
        if lowercase(modeString) == 'master' or lowercase(modeString) == 'm':
            self.runAsMasterOnCluster()
        elif lowercase(modeString) == 'slave' or lowercase(modeString) == 's':
            if sys.args[2] != None:
                self.runAsSlaveOnCluster(workerAmount=sys.args[2])
            else:
                self.runAsSlaveOnCluster()


    def runAsMasterOnCluster(self):
        master = redis.Redis(host=self.hostIP, port=self.hostPort)
        print('Master node starts.')
        # clean queues
        while master.rpop('rawQueue') != None:
            pass
        while master.rpop('cookedQueue') != None:
            pass
        print('Queues cleaned-up.')
        print('Start main calculation')
        _start = dt.datetime.now()
        # start main calculation
        # generate all initial points [x0] and push them to raw queue.
        for z0 in self.z0s:
            master.lpush('rawQueue', (self.I, self.coilRadius, self.coilZs, self.Z0, self.deltaT, 0.9*self.coilRadius, z0))
        # collect calculated trajectories
        trajectories = []
        for _ in range(len(z0s)):
            _, trajectory = master.brpop('cookedQueue')
            trajectories.append(trajectory)
        _end = dt.datetime.now()
        print('All {} trajectories generated. (cost {:.3g} hours)'.format(len(z0s), (_end-_start).total_seconds()/3600.0))
        # plot results
        # plot bs
        los = nu.linspace(0.2*coilRadius, 0.9*coilRadius, points)
        zs = nu.linspace(-2*Z0, 2*Z0, points)
        aphis = nu.zeros((points, points))
        bs_lo = nu.zeros((points, points))
        bs_z = nu.zeros((points, points))
        for i, lo in enumerate(los):
            for j, z in enumerate(zs):
                # bp = BpFromScalarPotential(I, r, theta, coilRadius)
                aphis[i, j] = Aphi(I, lo, z, coilRadius)
                bp = BpFromVectorPotential(I, lo, z, coilRadius)
                bs_lo[i, j] = bp[0]
                bs_z[i, j] = bp[1]
        _los, _zs = nu.meshgrid(los, zs, indexing='ij')
        pl.quiver(_los/coilRadius, _zs/Z0, bs_lo, bs_z)
        # plot trajectories
        for trajectory in trajectories:
            pl.plot(trajectory[:, 0], trajectory[:, 1], '--', c='gray')
        pl.show()


    def runAsSlaveOnCluster(self, workerAmount=min(mp.cpu_count()//2, 55), rawQueue='rawQueue', cookedQueue='cookedQueue'):
        workerTank = []
        print(f'Slave node starts with {workerAmount} workers.')
        for _ in range(workerAmount):
            worker = mp.Process(target=computeTrajectoryInCluster, args=(rawQueue, cookedQueue, self.hostIP, self.hostPort))
            worker.start()
        for worker in workerTank:
            worker.join()


def computeTrajectoryInCluster(rawQueue, cookedQueue, hostIP, hostPort):
    slave = redis.Redis(host=hostIP, port=hostPort)
    while True:
        _, args = slave.brpop(rawQueue)
        trajectory = drawTrajectory(*args)
        slave.lpush(cookedQueue, trajectory)


def drawTrajectory(I, coilRadius, coilZs, Z0, deltaT, x0_lo, x0_z):
    x = nu.array([x0_lo, x0_z])
    lastX = nu.array([x0_lo, x0_z])
    trajectory = []
    t = 0
    while 0.2 <= x[0]/coilRadius <= 0.9 and -2 <= x[1]/Z0 <= 2:
        if sqrt((x[0]-lastX[0])**2 + (x[1]-lastX[1])**2) >= coilRadius/100:
            trajectory.append([x[0]/coilRadius, x[1]/Z0])
            lastX = nu.array([x[0], x[1]])
        bp = nu.zeros(2)
        for coilZ in coilZs:
            bp_lo, bp_z = BpFromVectorPotential(I, x[0], x[1], coilRadius, coilZ)
            bp += nu.array([bp_lo, bp_z])
        m = nu.array([-bp[1], bp[0]])
        x += m * deltaT
        t += deltaT
    return nu.array(trajectory)


# Main


if __name__ == '__main__':
    mp.freeze_support()
    trajectoryGenerator = TrajectoryGenerator()
    trajectoryGenerator.run()



# if __name__ == '__main__':
#     coilRadius = 1.5e-2
#     coilZ = 0
#     points = 100
#     Z0 = coilRadius
#     I = 1.0
#     deltaT = 1e-5
#
#     los = nu.linspace(0.2*coilRadius, 0.9*coilRadius, points)
#     zs = nu.linspace(-2*Z0, 2*Z0, points)
#     omegas = nu.zeros((points, points))
#     aphis = nu.zeros((points, points))
#     bs_lo = nu.zeros((points, points))
#     bs_z = nu.zeros((points, points))
#     for i, lo in enumerate(los):
#         for j, z in enumerate(zs):
#             r = sqrt(lo**2 + z**2)
#             theta = arccos(z/r)
#             omegas[i, j] = Omega(I, r, theta, coilRadius)
#             # bp = BpFromScalarPotential(I, r, theta, coilRadius)
#             aphis[i, j] = Aphi(I, lo, z, coilRadius)
#             bp = BpFromVectorPotential(I, lo, z, coilRadius)
#             bs_lo[i, j] = bp[0]
#             bs_z[i, j] = bp[1]
#     # compute trajectories
#     args = []
#     for z0 in nu.linspace(-1.5*Z0, 1.5*Z0, 21):
#         args.append((I, coilRadius, Z0, deltaT, 0.9*coilRadius, z0))
#     trajectories = []
#     with mp.Pool(processes=min(mp.cpu_count()-1, 50)) as pool:
#         trajectories = pool.starmap(drawTrajectory, args)
#     # with open('trajectories.pickle', 'wb') as file:
#     #     pickle.dump(trajectories, file)
#     # plot bs
#     _los, _zs = nu.meshgrid(los, zs, indexing='ij')
#     pl.quiver(_los/coilRadius, _zs/Z0, bs_lo, bs_z)
#     # plot trajectories
#     for trajectory in trajectories:
#         pl.plot(trajectory[:, 0], trajectory[:, 1], '--', c='gray')
#     pl.show()
