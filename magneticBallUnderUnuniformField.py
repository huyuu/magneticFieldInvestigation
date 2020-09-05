import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import sympy as sp
from numpy import pi, abs, sin, cos, tan, sqrt, arccos


# Models

class BallAnalizer():
    def __init__(self, imposedField):
        self.imposedField = imposedField


    def plotScalarPotentialDistribution(self, innerRadius, outerRadius, myu, points=50):
        # get coeffs A, B, C, D
        imposedField = self.imposedField
        coeffMatrix = nu.array([
            [1/outerRadius**2, -outerRadius, -1/outerRadius**2, 0],
            [-2/outerRadius**3, -myu, 2*myu/outerRadius**3, 0],
            [0, innerRadius, 1/innerRadius**2, -innerRadius],
            [0, myu, -2*myu/innerRadius**3, -1]
        ])
        b = nu.array([
            -outerRadius * imposedField,
            -imposedField,
            0,
            0
        ]).reshape(-1, 1)
        A, B, C, D = (nu.linalg.inv(coeffMatrix) @ b).ravel()
        # Potential and H field definitines
        P_external = lambda r, theta: imposedField*r*cos(theta) + A*cos(theta)/r**2
        Hx_external = lambda r, theta: 3*A*sin(theta)*cos(theta)/r**3
        Hz_external = lambda r, theta: -imposedField + A/r**3*(2*cos(theta)**2 - sin(theta)**2)
        P_between = lambda r, theta: B*r*cos(theta) + C*cos(theta)/r**2
        Hx_between = lambda r, theta: 3*C*cos(theta)*sin(theta)/r**3
        Hz_between = lambda r, theta: -B + C/r**3*(2*cos(theta)**2 - sin(theta)**2)
        P_internal = lambda r, theta: D*r*cos(theta)
        Hx_internal = lambda r, theta: 0
        Hz_internal = lambda r, theta: -D
        # potential and H field distribution
        edgeIndicator = 2.0
        xs = nu.linspace(0.001*innerRadius, edgeIndicator*outerRadius, points)
        zs = nu.linspace(-edgeIndicator*outerRadius, edgeIndicator*outerRadius, points)
        p_distribution = nu.zeros((points, points))
        hx_distribution = nu.zeros((points, points))
        hz_distribution = nu.zeros((points, points))
        for i, x in enumerate(xs):
            for j, z in enumerate(zs):
                r = sqrt(x**2 + z**2)
                theta = arccos(z/r)
                if r > outerRadius:
                    p_distribution[i, j] = P_external(r, theta)
                    hx_distribution[i, j] = Hx_external(r, theta)
                    hz_distribution[i, j] = Hz_external(r, theta)
                elif innerRadius <= r <= outerRadius:
                    p_distribution[i, j] = P_between(r, theta)
                    hx_distribution[i, j] = Hx_between(r, theta)
                    hz_distribution[i, j] = Hz_between(r, theta)
                else: # inner
                    p_distribution[i, j] = P_internal(r, theta)
                    hx_distribution[i, j] = Hx_internal(r, theta)
                    hz_distribution[i, j] = Hz_internal(r, theta)
        # plot
        _xs, _zs = nu.meshgrid(xs, zs, indexing='ij')
        pl.contourf(_xs, _zs, p_distribution, levels=20)
        pl.colorbar()
        pl.quiver(_xs, _zs, hx_distribution, hz_distribution)
        pl.show()



if __name__ == '__main__':
    analizer = BallAnalizer(imposedField=1)
    analizer.plotScalarPotentialDistribution(innerRadius=1, outerRadius=1.2, myu=1000, points=100)
