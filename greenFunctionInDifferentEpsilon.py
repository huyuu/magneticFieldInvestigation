import numpy as nu
import pandas as pd
from matplotlib import pyplot as pl
import sympy as sp
from numpy import pi, abs, sin, cos, tan, sqrt, arccos


# Model

def V(q, r_, theta_, r, theta, eps1, eps2, R):
    # coeffs
    d = R**2/r_
    Sr_ = sqrt(R**2 + r_**2 - 2*R*r_*cos(theta-theta_))
    Sd = sqrt(R**2 + d**2 - 2*R*d*cos(theta-theta_))
    alpha = (R - r_*cos(theta-theta_)) / (R - d*cos(theta-theta_))
    # calculation
    if r < R:
        q1 = (alpha*eps1-eps2*Sd/Sr_)/(eps2-eps1) * q
        print(q1)
        return 1/(4*pi*eps1) * (q/sqrt(r**2+r_**2-2*r*r_*cos(theta-theta_)) - q1/sqrt(r**2+d**2-2*r*d*cos(theta-theta_)))
    else:
        q2 = (alpha*eps2 - Sd/Sr_*eps2) / (eps2-eps1) * q
        print(q2)
        return 1/(4*pi*eps2) * q2 / sqrt(r**2 + r_**2 + -2*r*r_*cos(theta-theta_))


# Main

R = 1
points = 300
eps1 = 1
eps2 = 10

q = 1
r_ = 1e-5
theta_ = 0

V(q, r_, theta_, 0.9*R, 0, eps1, eps2, R)
V(q, r_, theta_, 1.1*R, 0, eps1, eps2, R)
# xs = nu.linspace(-2*R, 2*R, points)
# ys = nu.linspace(-2*R, 2*R, points)
# vs = nu.zeros((points, points))
# for i, x in enumerate(xs):
#     for j, y in enumerate(ys):
#         r = sqrt(x**2+y**2)
#         theta = theta = arccos(x/r)
#         vs[i, j] = V(q, r_, theta_, r, theta, eps1, eps2, R)
#
# # Plot
# _xs, _ys = nu.meshgrid(xs, ys, indexing='ij')
# pl.contourf(_xs, _ys, vs, levels=20)
# pl.colorbar()
# pl.show()
