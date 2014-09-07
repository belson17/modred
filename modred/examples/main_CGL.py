#!/usr/bin/env python
"""Simulate linearized CGL and find BPOD modes and reduced-order model."""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from future.builtins import range

import numpy as np
import scipy.linalg as SL
import modred as mr
from . import hermite as H

plots = False

if plots:
    try:
        import matplotlib.pyplot as PLT
    except:
        plots = False

# Parameters for the subcritical case
nx = 220
dt = 1.0
s = 1.6
U = 2.0
c_u = 0.2
c_d = -1.0
mu_0 = 0.38
mu_2 = -0.01
x_1 = -(-2*(mu_0 - c_u**2)/mu_2)**0.5 # branch I
x_2 = -x_1 # branch II
x_s = x_2
nu = U + 2J * c_u
gamma = 1. + 1J * c_d
# chi: decay rate of global modes
chi = (-mu_2 / (2.*gamma))**0.25 

print('----- Parameters ------')
for var in ['nx','dt','U','c_u','c_d','mu_0','mu_2','s','x_s',
    'nu','gamma','chi']:
    print(var, '=', eval(var))
print('-----------------------')

# Collocation points in x are roughly [-85, 85], as in Ilak 2010
x, Ds = H.herdif(nx, 2, np.real(chi))

# Inner product weights, trapezoidal rule
weights = np.zeros(nx)
weights[0] = 0.5*(x[1] - x[0])
weights[-1] = 0.5*(x[-1] - x[-2])
weights[1:-1] = 0.5*(x[2:] - x[0:-2])
M = np.mat(np.diag(weights))
inv_M = np.linalg.inv(M)
M_sqrt = np.mat(np.diag(weights**0.5))
inv_M_sqrt = np.mat(np.diag(weights**-0.5))

# LTI system matrices for direct and adjoint ("_adj") systems
mu = (mu_0 - c_u**2) + mu_2 * x**2/2.
A = np.mat(-nu * Ds[0] + gamma * Ds[1] + np.diag(mu))

# Compute optimal disturbance and use it as B matrix
A_discrete = np.mat(SL.expm(A*dt))
exp_mat = np.mat(np.identity(nx, dtype=complex))
max_sing_val = 0
for i in range(1, 100):
    exp_mat *= A_discrete
    U,E,VH = np.linalg.svd(M_sqrt*exp_mat*inv_M_sqrt)
    if max_sing_val < E[0]:
        max_sing_val = E[0]
        optimal_dist = np.mat(VH).H[:,0]
        #print i, E[0], max_sing_val

B = -inv_M_sqrt * optimal_dist
C = np.mat(np.exp(-((x - x_s)/s)**2)) * M
A_adj = inv_M * A.H * M
C_adj = inv_M * C.H

# Plot spatial distributions of B and C
if plots:
    PLT.figure()
    PLT.hold(True)
    PLT.plot(x, B.real,'b')
    PLT.plot(x, B.imag,'r')
    PLT.xlabel('x')
    PLT.ylabel('B')
    PLT.legend(['Real','Imag'])
    PLT.xlim([-20, 20])
    PLT.grid(True)

    PLT.figure()
    PLT.hold(True)
    PLT.plot(x, C.T.real,'b')
    PLT.plot(x, C.T.imag,'r')
    PLT.xlabel('x')
    PLT.ylabel('C')
    PLT.legend(['Real','Imag'])
    PLT.xlim([-20, 20])
    PLT.grid(True)
    

# Simulate impulse responses to the direct and adjoint systems w/Crank-np.colson
# (q(i+1) - q(i)) / dt = 1/2 (A q(i+1) + A q(i)) + B u(i)
# => (I - dt/2 A) q(i+1) = q(i) + dt/2 A q(i) + dt B u(i)
#    LHS q(i+1) = RHS q(i) + dt B u(i)
LHS = np.identity(nx) - dt/2.*A
RHS = np.identity(nx) + dt/2.*A
LHS_adj = np.identity(nx) - dt/2.*A_adj
RHS_adj = np.identity(nx) + dt/2.*A_adj

nt = 300
q = np.mat(np.zeros((nx, nt), dtype=complex))
q_adj = np.mat(np.zeros((nx, nt), dtype=complex))

q[:,0] = B
q_adj[:,0] = C_adj

for ti in range(nt-1):
    q[:,ti+1] = np.linalg.solve(LHS, RHS*q[:,ti])
    q_adj[:,ti+1] = np.linalg.solve(LHS_adj, RHS_adj*q_adj[:,ti])
    
# Plot all snapshots as a contour plot
if plots:
    t = np.arange(0, nt*dt, dt)
    X, T = np.meshgrid(x, t)
    PLT.figure()
    PLT.contourf(T, X, np.array(q.real).T, 20, cmap=PLT.cm.binary)
    PLT.xlabel('t')
    PLT.ylabel('x')
    PLT.colorbar()
    PLT.figure()
    PLT.contourf(T, X, np.array(q_adj.real).T, 20, cmap=PLT.cm.binary)
    PLT.xlabel('t')
    PLT.ylabel('x')
    PLT.title('adjoint')
    PLT.colorbar()

# Compute the BPOD modes
r = 10
direct_modes, adjoint_modes, sing_vals = mr.compute_BPOD_matrices(
    q, q_adj, list(range(r)), list(range(r)), inner_product_weights=weights)

# Plot the first 3 modes
if plots:
    for i in range(3):
        PLT.figure()
        PLT.hold(True)
        PLT.plot(x, direct_modes[:,i].real, '-o')
        PLT.plot(x, adjoint_modes[:,i].real,'-x')
        PLT.xlabel('Space')
        PLT.ylabel('Real(q)')
        PLT.legend(['direct', 'adjoint'])
        PLT.title('Direct and adjoint mode %d'%(i+1))

# Project the linear dynamics onto the modes
projection = mr.LTIGalerkinProjectionMatrices(direct_modes,
    adjoint_basis_vecs=adjoint_modes, inner_product_weights=weights,
    is_basis_orthonormal=True)
A_direct_modes = A * direct_modes
Ar, Br, Cr = projection.compute_model(A_direct_modes, B, C.dot(direct_modes))

# Verify that the model accurately reproduces the impulse response
qr = np.mat(np.zeros((r, nt), dtype=complex))
qr[:,0] = Br
LHSr = np.identity(r) - dt/2.*Ar
RHSr = np.identity(r) + dt/2.*Ar
for ti in range(nt-1):
    qr[:,ti+1] = np.linalg.solve(LHSr, RHSr*qr[:,ti])
y = C*q
yr = Cr*qr

print('Max error in reduced system impulse response output y is', np.amax(np.abs(y-yr)))

if plots:
    PLT.figure()
    PLT.plot(t, y.T.real)
    PLT.hold(True)
    PLT.plot(t, yr.T.real)
    PLT.legend(['Full','Reduced, r=%d'%r])
    PLT.xlabel('t')
    PLT.ylabel('real(y)')

    PLT.show()
