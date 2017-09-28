#!/usr/bin/env python
"""Simulate linearized CGL and find BPOD modes and reduced-order model."""
import numpy as np
import scipy.linalg as spla

import modred as mr
import hermite as hr

plots = True
if plots:
    try:
        import matplotlib.pyplot as plt
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
x_1 = -(-2 * (mu_0 - c_u ** 2) / mu_2) ** 0.5   # branch I
x_2 = -x_1  # branch II
x_s = x_2
nu = U + 2j * c_u
gamma = 1. + 1j * c_d
chi = (-mu_2 / (2. * gamma)) ** 0.25    # chi: decay rate of global modes

# Print parameters
print('Parameters:')
for var in [
    'nx', 'dt', 'U', 'c_u', 'c_d', 'mu_0', 'mu_2', 's', 'x_s', 'nu', 'gamma',
    'chi']:
    print('    %s = %s' % (var, str(eval(var))))

# Collocation points in x are roughly [-85, 85], as in Ilak 2010
x, Ds = hr.herdif(nx, 2, np.real(chi))

# Inner product weights, trapezoidal rule
weights = np.zeros(nx)
weights[0] = 0.5 * (x[1] - x[0])
weights[-1] = 0.5 * (x[-1] - x[-2])
weights[1:-1] = 0.5 * (x[2:] - x[0:-2])
M = np.diag(weights)
inv_M = np.linalg.inv(M)
M_sqrt = np.diag(weights ** 0.5)
inv_M_sqrt = np.diag(weights ** -0.5)

# LTI system arrays for direct and adjoint ("_adj") systems
mu = (mu_0 - c_u ** 2) + mu_2 * x ** 2 / 2.
A = -nu * Ds[0] + gamma * Ds[1] + np.diag(mu)

# Compute optimal disturbance and use it as B array
A_discrete = spla.expm(A * dt)
exp_array = np.identity(nx, dtype=complex)
max_sing_val = 0
for i in mr.range(1, 100):
    exp_array = exp_array.dot(A_discrete)
    U, E, VH = np.linalg.svd(M_sqrt.dot(exp_array.dot(inv_M_sqrt)))
    if max_sing_val < E[0]:
        max_sing_val = E[0]
        optimal_dist = VH.conj().T[:, 0]
B = -inv_M_sqrt.dot(optimal_dist)
C = np.exp(-((x - x_s) / s) ** 2).dot(M)
A_adj = inv_M.dot(A.conj().T.dot(M))
C_adj = inv_M.dot(C.conj().T)

# Plot spatial distributions of B and C
if plots:
    plt.figure()
    plt.plot(x, B.real, 'b')
    plt.plot(x, B.imag, 'r')
    plt.xlabel('x')
    plt.ylabel('B')
    plt.legend(['Real', 'Imag'])
    plt.xlim([-20, 20])
    plt.title("Spatial distribution of B")
    plt.grid(True)

    plt.figure()
    plt.plot(x, C.T.real, 'b')
    plt.plot(x, C.T.imag, 'r')
    plt.xlabel('x')
    plt.ylabel('C')
    plt.legend(['Real', 'Imag'])
    plt.xlim([-20, 20])
    plt.title("Spatial distribution of C")
    plt.grid(True)

# Simulate impulse responses to the direct and adjoint systems w/Crank-Nicolson
# (q(i+1) - q(i)) / dt = 1/2 (A q(i+1) + A q(i)) + B u(i)
# => (I - dt/2 A) q(i+1) = q(i) + dt/2 A q(i) + dt B u(i)
#    LHS q(i+1) = RHS q(i) + dt B u(i)
LHS = np.identity(nx) - dt / 2. * A
RHS = np.identity(nx) + dt / 2. * A
LHS_adj = np.identity(nx) - dt / 2. * A_adj
RHS_adj = np.identity(nx) + dt / 2. * A_adj

nt = 300
q = np.zeros((nx, nt), dtype=complex)
q_adj = np.zeros((nx, nt), dtype=complex)

q[:, 0] = B
q_adj[:, 0] = C_adj

for ti in mr.range(nt - 1):
    q[:, ti + 1] = np.linalg.solve(LHS, RHS.dot(q[:, ti]))
    q_adj[:, ti + 1] = np.linalg.solve(LHS_adj, RHS_adj.dot(q_adj[:, ti]))

# Plot all snapshots as a contour plot
if plots:
    t = np.arange(0, nt * dt, dt)
    X, T = np.meshgrid(x, t)
    plt.figure()
    plt.contourf(T, X, q.real.T, 20, cmap=plt.cm.binary)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Direct snapshots (real part)')
    plt.colorbar()

    plt.figure()
    plt.contourf(T, X, q_adj.real.T, 20, cmap=plt.cm.binary)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Adjoint snapshots (real part)')
    plt.colorbar()

# Compute the BPOD modes
r = 10
BPOD_res = mr.compute_BPOD_arrays(
    q, q_adj,
    adjoint_mode_indices=list(mr.range(r)),
    direct_mode_indices=list(mr.range(r)),
    inner_product_weights=weights)

# Plot the first 3 modes
if plots:
    for i in mr.range(3):
        plt.figure()
        plt.plot(x, BPOD_res.direct_modes[:, i].real, '-o')
        plt.plot(x, BPOD_res.adjoint_modes[:, i].real, '-x')
        plt.xlabel('Space')
        plt.ylabel('Real(q)')
        plt.legend(['direct', 'adjoint'])
        plt.title('Direct and adjoint mode %d' % (i + 1))

# Project the linear dynamics onto the modes
projection = mr.LTIGalerkinProjectionArrays(
    BPOD_res.direct_modes, adjoint_basis_vecs=BPOD_res.adjoint_modes,
    inner_product_weights=weights, is_basis_orthonormal=True)
A_direct_modes = A.dot(BPOD_res.direct_modes)
Ar, Br, Cr = projection.compute_model(
    A_direct_modes, B, C.dot(BPOD_res.direct_modes))

# Verify that the model accurately reproduces the impulse response
qr = np.zeros((r, nt), dtype=complex)
qr[:, 0] = Br
LHSr = np.identity(r) - dt / 2. * Ar
RHSr = np.identity(r) + dt / 2. * Ar
for ti in mr.range(nt - 1):
    qr[:, ti + 1] = np.linalg.solve(LHSr, RHSr.dot(qr[:, ti]))
y = C.dot(q)
yr = Cr.dot(qr)

# Print error in reduced-order model impulse response
print(
    'Max error in reduced system impulse response output y is %0.4e'
    % np.abs(y - yr).max())

# Plot impulse response output
if plots:
    plt.figure()
    plt.plot(t, y.T.real, '-')
    plt.plot(t, yr.T.real, '--')
    plt.legend(['Full', 'Reduced, r=%d' % r])
    plt.xlabel('t')
    plt.ylabel('real(y)')

    plt.show()
