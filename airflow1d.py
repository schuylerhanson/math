import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import svds
from scipy import linalg
import numpy.linalg as la

def grid(Lx, Nx):
    dx = Lx/Nx
    x = np.array([i*dx for i in range(0, Nx)])
    return x, dx

def sources(x, x_s, x_k):
    x_s = x[x_s]
    x_k = x[x_k]
    i_s = np.where(x == x_s)[0][0]
    i_k = np.where(x == x_k)[0][0]    
    return x_s, i_s, x_k, i_k

def matrix_setup(x, Nx, Q_in, Q_out, BC=False):
    dim_x = len(x)
    A = np.zeros((dim_x, dim_x))
    b = np.zeros((dim_x))
    
    for i in range(0, Nx):
        if i == 0:
            A[i, i] = -1
            A[i, i+1] = 1
        elif i == Nx-1:
            A[i, i-1] = -1
            A[i, i] = 1
        elif i == i_s:
            A[i, i-1] = 1
            A[i, i] = -2
            A[i, i+1] = 1
            b[i] = Q_in*dx
        elif i == i_k:
            A[i, i-1] = 1
            A[i, i] = -2
            A[i, i+1] = 1
            b[i] = -Q_out*dx
        elif i == Nx:
            pass
        else:
            A[i, i-1] = 1
            A[i, i] = -2
            A[i, i+1] = 1
    
    if BC:
        # Dirichlet pin for now
        A[0, 0] = 1
        A[0, 1] = 0
        b[0] = 1
        A[-1, -1] = 1
        A[-1, -2] = 0
        b[-1] = 1
    return A, b

def solver(A, b, pinv=False):
    if pinv:
        return la.pinv(A)@b
    else:
        return spsolve(A, b)

def velocity_field(phi, Nx, dx, BC=False):
    u = np.zeros(Nx)
    for i in range(1, Nx-1):
        u[i] = (phi[i+1] - phi[i-1])/(2*dx)
    if BC:
        u[0] = (phi[1] - phi[0])/dx
        u[-1] = (phi[-2] - phi[-1])/dx
    return u 


Lx = 5
Nx = 100
Q_in = 1
Q_out = 1
x, dx = grid(Lx, Nx)
x_s, i_s, x_k, i_k = sources(x, 1, -1)

A, b = matrix_setup(x, Nx, Q_in, Q_out, BC=True)
phi = solver(A, b)
u = velocity_field(phi, Nx, dx, BC=True)
print('phi:', phi, '\n')
print('u:', u, '\n')

print("A:", A, '\n')

rank = la.matrix_rank(A)
#print(rank)

u, s, vt = linalg.svd(A, full_matrices=False)
eps = np.finfo(float).eps
tol = max(A.shape) * s[0] * eps
rank = np.count_nonzero(s > tol)
print(f"Numerical rank = {rank}")
