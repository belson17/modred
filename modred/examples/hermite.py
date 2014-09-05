"""
Spectral differentiation from J.A.C. Weideman and S.C. Reddy 1998, ACM TOMS.
"""

import numpy as N
import numpy.ma as ma

def herroots(n):
    """Returns the roots of the Hermite polynomial of degree n."""
    # Jacobi matrix
    J = N.diag(N.arange(1, n)**0.5, 1) + N.diag(N.arange(1, n)**0.5, -1)
    return N.sort(N.linalg.eigvalsh(J)) / (2**0.5)
    

def herdif(n, m, b):
    """Computes differentiation matrices D1, D2, ..., Dm on Hermite points.
    
    Args:
        n: Number of points, which is also the order of accuracy.
        m: Number of derivative matrices to return.
        b: Scaling parameter. Real and positive.
    
    Returns:
        x: Array of nodes, zeros of Hermite polynomial of degree n, scaled by b.
        Dm: A list s.t. Dm[i] is the (i+1)-th derivative matrix, i=0...m-1.
    
    Note: 0 < m < n-1.
    """
    x = herroots(n)
    # Compute weights
    alpha = N.exp(-x**2 / 2.)
    # Set up beta matrix s.t. beta[i,j] = 
    #  ( (i+1)-th derivative of alpha(x) )/alpha(x), evaluated at x = x(j).
    beta = N.zeros((m+1, x.shape[0]))
    beta[0] = 1.0
    beta[1] = -x
    for i in range(2, m+1):
        beta[i] = -x * beta[i-1] - (i-1) * beta[i-2]
    # Remove initializing row from beta
    beta = N.delete(beta, 0, 0)
    # Compute differentiation matrix (b=1).
    Dm = poldif(x, alpha=alpha, B=beta)
    # Scale nodes by the factor b.
    x = x/b
    # Adjust derivatives for b not equal to 1.
    for i in range(1, m+1):
        Dm[i-1] *= b**i
        
    return x, Dm


def poldif(x, m=None, alpha=None, B=None):
    """
    Computes the differentiation matrices D1, D2, ..., Dm on arbitrary nodes.
    
    The function is called with either keyword argument m OR 
    keyword args alpha and B.
    If m is given, then the weight function is constant.
    If alpha and B are given, then the weights are defined by alpha and B.
    
    Args:
        x: 1D array of n distinct nodes.
    
    Kwargs:
        m: Number of derivatives.
        alpha: 1D array of weight values alpha[x], evaluated at x = x[k].
        B: Array of size m x n where B[i,j] = beta[i,j] = ((i+1)-th derivative
            of alpha(x))/alpha(x), evaluated at x = x[j].
    
    Returns:
        Dm: A list s.t. Dm[i] is the (i+1)-th derivative matrix, i=0...m-1.
    
    Note: 0 < m < n-1.
    """
    x = x.flatten()
    n = x.shape[0]
    if m is not None and B is None and alpha is None: 
        alpha = N.ones(n)
        B = N.zeros((m, n))
    elif m is None and B is not None and alpha is not None:
       alpha = alpha.flatten()
       m = B.shape[0]
    else:
        raise RuntimeError('Keyword args to poldif are inconsistent.')
    
    XX = N.tile(x, (n, 1)).transpose()
    # DX contains entries x[k] - x[j].
    DX = XX - XX.transpose()
    # Put 1's one the main diagonal.
    N.fill_diagonal(DX, 1.)
    
    # C has entries c[k]/c[j].
    c = alpha * N.prod(DX, 1)
    C = N.tile(c, (n, 1)).transpose()
    C = C/C.transpose()
    
    # Z has entries 1/(x[k]-x[j])
    Z = 1./DX
    N.fill_diagonal(Z, 0.)
    
    # X is Z' but with the diagonal entries removed.
    X = Z.transpose()
    X = ma.array(X.transpose(), mask=N.identity(n)).compressed().\
        reshape((n, n-1)).transpose()
    
    # Y is matrix of cumulative sums and D is a differentiation matrix.
    Y = N.ones((n, n))
    D = N.eye(n)
    Dm = []
    for i in range(1, m+1):
        # Diagonals
        Y = N.cumsum(N.concatenate((B[i-1].reshape((1,n)), 
            i * Y[0:n-1] * X), axis=0), axis=0)
        # Off-diagonals
        D = i * Z * (C * N.tile(N.diag(D), (n,1)).transpose() - D)
        # Correct the diagonal
        D.flat[::n+1] = Y[-1]
        Dm.append(D)
    
    return Dm

