"""
cm_curvature.utils
==================
Mathematical primitives for the Cayley-Menger curvature pipeline.

Functions:
    effective_resistance_matrix  - Resistance distances via Laplacian pseudoinverse
    cayley_menger_determinant    - 5×5 bordered determinant for 4 points
    tetrahedron_volume_from_cm   - Volume extraction from CM determinant
    embed_tetrahedron            - Euclidean embedding of 4 points from distances
    circumradius                 - Circumradius of a tetrahedron in R³
    wasserstein_1_graph          - Wasserstein-1 distance on graph (for Ollivier-Ricci)
"""

import numpy as np
from numpy.linalg import pinv, det
from scipy.optimize import linprog
from itertools import combinations


# =============================================================================
# EFFECTIVE RESISTANCE DISTANCE
# =============================================================================

def effective_resistance_matrix(G):
    """
    Compute the effective resistance distance matrix from the graph Laplacian.
    
    The effective resistance between nodes i and j is defined as:
    
        Ω_ij = L⁺_ii + L⁺_jj - 2·L⁺_ij
    
    where L⁺ is the Moore-Penrose pseudoinverse of the graph Laplacian L.
    
    Effective resistance produces continuous (non-integer) distances that
    capture structural connectivity, making it superior to shortest-path
    distance for geometric analysis on graphs.
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph. Must be connected.
    
    Returns
    -------
    Omega : ndarray, shape (n, n)
        Symmetric matrix of effective resistance distances.
    nodes : list
        Ordered list of node labels corresponding to matrix indices.
    
    Notes
    -----
    - Requires G to be connected (otherwise L⁺ is not well-defined for
      cross-component pairs).
    - Complexity: O(n³) for the pseudoinverse computation.
    
    References
    ----------
    Klein, D.J. & Randić, M. (1993). Resistance distance. 
    J. Math. Chem., 12, 81-95.
    """
    import networkx as nx
    
    nodes = list(G.nodes())
    n = len(nodes)
    L = nx.laplacian_matrix(G, nodelist=nodes).toarray().astype(float)
    L_pinv = pinv(L)
    
    diag = np.diag(L_pinv)
    Omega = diag[:, None] + diag[None, :] - 2 * L_pinv
    
    # Enforce exact symmetry and zero diagonal
    Omega = (Omega + Omega.T) / 2
    np.fill_diagonal(Omega, 0.0)
    
    return Omega, nodes


# =============================================================================
# CAYLEY-MENGER DETERMINANT
# =============================================================================

def cayley_menger_determinant(d_sq):
    """
    Compute the Cayley-Menger determinant for 4 points.
    
    The 5×5 bordered determinant:
    
        | 0      d01²   d02²   d03²   1 |
        | d01²   0      d12²   d13²   1 |
        | d02²   d12²   0      d23²   1 |
        | d03²   d13²   d23²   0      1 |
        | 1      1      1      1      0 |
    
    Parameters
    ----------
    d_sq : dict
        Squared pairwise distances: {(i,j): d_ij²} for all 
        0 <= i < j <= 3 (6 entries).
    
    Returns
    -------
    delta_cm : float
        Value of the determinant.
    
    Notes
    -----
    - If Δ_CM = 0: points are coplanar.
    - If Δ_CM > 0: points form a non-degenerate tetrahedron in R³.
    - If Δ_CM < 0: distances are not realizable in R³ (non-Euclidean).
    """
    CM = np.zeros((5, 5))
    
    for i in range(4):
        for j in range(4):
            if i != j:
                key = (min(i, j), max(i, j))
                CM[i, j] = d_sq[key]
    
    # Border with ones
    CM[4, :4] = 1.0
    CM[:4, 4] = 1.0
    CM[4, 4] = 0.0
    
    return det(CM)


# =============================================================================
# VOLUME FROM CAYLEY-MENGER
# =============================================================================

def tetrahedron_volume_from_cm(delta_cm):
    """
    Compute tetrahedron volume from the Cayley-Menger determinant.
    
    The relationship is:  288 · V² = Δ_CM
    
    Parameters
    ----------
    delta_cm : float
        Value of the Cayley-Menger determinant.
    
    Returns
    -------
    V : float or None
        Volume of the tetrahedron. None if Δ_CM <= 0.
    is_embeddable : bool
        True if the distance configuration is realizable in R³.
    """
    if delta_cm > 0:
        V = np.sqrt(delta_cm / 288.0)
        return V, True
    return None, False


# =============================================================================
# EUCLIDEAN EMBEDDING OF TETRAHEDRON
# =============================================================================

def embed_tetrahedron(d):
    """
    Embed 4 points in R³ given their 6 pairwise distances.
    
    Uses the canonical placement:
        - p0 at origin
        - p1 on the positive x-axis
        - p2 in the xy-plane (y ≥ 0)
        - p3 in general position (z ≥ 0)
    
    Parameters
    ----------
    d : dict
        Pairwise distances: {(i,j): d_ij} for 0 <= i < j <= 3.
    
    Returns
    -------
    points : ndarray, shape (4, 3) or None
        Coordinates of the 4 embedded points. None if embedding fails
        (e.g., distances violate triangle inequality or are degenerate).
    """
    d01, d02, d03 = d[(0, 1)], d[(0, 2)], d[(0, 3)]
    d12, d13, d23 = d[(1, 2)], d[(1, 3)], d[(2, 3)]
    
    # p0 at origin
    p0 = np.array([0.0, 0.0, 0.0])
    
    # p1 on x-axis
    if d01 < 1e-12:
        return None
    p1 = np.array([d01, 0.0, 0.0])
    
    # p2 in xy-plane
    x2 = (d01**2 + d02**2 - d12**2) / (2 * d01)
    y2_sq = d02**2 - x2**2
    if y2_sq < -1e-10:
        return None
    y2 = np.sqrt(max(y2_sq, 0.0))
    if y2 < 1e-12:
        return None
    p2 = np.array([x2, y2, 0.0])
    
    # p3 in 3D
    x3 = (d01**2 + d03**2 - d13**2) / (2 * d01)
    y3 = (d02**2 + d03**2 - d23**2 - 2 * x2 * x3) / (2 * y2)
    z3_sq = d03**2 - x3**2 - y3**2
    if z3_sq < -1e-10:
        return None
    z3 = np.sqrt(max(z3_sq, 0.0))
    p3 = np.array([x3, y3, z3])
    
    return np.array([p0, p1, p2, p3])


# =============================================================================
# CIRCUMRADIUS
# =============================================================================

def circumradius(points):
    """
    Compute the circumradius of a tetrahedron given 4 points in R³.
    
    Solves the system |c - p_i|² = R² for i = 0,1,2,3 by subtracting
    the equation for p_0 from those for p_1, p_2, p_3, yielding a
    3×3 linear system for the circumcenter c.
    
    Parameters
    ----------
    points : ndarray, shape (4, 3)
        Coordinates of the 4 vertices.
    
    Returns
    -------
    R : float or None
        Circumradius, or None if the system is degenerate (coplanar points).
    """
    p = points
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    for i in range(1, 4):
        A[i - 1] = 2 * (p[i] - p[0])
        b[i - 1] = np.sum(p[i]**2) - np.sum(p[0]**2)
    
    try:
        c = np.linalg.solve(A, b)
        R = np.linalg.norm(c - p[0])
        return R
    except np.linalg.LinAlgError:
        return None


# =============================================================================
# WASSERSTEIN-1 DISTANCE ON GRAPHS (for Ollivier-Ricci)
# =============================================================================

def wasserstein_1_graph(G, u, v, dist_matrix, node_to_idx):
    """
    Compute the Wasserstein-1 (Earth Mover's) distance between uniform 
    distributions on the neighbors of u and v.
    
    Uses the linear programming formulation of optimal transport with
    shortest-path distance as the ground metric.
    
    Parameters
    ----------
    G : networkx.Graph
    u, v : nodes
        Endpoints of the edge.
    dist_matrix : ndarray, shape (n, n)
        Shortest path distance matrix.
    node_to_idx : dict
        Mapping from node labels to matrix indices.
    
    Returns
    -------
    W1 : float
        Wasserstein-1 distance.
    """
    nbrs_u = list(G.neighbors(u))
    nbrs_v = list(G.neighbors(v))
    
    if not nbrs_u or not nbrs_v:
        return 0.0
    
    nu, nv = len(nbrs_u), len(nbrs_v)
    
    # Uniform distributions on neighbor sets
    mu_u = np.ones(nu) / nu
    mu_v = np.ones(nv) / nv
    
    # Cost matrix: shortest path distances between supports
    C = np.zeros((nu, nv))
    for i, ni in enumerate(nbrs_u):
        for j, nj in enumerate(nbrs_v):
            C[i, j] = dist_matrix[node_to_idx[ni], node_to_idx[nj]]
    
    # LP: minimize c^T x subject to Ax = b, x >= 0
    n_vars = nu * nv
    c_vec = C.flatten()
    
    A_eq = np.zeros((nu + nv, n_vars))
    b_eq = np.zeros(nu + nv)
    
    # Row marginal constraints
    for i in range(nu):
        A_eq[i, i * nv:(i + 1) * nv] = 1.0
        b_eq[i] = mu_u[i]
    
    # Column marginal constraints
    for j in range(nv):
        for i in range(nu):
            A_eq[nu + j, i * nv + j] = 1.0
        b_eq[nu + j] = mu_v[j]
    
    result = linprog(c_vec, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(0, None)] * n_vars, method='highs')
    
    return result.fun if result.success else 0.0
