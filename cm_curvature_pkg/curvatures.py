"""
cm_curvature.curvatures
=======================
Discrete curvature computations on graphs.

Functions:
    cm_curvature                - Cayley-Menger curvature (κ_CM) [novel]
    forman_ricci_curvature      - Forman-Ricci curvature (κ_FR)
    ollivier_ricci_curvature    - Ollivier-Ricci curvature (κ_OR)
"""

import numpy as np
import networkx as nx
from itertools import combinations

from .utils import (
    effective_resistance_matrix,
    cayley_menger_determinant,
    tetrahedron_volume_from_cm,
    embed_tetrahedron,
    circumradius,
    wasserstein_1_graph,
)


# =============================================================================
# CAYLEY-MENGER CURVATURE (κ_CM)
# =============================================================================

def cm_curvature(G, k=5, return_details=False):
    """
    Compute Cayley-Menger curvature for all nodes in a graph.
    
    Pipeline:
        1. Compute effective resistance distance matrix (Laplacian pseudoinverse)
        2. For each node v, identify k nearest neighbors by eff. resistance
        3. For each C(k,3) triples of neighbors, form tetrahedron (v, u1, u2, u3)
        4. Compute CM determinant → embed in R³ → circumradius R
        5. κ_CM(v) = median(1/R) over all tetrahedra of v
    
    Parameters
    ----------
    G : networkx.Graph
        Input graph. Must be connected.
    k : int, default=5
        Number of nearest neighbors to consider per node.
        k=5 → C(5,3) = 10 tetrahedra per node (recommended default).
        k=4 → 4 tetrahedra, k=6 → 20 tetrahedra.
    return_details : bool, default=False
        If True, return additional diagnostic information per node
        (embeddability, individual tetrahedra volumes, circumradii).
    
    Returns
    -------
    curvatures : dict
        {node: κ_CM value} for each node in G.
    details : dict (only if return_details=True)
        {node: {'n_tetrahedra': int, 'n_embeddable': int, 
                'tetrahedra': list of dicts}}
    
    Notes
    -----
    - Uses effective resistance distance (continuous, structure-aware)
      rather than shortest-path distance (integer, often degenerate).
    - Aggregation by median provides robustness to outlier tetrahedra.
    - Non-embeddable configurations (Δ_CM < 0) indicate non-Euclidean
      local geometry. A fallback curvature is assigned based on |Δ_CM|
      with negative sign.
    - Computational complexity: O(n³) for the pseudoinverse, then
      O(n · C(k,3)) for the tetrahedra. With k=5, this is O(10n).
    """
    Omega, nodes = effective_resistance_matrix(G)
    n = len(nodes)
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    
    curvatures = {}
    details = {} if return_details else None
    
    for v in nodes:
        vi = node_to_idx[v]
        
        # Distances from v to all other nodes
        dists = Omega[vi].copy()
        dists[vi] = np.inf  # exclude self
        
        # Select k nearest neighbors
        k_actual = min(k, n - 1)
        nearest_idx = np.argsort(dists)[:k_actual]
        
        curvature_values = []
        tet_details = []
        
        for triple in combinations(nearest_idx, 3):
            u1_idx, u2_idx, u3_idx = triple
            indices = [vi, u1_idx, u2_idx, u3_idx]
            
            # Extract 6 pairwise distances and squared distances
            d = {}
            d_sq = {}
            for a, b in combinations(range(4), 2):
                raw = Omega[indices[a], indices[b]]
                dist_val = np.sqrt(max(raw, 0.0))
                d[(a, b)] = dist_val
                d_sq[(a, b)] = raw
            
            # Cayley-Menger determinant
            delta_cm = cayley_menger_determinant(d_sq)
            
            # Attempt Euclidean embedding → circumradius
            points = embed_tetrahedron(d)
            if points is not None:
                R = circumradius(points)
                if R is not None and R > 1e-12:
                    kappa = 1.0 / R
                    curvature_values.append(kappa)
                    if return_details:
                        vol, _ = tetrahedron_volume_from_cm(delta_cm)
                        tet_details.append({
                            'delta_cm': delta_cm,
                            'volume': vol,
                            'circumradius': R,
                            'kappa': kappa,
                            'embeddable': True,
                        })
            else:
                # Non-embeddable: fallback using |Δ_CM| with negative sign
                if abs(delta_cm) > 1e-15:
                    mean_d_sq = np.mean([d_sq[key] for key in d_sq])
                    if mean_d_sq > 1e-12:
                        kappa_fb = -np.abs(delta_cm) ** (1 / 5) / mean_d_sq
                        curvature_values.append(kappa_fb)
                        if return_details:
                            tet_details.append({
                                'delta_cm': delta_cm,
                                'volume': None,
                                'circumradius': None,
                                'kappa': kappa_fb,
                                'embeddable': False,
                            })
        
        # Aggregate: median for robustness
        curvatures[v] = np.median(curvature_values) if curvature_values else 0.0
        
        if return_details:
            details[v] = {
                'n_tetrahedra': len(curvature_values),
                'n_embeddable': sum(
                    1 for t in tet_details if t.get('embeddable', False)
                ),
                'tetrahedra': tet_details,
            }
    
    if return_details:
        return curvatures, details
    return curvatures


# =============================================================================
# FORMAN-RICCI CURVATURE (κ_FR)
# =============================================================================

def forman_ricci_curvature(G):
    """
    Compute Forman-Ricci curvature for each node in an unweighted graph.
    
    For each edge (u, v):
        κ_FR(u, v) = 4 - deg(u) - deg(v) + 3 · |Δ(u,v)|
    
    where |Δ(u,v)| is the number of triangles containing the edge.
    
    Node curvature is the mean of all incident edge curvatures.
    
    Parameters
    ----------
    G : networkx.Graph
    
    Returns
    -------
    curvatures : dict
        {node: κ_FR value}
    
    References
    ----------
    Sreejith, R.P. et al. (2016). Forman curvature for complex networks.
    J. Stat. Mech., 063206.
    """
    edge_curv = {}
    
    for u, v in G.edges():
        common = len(list(nx.common_neighbors(G, u, v)))
        kappa = 4 - G.degree(u) - G.degree(v) + 3 * common
        edge_curv[(u, v)] = kappa
        edge_curv[(v, u)] = kappa
    
    node_curv = {}
    for node in G.nodes():
        incident = [edge_curv[(u, v)]
                     for u, v in G.edges(node)
                     if (u, v) in edge_curv]
        node_curv[node] = np.mean(incident) if incident else 0.0
    
    return node_curv


# =============================================================================
# OLLIVIER-RICCI CURVATURE (κ_OR)
# =============================================================================

def ollivier_ricci_curvature(G):
    """
    Compute Ollivier-Ricci curvature for each node.
    
    For each edge (u, v):
        κ_OR(u, v) = 1 - W₁(μ_u, μ_v) / d(u, v)
    
    where μ_u is the uniform distribution over neighbors of u,
    W₁ is the Wasserstein-1 distance with shortest-path ground metric,
    and d(u,v) is the shortest-path distance.
    
    Node curvature is the mean of all incident edge curvatures.
    
    Parameters
    ----------
    G : networkx.Graph
    
    Returns
    -------
    curvatures : dict
        {node: κ_OR value}
    
    Notes
    -----
    - Uses linear programming (scipy.optimize.linprog) for the optimal
      transport computation on each edge.
    - Computational complexity: O(|E| · max_deg²) for the LP solves.
    
    References
    ----------
    Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces.
    J. Funct. Anal., 256(3), 810-864.
    """
    nodes = list(G.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    
    # Shortest path distance matrix
    dist_matrix = np.zeros((n, n))
    sp = dict(nx.shortest_path_length(G))
    for u in nodes:
        for v in nodes:
            dist_matrix[node_to_idx[u], node_to_idx[v]] = sp.get(u, {}).get(v, np.inf)
    
    edge_curv = {}
    for u, v in G.edges():
        d_uv = dist_matrix[node_to_idx[u], node_to_idx[v]]
        if d_uv > 0:
            W1 = wasserstein_1_graph(G, u, v, dist_matrix, node_to_idx)
            kappa = 1.0 - W1 / d_uv
        else:
            kappa = 0.0
        edge_curv[(u, v)] = kappa
        edge_curv[(v, u)] = kappa
    
    node_curv = {}
    for node in G.nodes():
        incident = [edge_curv[(u, v)]
                     for u, v in G.edges(node)
                     if (u, v) in edge_curv]
        node_curv[node] = np.mean(incident) if incident else 0.0
    
    return node_curv
