"""
cm_curvature: Cayley-Menger Graph Curvature (κ_CM)
===================================================
A novel discrete curvature metric for graphs based on the Cayley-Menger
determinant applied to local tetrahedra formed by effective resistance distances.

Modules:
    utils       - Mathematical primitives (distances, determinants, embedding)
    curvatures  - Curvature computation (κ_CM, κ_FR, κ_OR)
    analysis    - Comparison, discrepancy analysis, and visualization

Author: D. Sierra-Porta
"""

from .curvatures import cm_curvature, forman_ricci_curvature, ollivier_ricci_curvature
from .analysis import analyze_graph, sensitivity_analysis

__version__ = "0.1.0"
__author__ = "D. Sierra-Porta"
