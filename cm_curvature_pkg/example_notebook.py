"""
Example Usage: cm_curvature package
====================================
This script demonstrates how to use the modular cm_curvature package
from a notebook or script environment.

Usage:
    # From the parent directory containing cm_curvature_pkg/
    python example_notebook.py
    
    # Or in a Jupyter notebook:
    # Just copy the cells below.
"""

# %% [markdown]
# # Cayley-Menger Graph Curvature (κ_CM)
# ## Comparison with Forman-Ricci and Ollivier-Ricci on benchmark graphs

# %% Cell 1: Imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from cm_curvature_pkg import cm_curvature, forman_ricci_curvature, ollivier_ricci_curvature
from cm_curvature_pkg.analysis import (
    analyze_graph,
    sensitivity_analysis,
    build_graph_suite,
    summary_table,
    generate_all_figures,
    plot_curvatures_on_graph,
    plot_correlation_scatter,
    plot_discrepancy_map,
    plot_zscore_profiles,
)

print("cm_curvature package loaded successfully.")


# %% Cell 2: Build graph suite
graphs = build_graph_suite(include_model_graphs=True)

print("Available graphs:")
for name, G in graphs.items():
    print(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


# %% Cell 3: Run analysis on all graphs
all_results = {}
for name, G in graphs.items():
    all_results[name] = analyze_graph(G, name, k=5)


# %% Cell 4: Sensitivity analysis
sensitivity_results = {}
for name, G in graphs.items():
    sr, corrs = sensitivity_analysis(G, name)
    sensitivity_results[name] = sr


# %% Cell 5: Summary table
summary_table(graphs, all_results)


# %% Cell 6: Generate all figures
print("\nGenerating figures...")
filepaths = generate_all_figures(
    graphs, all_results, sensitivity_results,
    output_dir='.', prefix='fig', dpi=150
)
print(f"\nSaved {len(filepaths)} figures.")


# %% Cell 7: Quick single-graph analysis (example)
# You can also analyze a single graph directly:

G = nx.karate_club_graph()
results = analyze_graph(G, name='Karate Club', k=5)

# Plot just one graph
fig, layout = plot_curvatures_on_graph(G, results)
plt.suptitle('Zachary Karate Club', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('karate_quick.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved karate_quick.png")


# %% Cell 8: Compute κ_CM only (lightweight usage)
# If you just need CM curvature without the full comparison:

G = nx.les_miserables_graph()
kCM = cm_curvature(G, k=5)

# Top-5 most curved nodes
sorted_nodes = sorted(kCM.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 most curved nodes (Les Misérables):")
for node, kappa in sorted_nodes[:5]:
    print(f"  {node}: κ_CM = {kappa:.4f}")


# %% Cell 9: Custom graph analysis
# You can bring your own graph:

# Example: Visibility Graph from a time series (placeholder)
# from your_vg_module import build_visibility_graph
# G_vg = build_visibility_graph(cosmic_ray_series)
# results_vg = analyze_graph(G_vg, name='CR Visibility Graph', k=5)

# Example: loaded from GML file
# G_dolphins = nx.read_gml('dolphins.gml')
# results_dolphins = analyze_graph(G_dolphins, name='Dolphins', k=5)

print("\nDone! Package is ready for any connected networkx graph.")
