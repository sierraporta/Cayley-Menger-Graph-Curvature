"""
run_analysis.py
===============
Main script to run the complete Cayley-Menger curvature analysis.

Usage:
    python run_analysis.py

Outputs:
    - 6 PNG figures in ./output/
    - Console summary of correlations and discrepancy reports

Requirements:
    - cm_curvature_pkg/ must be in the same directory (or in PYTHONPATH)
    - numpy, scipy, networkx, matplotlib
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Ensure the package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cm_curvature_pkg.analysis import (
    build_graph_suite,
    analyze_graph,
    sensitivity_analysis,
    generate_all_figures,
    summary_table,
)


def main():
    # ── Configuration ────────────────────────────────────────────────
    K = 5                       # nearest neighbors (4, 5, or 6)
    OUTPUT_DIR = './output'     # where figures are saved
    INCLUDE_MODEL = True        # include SBM and Barabási-Albert
    
    # ── Build graphs ─────────────────────────────────────────────────
    print("=" * 65)
    print("  CAYLEY-MENGER GRAPH CURVATURE (κ_CM)")
    print("  Comparison with Forman-Ricci and Ollivier-Ricci")
    print("=" * 65)
    
    graphs = build_graph_suite(include_model_graphs=INCLUDE_MODEL)
    
    # ── To add real benchmark graphs, uncomment and adjust paths: ───
    # import networkx as nx
    # graphs['Dolphin Social Network'] = nx.read_gml('data/dolphins.gml')
    # graphs['College Football'] = nx.read_gml('data/football.gml')
    # graphs['US Politics Books'] = nx.read_gml('data/polbooks.gml')
    
    # ── Run curvature analysis ───────────────────────────────────────
    all_results = {}
    for name, G in graphs.items():
        all_results[name] = analyze_graph(G, name, k=K)
    
    # ── Sensitivity analysis (k = 4, 5, 6) ──────────────────────────
    print(f"\n{'=' * 65}")
    print("  SENSITIVITY ANALYSIS (k = 4, 5, 6)")
    print(f"{'=' * 65}")
    
    sensitivity_results = {}
    for name, G in graphs.items():
        sr, corrs = sensitivity_analysis(G, name)
        sensitivity_results[name] = sr
    
    # ── Summary table ────────────────────────────────────────────────
    summary_table(graphs, all_results)
    
    # ── Generate figures ─────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  GENERATING FIGURES")
    print(f"{'=' * 65}")
    
    filepaths = generate_all_figures(
        graphs, all_results, sensitivity_results,
        output_dir=OUTPUT_DIR, prefix='fig', dpi=150
    )
    
    print(f"\n  Done! {len(filepaths)} figures saved to {OUTPUT_DIR}/")
    for fp in filepaths:
        print(f"    {os.path.basename(fp)}")


if __name__ == "__main__":
    main()
