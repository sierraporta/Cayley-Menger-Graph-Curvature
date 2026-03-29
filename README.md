![license](https://img.shields.io/badge/license-MIT-green?style=flat)
![coverage](https://img.shields.io/badge/coverage-95%25-blue?style=flat&logo=GitHub)
![build](https://img.shields.io/badge/build-passing-brightgreen?style=flat&logo=GitHub)
![](https://img.shields.io/badge/-Python-inactive?style=flat&logo=Python)

# Cayley-Menger Graph Curvature (κ_CM)

A novel discrete curvature metric for graphs based on the Cayley-Menger determinant and effective resistance distances.

## Overview

This repository contains the code and manuscript materials for the paper:

> **Cayley-Menger curvature: A geometric approach to discrete curvature on graphs via local simplex embedding**  
> D. Sierra-Porta  
> Universidad Tecnológica de Bolívar, Cartagena de Indias, Colombia

The method assigns a curvature value κ_CM(v) to each node of a graph by constructing local tetrahedra from nearest neighbors under the effective resistance metric, computing the Cayley-Menger determinant, embedding in ℝ³, and extracting the circumradius. Comparison with Forman-Ricci and Ollivier-Ricci curvatures shows that κ_CM captures genuinely independent geometric information.

## Repository Structure

```
├── cm_curvature_pkg/           # Python package
│   ├── __init__.py
│   ├── utils.py                # Mathematical primitives (distances, CM det, embedding)
│   ├── curvatures.py           # κ_CM, κ_FR, κ_OR computation
│   ├── analysis.py             # Comparison, discrepancy, visualization
│   └── example_notebook.py     # Usage examples (Jupyter-ready cells)
├── run_analysis.py             # Main analysis: 5 benchmark graphs
├── run_extensions.py           # Extended validation, community detection, scalability
├── generate_paper_figures.py   # Generate Figures 1–3 for the paper
├── cm_curvature_paper.tex      # Manuscript (LaTeX)
├── cm_curvature.bib            # Bibliography
└── output/                     # Generated figures (after running scripts)
```

## Quick Start

```bash
pip install numpy scipy networkx matplotlib scikit-learn

# Run the main curvature analysis
python run_analysis.py

# Run extended validation (known-curvature graphs, community detection, scalability)
python run_extensions.py

# Generate paper figures
python generate_paper_figures.py
```

## Minimal Usage

```python
import networkx as nx
from cm_curvature_pkg import cm_curvature

G = nx.karate_club_graph()
kCM = cm_curvature(G, k=5)

for node, kappa in sorted(kCM.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Node {node}: κ_CM = {kappa:.4f}")
```

## Requirements

- Python ≥ 3.9
- numpy, scipy, networkx, matplotlib, scikit-learn

## License

MIT

## Citation

If you use this code, please cite:

```bibtex
@article{sierraporta2026cmcurvature,
  author  = {Sierra-Porta, D.},
  title   = {Cayley-Menger curvature: A geometric approach to discrete 
             curvature on graphs via local simplex embedding},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```
