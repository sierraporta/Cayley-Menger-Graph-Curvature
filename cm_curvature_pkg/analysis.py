"""
cm_curvature.analysis
=====================
Analysis, comparison, and visualization tools for graph curvatures.

Functions:
    analyze_graph           - Full curvature analysis on a single graph
    sensitivity_analysis    - κ_CM stability across k = 4, 5, 6
    compute_discrepancy     - Z-score discrepancy between κ_CM and κ_FR/κ_OR
    build_graph_suite       - Construct benchmark graph collection
    
    plot_curvatures_on_graph    - Node-colored graph layouts
    plot_correlation_scatter    - Pairwise curvature scatter plots
    plot_discrepancy_map        - Discrepancy-colored graph layouts
    plot_sensitivity            - k-sensitivity line plots
    plot_distributions          - Curvature histograms
    plot_zscore_profiles        - Z-score parallel coordinate profiles
    generate_all_figures        - Complete figure suite
"""

import numpy as np
import networkx as nx
from scipy.stats import spearmanr, zscore
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .curvatures import cm_curvature, forman_ricci_curvature, ollivier_ricci_curvature


# =============================================================================
# GRAPH SUITE
# =============================================================================

def build_graph_suite(include_model_graphs=True):
    """
    Build a collection of benchmark graphs for curvature analysis.
    
    Parameters
    ----------
    include_model_graphs : bool, default=True
        Whether to include SBM and Barabási-Albert model graphs.
    
    Returns
    -------
    graphs : dict
        {name: networkx.Graph} ordered collection.
    """
    graphs = {}
    
    # Empirical networks
    graphs['Zachary Karate Club'] = nx.karate_club_graph()
    graphs['Les Misérables'] = nx.les_miserables_graph()
    graphs['Florentine Families'] = nx.florentine_families_graph()
    
    if include_model_graphs:
        # SBM: community structure (analogous to Football network)
        np.random.seed(42)
        sizes = [10, 9, 11, 10, 10, 9, 11, 10, 10, 10]
        n_blocks = len(sizes)
        p_matrix = np.full((n_blocks, n_blocks), 0.03)
        np.fill_diagonal(p_matrix, 0.6)
        G_sbm = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=42)
        G_sbm = nx.convert_node_labels_to_integers(G_sbm)
        graphs['SBM Community'] = G_sbm
        
        # Barabási-Albert: scale-free degree distribution
        G_ba = nx.barabasi_albert_graph(80, 3, seed=42)
        graphs['Barabási-Albert'] = G_ba
    
    # Ensure connectivity
    for name, G in graphs.items():
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            graphs[name] = G.subgraph(largest_cc).copy()
            graphs[name] = nx.convert_node_labels_to_integers(graphs[name])
    
    return graphs


# =============================================================================
# DISCREPANCY ANALYSIS
# =============================================================================

def compute_discrepancy(results):
    """
    Identify nodes with maximum discrepancy between κ_CM and κ_FR/κ_OR.
    
    Computes z-scores of all three curvatures, then measures discrepancy
    as the L2 norm of (z_CM - z_FR, z_CM - z_OR).
    
    Parameters
    ----------
    results : dict
        Output from analyze_graph().
    
    Returns
    -------
    results : dict
        Same dict, enriched with keys:
        'z_cm', 'z_fr', 'z_or'        - z-scored curvature vectors
        'disc_cm_fr', 'disc_cm_or'     - signed discrepancies
        'disc_combined'                 - combined L2 discrepancy
        'disc_rank'                     - node indices sorted by discrepancy (desc)
    """
    n = len(results['nodes'])
    
    cm = results['cm_vals']
    fr = results['fr_vals']
    or_ = results['or_vals']
    
    z_cm = zscore(cm) if np.std(cm) > 1e-12 else np.zeros(n)
    z_fr = zscore(fr) if np.std(fr) > 1e-12 else np.zeros(n)
    z_or = zscore(or_) if np.std(or_) > 1e-12 else np.zeros(n)
    
    disc_cm_fr = z_cm - z_fr
    disc_cm_or = z_cm - z_or
    disc_combined = np.sqrt(disc_cm_fr**2 + disc_cm_or**2)
    
    rank = np.argsort(disc_combined)[::-1]
    
    results.update({
        'z_cm': z_cm, 'z_fr': z_fr, 'z_or': z_or,
        'disc_cm_fr': disc_cm_fr,
        'disc_cm_or': disc_cm_or,
        'disc_combined': disc_combined,
        'disc_rank': rank,
    })
    
    return results


def discrepancy_report(G, results, top_n=8):
    """
    Print a formatted report of nodes with highest CM discrepancy.
    
    Parameters
    ----------
    G : networkx.Graph
    results : dict
        Output from analyze_graph() (with discrepancy computed).
    top_n : int
        Number of top-discrepancy nodes to show.
    
    Returns
    -------
    report : dict
        'high_cm_low_others': nodes with high κ_CM but low κ_FR and κ_OR
        'low_cm_high_others': nodes with low κ_CM but high κ_FR or κ_OR
    """
    nodes = results['nodes']
    rank = results['disc_rank']
    
    print(f"\n  {'─' * 70}")
    print(f"  Top {top_n} nodes by κ_CM discrepancy:")
    print(f"  {'─' * 70}")
    header = (f"  {'Node':<12} {'Deg':>4} {'κ_CM':>8} {'κ_FR':>8} {'κ_OR':>8} "
              f"{'z_CM':>7} {'z_FR':>7} {'z_OR':>7} {'Disc':>7}")
    print(header)
    print(f"  {'─' * 70}")
    
    for i in range(min(top_n, len(nodes))):
        idx = rank[i]
        v = nodes[idx]
        deg = G.degree(v)
        print(f"  {str(v):<12} {deg:>4} "
              f"{results['cm_vals'][idx]:>8.3f} "
              f"{results['fr_vals'][idx]:>8.3f} "
              f"{results['or_vals'][idx]:>8.3f} "
              f"{results['z_cm'][idx]:>+7.2f} "
              f"{results['z_fr'][idx]:>+7.2f} "
              f"{results['z_or'][idx]:>+7.2f} "
              f"{results['disc_combined'][idx]:>7.3f}")
    
    # Classify discrepancy types
    high_cm = []
    low_cm = []
    
    for i in range(len(nodes)):
        z = results['z_cm'][i]
        zf = results['z_fr'][i]
        zo = results['z_or'][i]
        if z > 1.0 and zf < 0 and zo < 0:
            high_cm.append(nodes[i])
        elif z < -1.0 and (zf > 0 or zo > 0):
            low_cm.append(nodes[i])
    
    if high_cm:
        print(f"\n  CM-unique (high κ_CM, low κ_FR & κ_OR): {high_cm}")
    if low_cm:
        print(f"  CM-blind (low κ_CM, high κ_FR or κ_OR): {low_cm}")
    
    return {'high_cm_low_others': high_cm, 'low_cm_high_others': low_cm}


# =============================================================================
# CORE ANALYSIS
# =============================================================================

def analyze_graph(G, name='Graph', k=5, verbose=True):
    """
    Run complete curvature analysis on a graph.
    
    Computes κ_CM, κ_FR, κ_OR for all nodes, their pairwise Spearman
    correlations, and the CM discrepancy analysis.
    
    Parameters
    ----------
    G : networkx.Graph
        Must be connected.
    name : str
        Display name for reporting.
    k : int
        Number of nearest neighbors for κ_CM (default 5).
    verbose : bool
        Print progress and results.
    
    Returns
    -------
    results : dict
        Complete results including curvature vectors, correlations,
        discrepancy analysis, and diagnostic details.
    """
    nodes = list(G.nodes())
    n_tet = len(list(combinations(range(min(k, G.number_of_nodes() - 1)), 3)))
    
    if verbose:
        print(f"\n{'=' * 65}")
        print(f"  {name}")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}, "
              f"k={k} → {n_tet} tet/node")
        print(f"{'=' * 65}")
    
    # Compute curvatures
    if verbose:
        print("  Computing κ_CM...", end=" ", flush=True)
    kCM, details = cm_curvature(G, k=k, return_details=True)
    total_tet = sum(d['n_tetrahedra'] for d in details.values())
    total_emb = sum(d['n_embeddable'] for d in details.values())
    if verbose:
        print(f"done. ({total_emb}/{total_tet} embeddable, "
              f"{100 * total_emb / max(total_tet, 1):.1f}%)")
    
    if verbose:
        print("  Computing κ_FR...", end=" ", flush=True)
    kFR = forman_ricci_curvature(G)
    if verbose:
        print("done.")
    
    if verbose:
        print("  Computing κ_OR...", end=" ", flush=True)
    kOR = ollivier_ricci_curvature(G)
    if verbose:
        print("done.")
    
    # Align as arrays
    cm_vals = np.array([kCM[v] for v in nodes])
    fr_vals = np.array([kFR[v] for v in nodes])
    or_vals = np.array([kOR[v] for v in nodes])
    
    # Correlations
    rho_cm_fr, p_cm_fr = spearmanr(cm_vals, fr_vals)
    rho_cm_or, p_cm_or = spearmanr(cm_vals, or_vals)
    rho_fr_or, p_fr_or = spearmanr(fr_vals, or_vals)
    
    if verbose:
        print(f"\n  Spearman correlations:")
        print(f"    κ_CM vs κ_FR:  ρ = {rho_cm_fr:+.4f}  (p = {p_cm_fr:.2e})")
        print(f"    κ_CM vs κ_OR:  ρ = {rho_cm_or:+.4f}  (p = {p_cm_or:.2e})")
        print(f"    κ_FR vs κ_OR:  ρ = {rho_fr_or:+.4f}  (p = {p_fr_or:.2e})")
    
    results = {
        'name': name, 'nodes': nodes,
        'kCM': kCM, 'kFR': kFR, 'kOR': kOR,
        'cm_vals': cm_vals, 'fr_vals': fr_vals, 'or_vals': or_vals,
        'correlations': {
            'CM_FR': (rho_cm_fr, p_cm_fr),
            'CM_OR': (rho_cm_or, p_cm_or),
            'FR_OR': (rho_fr_or, p_fr_or),
        },
        'details': details,
    }
    
    # Discrepancy
    results = compute_discrepancy(results)
    if verbose:
        discrepancy_report(G, results)
    
    return results


def sensitivity_analysis(G, name='Graph', k_values=(4, 5, 6), verbose=True):
    """
    Assess stability of κ_CM across different k values.
    
    Parameters
    ----------
    G : networkx.Graph
    name : str
    k_values : tuple of int
        Values of k to compare.
    verbose : bool
    
    Returns
    -------
    results_k : dict
        {k: curvature_array} for each k.
    correlations : dict
        {(k1,k2): spearman_rho} for all pairs.
    """
    nodes = list(G.nodes())
    results_k = {}
    
    if verbose:
        print(f"\n  Sensitivity ({name}):")
    
    for k in k_values:
        kCM = cm_curvature(G, k=k)
        vals = np.array([kCM[v] for v in nodes])
        results_k[k] = vals
        n_tet = len(list(combinations(range(min(k, len(nodes) - 1)), 3)))
        if verbose:
            print(f"    k={k} ({n_tet} tet): "
                  f"μ={np.mean(vals):.4f}, σ={np.std(vals):.4f}")
    
    correlations = {}
    for k1, k2 in combinations(k_values, 2):
        rho, _ = spearmanr(results_k[k1], results_k[k2])
        correlations[(k1, k2)] = rho
        if verbose:
            print(f"    ρ(k={k1},k={k2}) = {rho:+.4f}")
    
    return results_k, correlations


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def _get_layout(G, nodes, seed=42):
    """Compute spring layout with consistent seed."""
    return nx.spring_layout(G, seed=seed, k=2.0 / np.sqrt(len(nodes)),
                            iterations=80)


def _get_node_size(n):
    """Adaptive node size based on graph order."""
    return max(20, min(100, 2000 / n))


def plot_curvatures_on_graph(G, results, axes=None, layout=None, figsize=(17, 5)):
    """
    Plot three curvatures as node colors on the graph layout.
    
    Parameters
    ----------
    G : networkx.Graph
    results : dict from analyze_graph()
    axes : array of 3 Axes, optional
    layout : dict, optional (node positions)
    figsize : tuple (only used if axes is None)
    
    Returns
    -------
    fig : Figure (or None if axes provided)
    layout : dict of node positions
    """
    nodes = results['nodes']
    fig = None
    
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    if layout is None:
        layout = _get_layout(G, nodes)
    
    node_size = _get_node_size(len(nodes))
    cmap = 'RdYlBu_r'
    
    for j, (vals, title) in enumerate([
        (results['cm_vals'], 'κ_CM (Cayley-Menger)'),
        (results['fr_vals'], 'κ_FR (Forman-Ricci)'),
        (results['or_vals'], 'κ_OR (Ollivier-Ricci)'),
    ]):
        ax = axes[j]
        vmin, vmax = np.percentile(vals, [5, 95])
        if abs(vmax - vmin) < 1e-6:
            vmin, vmax = vals.min() - 0.01, vals.max() + 0.01
        
        nx.draw_networkx_edges(G, layout, ax=ax, alpha=0.15, width=0.4,
                               edge_color='gray')
        nx.draw_networkx_nodes(G, layout, ax=ax, node_size=node_size,
                               node_color=list(vals), cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               edgecolors='black', linewidths=0.2)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, aspect=25, pad=0.01)
    
    return fig, layout


def plot_correlation_scatter(results, axes=None, figsize=(15, 4.5)):
    """
    Scatter plots of pairwise curvature correlations.
    
    Parameters
    ----------
    results : dict from analyze_graph()
    axes : array of 3 Axes, optional
    """
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    pairs = [
        ('cm_vals', 'fr_vals', 'κ_CM', 'κ_FR', 'CM_FR'),
        ('cm_vals', 'or_vals', 'κ_CM', 'κ_OR', 'CM_OR'),
        ('fr_vals', 'or_vals', 'κ_FR', 'κ_OR', 'FR_OR'),
    ]
    
    for ax, (k1, k2, l1, l2, ck) in zip(axes, pairs):
        x, y = results[k1], results[k2]
        rho, pval = results['correlations'][ck]
        
        ax.scatter(x, y, s=20, alpha=0.6, c='steelblue',
                   edgecolors='navy', linewidths=0.3)
        z = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.7, linewidth=1.2)
        
        ax.set_xlabel(l1, fontsize=9)
        ax.set_ylabel(l2, fontsize=9)
        ax.set_title(f'ρ = {rho:+.3f} (p = {pval:.1e})',
                     fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    
    return fig


def plot_discrepancy_map(G, results, axes=None, layout=None, figsize=(17, 5)):
    """
    Plot discrepancy values on graph layout.
    
    Three panels: Δz(CM-FR), Δz(CM-OR), ||Δz|| combined.
    """
    nodes = results['nodes']
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    if layout is None:
        layout = _get_layout(G, nodes)
    
    node_size = _get_node_size(len(nodes))
    
    disc_data = [
        (results['disc_cm_fr'], 'Δz(CM−FR)', 'RdBu_r'),
        (results['disc_cm_or'], 'Δz(CM−OR)', 'RdBu_r'),
        (results['disc_combined'], '||Δz|| combined', 'hot_r'),
    ]
    
    for j, (vals, title, cmap) in enumerate(disc_data):
        ax = axes[j]
        
        if j < 2:  # signed
            vlim = max(abs(np.percentile(vals, 5)),
                       abs(np.percentile(vals, 95)), 0.1)
            vmin_d, vmax_d = -vlim, vlim
        else:  # combined (positive)
            vmin_d = 0
            vmax_d = max(np.percentile(vals, 95), 0.1)
        
        nx.draw_networkx_edges(G, layout, ax=ax, alpha=0.12, width=0.3,
                               edge_color='gray')
        nx.draw_networkx_nodes(G, layout, ax=ax, node_size=node_size,
                               node_color=list(vals), cmap=cmap,
                               vmin=vmin_d, vmax=vmax_d,
                               edgecolors='black', linewidths=0.2)
        
        # Label top-3 discrepant nodes
        if j == 2:
            for idx in results['disc_rank'][:3]:
                v = nodes[idx]
                pos = layout[v]
                ax.annotate(str(v), xy=pos, fontsize=6, fontweight='bold',
                            color='black', ha='center', va='bottom',
                            xytext=(0, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.15',
                                      fc='yellow', alpha=0.7))
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        
        norm = Normalize(vmin=vmin_d, vmax=vmax_d)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.7, aspect=25, pad=0.01)
    
    return fig, layout


def plot_sensitivity(sensitivity_results, graph_names, axes=None, figsize=None):
    """
    Plot κ_CM sensitivity to k (4, 5, 6).
    """
    n = len(graph_names)
    fig = None
    if axes is None:
        if figsize is None:
            figsize = (5.5 * min(3, n), 4.5 * int(np.ceil(n / 3)))
        n_cols = min(3, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        axes_flat = axes.flatten()
    else:
        axes_flat = np.array(axes).flatten()
    
    for idx, name in enumerate(graph_names):
        ax = axes_flat[idx]
        sk = sensitivity_results[name]
        n_nodes = len(sk[5])
        sorted_idx = np.argsort(sk[5])
        
        for k, color, marker in [(4, '#E74C3C', 'o'),
                                  (5, '#3498DB', 's'),
                                  (6, '#2ECC71', '^')]:
            ax.plot(range(n_nodes), sk[k][sorted_idx],
                    marker=marker, markersize=2.5, alpha=0.7,
                    label=f'k={k}', linewidth=0.7, color=color)
        
        ax.set_xlabel('Node rank (sorted by κ_CM, k=5)', fontsize=8)
        ax.set_ylabel('κ_CM', fontsize=9)
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
    
    # Hide extras
    for idx in range(len(graph_names), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    return fig


def plot_distributions(all_results, graph_names, axes=None, figsize=None):
    """
    Histogram distributions of all three curvatures.
    """
    n = len(graph_names)
    fig = None
    if axes is None:
        if figsize is None:
            figsize = (14, 3.5 * n)
        fig, axes = plt.subplots(n, 3, figsize=figsize)
        if n == 1:
            axes = axes.reshape(1, -1)
    
    for i, name in enumerate(graph_names):
        res = all_results[name]
        for j, (vals, label, color) in enumerate([
            (res['cm_vals'], 'κ_CM', '#3498DB'),
            (res['fr_vals'], 'κ_FR', '#E74C3C'),
            (res['or_vals'], 'κ_OR', '#2ECC71'),
        ]):
            ax = axes[i][j]
            n_bins = min(20, max(8, len(vals) // 4))
            ax.hist(vals, bins=n_bins, color=color, alpha=0.65,
                    edgecolor='black', linewidth=0.5)
            ax.axvline(np.median(vals), color='black', linestyle='--',
                       linewidth=1.2, label=f'median = {np.median(vals):.3f}')
            ax.axvline(np.mean(vals), color='darkred', linestyle=':',
                       linewidth=1, label=f'mean = {np.mean(vals):.3f}')
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel('Count', fontsize=8)
            if i == 0:
                ax.set_title(label, fontsize=10, fontweight='bold')
            ax.legend(fontsize=6.5)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3, axis='y')
        
        axes[i][0].text(-0.3, 0.5, name, transform=axes[i][0].transAxes,
                         fontsize=10, fontweight='bold', rotation=90, va='center')
    
    return fig


def plot_zscore_profiles(all_results, graph_names, top_n=6, axes=None, figsize=None):
    """
    Parallel coordinate plot of z-score profiles for top-discrepancy nodes.
    """
    n = len(graph_names)
    fig = None
    if axes is None:
        if figsize is None:
            figsize = (4.5 * n, 5)
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
    
    for i, name in enumerate(graph_names):
        ax = axes[i]
        res = all_results[name]
        nodes = res['nodes']
        actual_top = min(top_n, len(nodes))
        top_idx = res['disc_rank'][:actual_top]
        
        x_labels = ['z(κ_CM)', 'z(κ_FR)', 'z(κ_OR)']
        x_pos = [0, 1, 2]
        
        colors = plt.cm.Set1(np.linspace(0, 1, actual_top))
        
        for j, idx in enumerate(top_idx):
            z_vals = [res['z_cm'][idx], res['z_fr'][idx], res['z_or'][idx]]
            ax.plot(x_pos, z_vals, 'o-', color=colors[j], linewidth=1.5,
                    markersize=6, alpha=0.8,
                    label=f'{nodes[idx]} (d={res["disc_combined"][idx]:.2f})')
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel('Z-score', fontsize=9)
        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.legend(fontsize=6.5, loc='best',
                  title='Node (disc.)', title_fontsize=6.5)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=8)
    
    return fig


def generate_all_figures(graphs, all_results, sensitivity_results,
                          output_dir='.', prefix='fig', dpi=150):
    """
    Generate the complete suite of publication figures.
    
    Parameters
    ----------
    graphs : dict {name: Graph}
    all_results : dict {name: results}
    sensitivity_results : dict {name: {k: array}}
    output_dir : str
        Directory to save PNGs.
    prefix : str
        Filename prefix.
    dpi : int
    
    Returns
    -------
    filepaths : list of str
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    graph_names = list(graphs.keys())
    n_graphs = len(graph_names)
    filepaths = []
    
    # --- Fig 1: Curvatures on layouts ---
    fig1, ax1 = plt.subplots(n_graphs, 3, figsize=(17, 4.8 * n_graphs))
    fig1.suptitle('Node Curvatures on Graph Layouts',
                   fontsize=15, fontweight='bold', y=0.995)
    for i, name in enumerate(graph_names):
        plot_curvatures_on_graph(graphs[name], all_results[name], axes=ax1[i])
        ax1[i][0].text(-0.15, 0.5, name, transform=ax1[i][0].transAxes,
                        fontsize=11, fontweight='bold', rotation=90,
                        va='center', ha='center')
    fig1.tight_layout(rect=[0.02, 0, 1, 0.98])
    fp = os.path.join(output_dir, f'{prefix}1_curvatures_on_graphs.png')
    fig1.savefig(fp, dpi=dpi, bbox_inches='tight')
    filepaths.append(fp)
    plt.close(fig1)
    print(f"  Saved {os.path.basename(fp)}")
    
    # --- Fig 2: Correlations ---
    fig2, ax2 = plt.subplots(n_graphs, 3, figsize=(15, 4.5 * n_graphs))
    fig2.suptitle('Pairwise Curvature Correlations',
                   fontsize=14, fontweight='bold', y=0.998)
    for i, name in enumerate(graph_names):
        plot_correlation_scatter(all_results[name], axes=ax2[i])
        ax2[i][0].text(-0.3, 0.5, name, transform=ax2[i][0].transAxes,
                        fontsize=10, fontweight='bold', rotation=90, va='center')
    fig2.tight_layout(rect=[0.03, 0, 1, 0.97])
    fp = os.path.join(output_dir, f'{prefix}2_correlations.png')
    fig2.savefig(fp, dpi=dpi, bbox_inches='tight')
    filepaths.append(fp)
    plt.close(fig2)
    print(f"  Saved {os.path.basename(fp)}")
    
    # --- Fig 3: Discrepancy ---
    fig3, ax3 = plt.subplots(n_graphs, 3, figsize=(17, 4.8 * n_graphs))
    fig3.suptitle('Curvature Discrepancy Analysis',
                   fontsize=13, fontweight='bold', y=1.0)
    for i, name in enumerate(graph_names):
        plot_discrepancy_map(graphs[name], all_results[name], axes=ax3[i])
        ax3[i][0].text(-0.15, 0.5, name, transform=ax3[i][0].transAxes,
                        fontsize=10, fontweight='bold', rotation=90, va='center')
    fig3.tight_layout(rect=[0.02, 0, 1, 0.96])
    fp = os.path.join(output_dir, f'{prefix}3_discrepancy.png')
    fig3.savefig(fp, dpi=dpi, bbox_inches='tight')
    filepaths.append(fp)
    plt.close(fig3)
    print(f"  Saved {os.path.basename(fp)}")
    
    # --- Fig 4: Sensitivity ---
    # Extract just the k->array dicts
    sens_arrays = {name: sr[0] if isinstance(sr, tuple) else sr
                    for name, sr in sensitivity_results.items()}
    fig4 = plot_sensitivity(sens_arrays, graph_names)
    if fig4 is not None:
        fig4.suptitle('Sensitivity of κ_CM to k',
                       fontsize=13, fontweight='bold')
        fig4.tight_layout()
        fp = os.path.join(output_dir, f'{prefix}4_sensitivity.png')
        fig4.savefig(fp, dpi=dpi, bbox_inches='tight')
        filepaths.append(fp)
        plt.close(fig4)
        print(f"  Saved {os.path.basename(fp)}")
    
    # --- Fig 5: Distributions ---
    fig5 = plot_distributions(all_results, graph_names)
    if fig5 is not None:
        fig5.suptitle('Curvature Distributions',
                       fontsize=14, fontweight='bold', y=0.998)
        fig5.tight_layout(rect=[0.03, 0, 1, 0.97])
        fp = os.path.join(output_dir, f'{prefix}5_distributions.png')
        fig5.savefig(fp, dpi=dpi, bbox_inches='tight')
        filepaths.append(fp)
        plt.close(fig5)
        print(f"  Saved {os.path.basename(fp)}")
    
    # --- Fig 6: Z-score profiles ---
    fig6 = plot_zscore_profiles(all_results, graph_names)
    if fig6 is not None:
        fig6.suptitle('Z-score Profiles of Top Discrepant Nodes',
                       fontsize=13, fontweight='bold')
        fig6.tight_layout()
        fp = os.path.join(output_dir, f'{prefix}6_zscore_profiles.png')
        fig6.savefig(fp, dpi=dpi, bbox_inches='tight')
        filepaths.append(fp)
        plt.close(fig6)
        print(f"  Saved {os.path.basename(fp)}")
    
    return filepaths


def summary_table(graphs, all_results):
    """
    Print a summary correlation table for all graphs.
    
    Parameters
    ----------
    graphs : dict {name: Graph}
    all_results : dict {name: results}
    """
    print(f"\n{'=' * 65}")
    print("  SUMMARY: SPEARMAN CORRELATION TABLE")
    print(f"{'=' * 65}")
    print(f"  {'Graph':<25} {'N':>4} {'E':>5}  "
          f"{'CM↔FR':>8} {'CM↔OR':>8} {'FR↔OR':>8}")
    print(f"  {'─' * 25} {'─' * 4} {'─' * 5}  {'─' * 8} {'─' * 8} {'─' * 8}")
    
    for name, G in graphs.items():
        c = all_results[name]['correlations']
        print(f"  {name:<25} {G.number_of_nodes():>4} {G.number_of_edges():>5}  "
              f"{c['CM_FR'][0]:>+8.3f} {c['CM_OR'][0]:>+8.3f} "
              f"{c['FR_OR'][0]:>+8.3f}")
