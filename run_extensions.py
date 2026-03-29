"""
Extended Validation for Cayley-Menger Graph Curvature (κ_CM)
============================================================

Three extensions to strengthen the methodology:

    1. VALIDATION ON GRAPHS WITH KNOWN CURVATURE
       - Grid lattice (expected: ~flat, curvature ~0)
       - Balanced tree (expected: negative curvature)
       - Complete graph (expected: positive curvature, sphere-like)
       - Cycle graph (expected: ~flat/zero curvature)
    
    2. USE CASE: COMMUNITY DETECTION
       - Compare community detection accuracy using κ_CM, κ_FR, κ_OR
         as node features, alone and combined.
       - Benchmark: SBM with known ground-truth communities.
    
    3. SCALABILITY ANALYSIS
       - Timing benchmarks for n = 100, 250, 500, 1000, 1500, 2000, 3500, 5000
       - Component-wise profiling (pseudoinverse, kNN, tetrahedra)

Author: D. Sierra-Porta
"""

import sys
import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import spearmanr, zscore
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cm_curvature_pkg.curvatures import cm_curvature, forman_ricci_curvature, ollivier_ricci_curvature
from cm_curvature_pkg.utils import effective_resistance_matrix


# =============================================================================
# PART 1: GRAPHS WITH KNOWN CURVATURE
# =============================================================================

def build_known_curvature_graphs():
    """
    Construct graphs with analytically known or expected curvature properties.
    
    Returns dict of (graph, expected_curvature_sign, description).
    """
    graphs = {}
    
    # ── Grid lattice: discrete analog of flat R² ──
    # Interior nodes: 4 neighbors in regular pattern → curvature ≈ 0
    # Boundary nodes: fewer neighbors → boundary effects
    G_grid = nx.grid_2d_graph(10, 10)
    G_grid = nx.convert_node_labels_to_integers(G_grid)
    graphs['Grid 10×10'] = {
        'graph': G_grid,
        'expected': 'near-zero (flat)',
        'description': 'Discrete R² lattice. Interior nodes have uniform'
                       ' 4-neighborhood, yielding zero Gaussian curvature.'
    }
    
    # ── Balanced tree: discrete hyperbolic space ──
    # Exponential growth of neighborhoods → negative curvature
    G_tree = nx.balanced_tree(3, 4)
    graphs['Balanced Tree (3,4)'] = {
        'graph': G_tree,
        'expected': 'negative (hyperbolic)',
        'description': 'Regular tree with branching factor 3. Exponential '
                       'volume growth is the hallmark of negative curvature.'
    }
    
    # ── Complete graph: discrete sphere ──
    # Maximal connectivity, all distances = 1 (geodesic)
    # In effective resistance, all distances = 2/n → uniform geometry
    # Positive curvature expected (finite diameter, polynomial growth)
    G_complete = nx.complete_graph(30)
    graphs['Complete K₃₀'] = {
        'graph': G_complete,
        'expected': 'positive (spherical)',
        'description': 'K₃₀: maximally connected, finite diameter. Analogous '
                       'to a positively curved space (sphere).'
    }
    
    # ── Cycle graph: discrete S¹, constant zero curvature ──
    # 1D manifold with no intrinsic curvature (all nodes equivalent)
    G_cycle = nx.cycle_graph(50)
    graphs['Cycle C₅₀'] = {
        'graph': G_cycle,
        'expected': 'zero (flat S¹)',
        'description': 'Discrete circle. All nodes are structurally identical. '
                       'Expected: uniform curvature (zero variance).'
    }
    
    # ── Random Geometric Graph: embeds in R² with local Euclidean structure ──
    G_rgg = nx.random_geometric_graph(100, 0.25, seed=42)
    if not nx.is_connected(G_rgg):
        G_rgg = G_rgg.subgraph(max(nx.connected_components(G_rgg), key=len)).copy()
        G_rgg = nx.convert_node_labels_to_integers(G_rgg)
    graphs['RGG (n=100, r=0.25)'] = {
        'graph': G_rgg,
        'expected': 'near-zero interior, positive at boundary',
        'description': 'Points in unit square connected within radius 0.25. '
                       'Interior approximates flat R², boundary introduces curvature.'
    }
    
    return graphs


def run_known_curvature_analysis(output_dir='.'):
    """Run curvature analysis on graphs with known geometric properties."""
    
    print("\n" + "=" * 70)
    print("  PART 1: VALIDATION ON GRAPHS WITH KNOWN CURVATURE")
    print("=" * 70)
    
    known_graphs = build_known_curvature_graphs()
    results = {}
    
    for name, info in known_graphs.items():
        G = info['graph']
        expected = info['expected']
        
        print(f"\n  ── {name} ──")
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  Expected curvature: {expected}")
        
        # Compute all three curvatures
        kCM = cm_curvature(G, k=5)
        kFR = forman_ricci_curvature(G)
        kOR = ollivier_ricci_curvature(G)
        
        nodes = list(G.nodes())
        cm_vals = np.array([kCM[v] for v in nodes])
        fr_vals = np.array([kFR[v] for v in nodes])
        or_vals = np.array([kOR[v] for v in nodes])
        
        # Statistics
        print(f"\n  {'Metric':<8} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'CV':>10}")
        print(f"  {'─' * 58}")
        for label, vals in [('κ_CM', cm_vals), ('κ_FR', fr_vals), ('κ_OR', or_vals)]:
            cv = np.std(vals) / abs(np.mean(vals)) if abs(np.mean(vals)) > 1e-10 else np.inf
            print(f"  {label:<8} {np.mean(vals):>10.4f} {np.std(vals):>10.4f} "
                  f"{np.min(vals):>10.4f} {np.max(vals):>10.4f} {cv:>10.3f}")
        
        # Spearman correlations
        rho_cf, _ = spearmanr(cm_vals, fr_vals)
        rho_co, _ = spearmanr(cm_vals, or_vals)
        rho_fo, _ = spearmanr(fr_vals, or_vals)
        print(f"\n  Spearman: CM↔FR={rho_cf:+.3f}, CM↔OR={rho_co:+.3f}, FR↔OR={rho_fo:+.3f}")
        
        results[name] = {
            'graph': G, 'expected': expected, 'nodes': nodes,
            'cm_vals': cm_vals, 'fr_vals': fr_vals, 'or_vals': or_vals,
            'kCM': kCM, 'kFR': kFR, 'kOR': kOR,
        }
    
    # ── Figure: Known curvature validation ──
    names = list(known_graphs.keys())
    n_graphs = len(names)
    
    fig, axes = plt.subplots(n_graphs, 4, figsize=(20, 4.2 * n_graphs))
    fig.suptitle('Validation on Graphs with Known Curvature Properties',
                  fontsize=15, fontweight='bold', y=0.995)
    
    for i, name in enumerate(names):
        res = results[name]
        G = res['graph']
        nodes = res['nodes']
        
        # Use appropriate layout for each graph type
        if 'Grid' in name:
            n_side = int(np.sqrt(len(nodes)))
            layout = {v: (v % n_side, v // n_side) for v in nodes}
        elif 'Cycle' in name:
            layout = nx.circular_layout(G)
        elif 'Tree' in name:
            try:
                layout = nx.planar_layout(G)
            except:
                layout = nx.spring_layout(G, seed=42, iterations=100)
        else:
            layout = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(len(nodes)),
                                       iterations=80)
        
        node_size = max(15, min(80, 5000 / len(nodes)))
        
        # Columns: κ_CM on graph, κ_FR on graph, κ_OR on graph, distributions
        for j, (vals, label, cmap) in enumerate([
            (res['cm_vals'], 'κ_CM', 'RdYlBu_r'),
            (res['fr_vals'], 'κ_FR', 'RdYlBu_r'),
            (res['or_vals'], 'κ_OR', 'RdYlBu_r'),
        ]):
            ax = axes[i][j]
            vmin, vmax = np.percentile(vals, [5, 95])
            if abs(vmax - vmin) < 1e-6:
                vmin, vmax = vals.min() - 0.01, vals.max() + 0.01
            
            nx.draw_networkx_edges(G, layout, ax=ax, alpha=0.1, width=0.3,
                                    edge_color='gray')
            nx.draw_networkx_nodes(G, layout, ax=ax, node_size=node_size,
                                    node_color=list(vals), cmap=cmap,
                                    vmin=vmin, vmax=vmax,
                                    edgecolors='black', linewidths=0.15)
            if i == 0:
                ax.set_title(label, fontsize=11, fontweight='bold')
            ax.axis('off')
            sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, shrink=0.65, aspect=20, pad=0.01)
        
        # Column 4: Overlaid distributions
        ax = axes[i][3]
        for vals, label, color in [
            (res['cm_vals'], 'κ_CM', '#3498DB'),
            (res['fr_vals'], 'κ_FR', '#E74C3C'),
            (res['or_vals'], 'κ_OR', '#2ECC71'),
        ]:
            # Normalize to z-scores for comparable display
            if np.std(vals) > 1e-10:
                z = zscore(vals)
                ax.hist(z, bins=15, alpha=0.4, color=color, label=label,
                        edgecolor='black', linewidth=0.3)
            else:
                # Constant value: plot a single bar
                ax.axvline(0, color=color, linewidth=2.5, alpha=0.7,
                           label=f'{label} (const={np.mean(vals):.3f})')
        
        ax.set_xlabel('Z-score', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        if i == 0:
            ax.set_title('Distributions (z-scored)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(labelsize=7)
        
        # Row label
        axes[i][0].text(-0.2, 0.5, f'{name}\n({res["expected"]})',
                         transform=axes[i][0].transAxes,
                         fontsize=9, fontweight='bold', rotation=90,
                         va='center', ha='center',
                         bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                                   alpha=0.8))
    
    fig.tight_layout(rect=[0.04, 0, 1, 0.97])
    fp = os.path.join(output_dir, 'fig7_known_curvature_validation.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {os.path.basename(fp)}")
    
    # ── Summary table ──
    print(f"\n  {'─' * 70}")
    print(f"  CURVATURE VALIDATION SUMMARY")
    print(f"  {'─' * 70}")
    print(f"  {'Graph':<25} {'Expected':<20} {'κ_CM mean':>10} {'κ_CM σ':>8} {'κ_CM CV':>8}")
    print(f"  {'─' * 25} {'─' * 20} {'─' * 10} {'─' * 8} {'─' * 8}")
    
    for name in names:
        res = results[name]
        expected = known_graphs[name]['expected'].split(' ')[0]
        mean = np.mean(res['cm_vals'])
        std = np.std(res['cm_vals'])
        cv = std / abs(mean) if abs(mean) > 1e-10 else float('inf')
        print(f"  {name:<25} {expected:<20} {mean:>10.4f} {std:>8.4f} {cv:>8.3f}")
    
    return results


# =============================================================================
# PART 2: COMMUNITY DETECTION USE CASE
# =============================================================================

def build_community_graphs():
    """Build SBM graphs with known ground-truth communities."""
    graphs = {}
    
    # Clear community structure
    np.random.seed(42)
    sizes_clear = [30, 30, 30, 30]  # 120 nodes, 4 communities
    n_b = len(sizes_clear)
    p_clear = np.full((n_b, n_b), 0.02)
    np.fill_diagonal(p_clear, 0.5)
    G1 = nx.stochastic_block_model(sizes_clear, p_clear.tolist(), seed=42)
    labels1 = [G1.nodes[v]['block'] for v in G1.nodes()]
    graphs['SBM Clear (4 comm.)'] = {'graph': G1, 'labels': labels1}
    
    # Fuzzy community structure (harder)
    p_fuzzy = np.full((n_b, n_b), 0.08)
    np.fill_diagonal(p_fuzzy, 0.3)
    G2 = nx.stochastic_block_model(sizes_clear, p_fuzzy.tolist(), seed=123)
    labels2 = [G2.nodes[v]['block'] for v in G2.nodes()]
    graphs['SBM Fuzzy (4 comm.)'] = {'graph': G2, 'labels': labels2}
    
    # Many communities
    sizes_many = [15] * 8  # 120 nodes, 8 communities
    n_b2 = 8
    p_many = np.full((n_b2, n_b2), 0.02)
    np.fill_diagonal(p_many, 0.5)
    G3 = nx.stochastic_block_model(sizes_many, p_many.tolist(), seed=456)
    labels3 = [G3.nodes[v]['block'] for v in G3.nodes()]
    graphs['SBM 8 communities'] = {'graph': G3, 'labels': labels3}
    
    return graphs


def curvature_based_clustering(G, curvatures_dict, n_clusters, nodes):
    """
    Cluster nodes using curvature values as features via 
    hierarchical agglomerative clustering.
    
    Parameters
    ----------
    G : networkx.Graph
    curvatures_dict : dict of {feature_name: {node: value}}
    n_clusters : int
    nodes : list
    
    Returns
    -------
    labels_pred : array of cluster assignments
    """
    # Build feature matrix
    n = len(nodes)
    n_features = len(curvatures_dict)
    X = np.zeros((n, n_features))
    
    for j, (name, curv) in enumerate(curvatures_dict.items()):
        for i, v in enumerate(nodes):
            X[i, j] = curv[v]
    
    # Z-score normalize
    for j in range(n_features):
        col_std = np.std(X[:, j])
        if col_std > 1e-10:
            X[:, j] = (X[:, j] - np.mean(X[:, j])) / col_std
    
    # Hierarchical clustering
    Z = linkage(X, method='ward')
    labels_pred = fcluster(Z, t=n_clusters, criterion='maxclust') - 1
    
    return labels_pred


def run_community_detection_analysis(output_dir='.'):
    """
    Compare community detection using different curvature feature sets.
    """
    print("\n" + "=" * 70)
    print("  PART 2: USE CASE — COMMUNITY DETECTION WITH CURVATURE FEATURES")
    print("=" * 70)
    
    community_graphs = build_community_graphs()
    all_scores = {}
    
    for gname, ginfo in community_graphs.items():
        G = ginfo['graph']
        labels_true = np.array(ginfo['labels'])
        n_clusters = len(set(labels_true))
        nodes = list(G.nodes())
        
        # Ensure connected
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            nodes = list(G.nodes())
            labels_true = np.array([ginfo['labels'][v] for v in nodes])
        
        print(f"\n  ── {gname} ──")
        print(f"  Nodes: {len(nodes)}, Communities: {n_clusters}")
        
        # Compute curvatures
        kCM = cm_curvature(G, k=5)
        kFR = forman_ricci_curvature(G)
        kOR = ollivier_ricci_curvature(G)
        
        # Also compute degree as baseline
        deg = {v: G.degree(v) for v in G.nodes()}
        
        # Define feature sets to compare
        feature_sets = {
            'Degree only':      {'degree': deg},
            'κ_FR only':        {'FR': kFR},
            'κ_OR only':        {'OR': kOR},
            'κ_CM only':        {'CM': kCM},
            'κ_FR + κ_OR':      {'FR': kFR, 'OR': kOR},
            'κ_CM + κ_FR':      {'CM': kCM, 'FR': kFR},
            'κ_CM + κ_OR':      {'CM': kCM, 'OR': kOR},
            'All three (CM+FR+OR)': {'CM': kCM, 'FR': kFR, 'OR': kOR},
            'All + Degree':     {'CM': kCM, 'FR': kFR, 'OR': kOR, 'deg': deg},
        }
        
        scores = {}
        print(f"\n  {'Feature set':<25} {'ARI':>8} {'NMI':>8}")
        print(f"  {'─' * 25} {'─' * 8} {'─' * 8}")
        
        for fs_name, fs_curvs in feature_sets.items():
            labels_pred = curvature_based_clustering(G, fs_curvs, n_clusters, nodes)
            ari = adjusted_rand_score(labels_true, labels_pred)
            nmi = normalized_mutual_info_score(labels_true, labels_pred)
            scores[fs_name] = {'ARI': ari, 'NMI': nmi}
            print(f"  {fs_name:<25} {ari:>8.4f} {nmi:>8.4f}")
        
        all_scores[gname] = scores
    
    # ── Figure: Community detection comparison ──
    fig, axes = plt.subplots(1, len(community_graphs), figsize=(7 * len(community_graphs), 6))
    if len(community_graphs) == 1:
        axes = [axes]
    
    fig.suptitle('Community Detection Accuracy Using Curvature Features',
                  fontsize=14, fontweight='bold', y=1.02)
    
    for i, (gname, scores) in enumerate(all_scores.items()):
        ax = axes[i]
        
        fs_names = list(scores.keys())
        ari_vals = [scores[fs]['ARI'] for fs in fs_names]
        nmi_vals = [scores[fs]['NMI'] for fs in fs_names]
        
        x = np.arange(len(fs_names))
        width = 0.35
        
        bars1 = ax.barh(x + width / 2, ari_vals, width, label='ARI',
                          color='#3498DB', alpha=0.8, edgecolor='black', linewidth=0.3)
        bars2 = ax.barh(x - width / 2, nmi_vals, width, label='NMI',
                          color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.3)
        
        ax.set_yticks(x)
        ax.set_yticklabels(fs_names, fontsize=8)
        ax.set_xlabel('Score', fontsize=9)
        ax.set_title(gname, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(labelsize=7)
        
        # Highlight best
        best_ari_idx = np.argmax(ari_vals)
        ax.barh(best_ari_idx + width / 2, ari_vals[best_ari_idx], width,
                color='#3498DB', alpha=1.0, edgecolor='gold', linewidth=2)
    
    fig.tight_layout()
    fp = os.path.join(output_dir, 'fig8_community_detection.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {os.path.basename(fp)}")
    
    return all_scores


# =============================================================================
# PART 3: SCALABILITY ANALYSIS
# =============================================================================

def run_scalability_analysis(output_dir='.'):
    """
    Benchmark timing of κ_CM, κ_FR, κ_OR for increasing graph sizes.
    Also profiles individual pipeline steps.
    """
    print("\n" + "=" * 70)
    print("  PART 3: SCALABILITY ANALYSIS")
    print("=" * 70)
    
    sizes = [50, 100, 250, 500, 750, 1000, 1500, 2000, 3500, 5000]
    m = 3  # BA parameter
    k = 5
    
    timing = {
        'n': [], 'edges': [],
        'cm_total': [], 'cm_pinv': [], 'cm_tetra': [],
        'fr_total': [], 'or_total': [],
    }
    
    print(f"\n  {'n':>6} {'|E|':>7}  {'κ_CM':>8} {'(pinv)':>8} {'(tetra)':>8}  "
          f"{'κ_FR':>8} {'κ_OR':>8}")
    print(f"  {'─' * 6} {'─' * 7}  {'─' * 8} {'─' * 8} {'─' * 8}  {'─' * 8} {'─' * 8}")
    
    for n in sizes:
        G = nx.barabasi_albert_graph(n, m, seed=42)
        n_edges = G.number_of_edges()
        
        timing['n'].append(n)
        timing['edges'].append(n_edges)
        
        # ── κ_CM with profiling ──
        t0 = time.time()
        
        # Step 1: Pseudoinverse
        t_pinv_start = time.time()
        Omega, nodes = effective_resistance_matrix(G)
        t_pinv = time.time() - t_pinv_start
        
        # Step 2+3+4: kNN + tetrahedra + circumradius
        t_tetra_start = time.time()
        kCM = cm_curvature(G, k=k)  # recomputes Omega internally but measures total
        t_tetra = time.time() - t_tetra_start
        
        t_cm_total = time.time() - t0
        timing['cm_total'].append(t_cm_total)
        timing['cm_pinv'].append(t_pinv)
        timing['cm_tetra'].append(t_tetra - t_pinv)  # approximate tetra-only time
        
        # ── κ_FR ──
        t0 = time.time()
        kFR = forman_ricci_curvature(G)
        t_fr = time.time() - t0
        timing['fr_total'].append(t_fr)
        
        # ── κ_OR ──
        t0 = time.time()
        kOR = ollivier_ricci_curvature(G)
        t_or = time.time() - t0
        timing['or_total'].append(t_or)
        
        print(f"  {n:>6} {n_edges:>7}  {t_cm_total:>7.2f}s {t_pinv:>7.2f}s "
              f"{t_tetra - t_pinv:>7.2f}s  {t_fr:>7.2f}s {t_or:>7.2f}s")
    
    # ── Figure: Scaling ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Scalability Analysis', fontsize=14, fontweight='bold')
    
    ns = timing['n']
    
    # Panel 1: Total time comparison
    ax = axes[0]
    ax.plot(ns, timing['cm_total'], 'o-', color='#3498DB', linewidth=2,
            markersize=7, label='κ_CM (total)')
    ax.plot(ns, timing['fr_total'], 's--', color='#E74C3C', linewidth=2,
            markersize=7, label='κ_FR')
    ax.plot(ns, timing['or_total'], '^-.', color='#2ECC71', linewidth=2,
            markersize=7, label='κ_OR')
    ax.set_xlabel('Number of nodes', fontsize=10)
    ax.set_ylabel('Time (seconds)', fontsize=10)
    ax.set_title('Total Computation Time', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.tick_params(labelsize=8)
    
    # Panel 2: κ_CM component breakdown
    ax = axes[1]
    ax.stackplot(ns,
                  timing['cm_pinv'],
                  [t - p for t, p in zip(timing['cm_total'], timing['cm_pinv'])],
                  labels=['Pseudoinverse L⁺', 'kNN + Tetrahedra + 1/R'],
                  colors=['#3498DB', '#85C1E9'], alpha=0.8)
    ax.set_xlabel('Number of nodes', fontsize=10)
    ax.set_ylabel('Time (seconds)', fontsize=10)
    ax.set_title('κ_CM Component Breakdown', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    # Panel 3: Scaling exponent (log-log)
    ax = axes[2]
    for label, times, color, marker in [
        ('κ_CM', timing['cm_total'], '#3498DB', 'o'),
        ('κ_FR', timing['fr_total'], '#E74C3C', 's'),
        ('κ_OR', timing['or_total'], '#2ECC71', '^'),
    ]:
        log_n = np.log10(ns)
        log_t = np.log10([max(t, 1e-6) for t in times])
        ax.plot(log_n, log_t, f'{marker}-', color=color, linewidth=2,
                markersize=7, label=label)
        
        # Fit slope (scaling exponent)
        if len(log_n) > 2:
            slope, intercept = np.polyfit(log_n, log_t, 1)
            ax.plot(log_n, slope * log_n + intercept, '--', color=color, alpha=0.5)
            ax.text(log_n[-1] + 0.05, log_t[-1], f'O(n^{slope:.1f})',
                    fontsize=10, ha="right", color=color, fontweight='bold')
    
    ax.set_xlabel('log₁₀(n)', fontsize=10)
    ax.set_ylabel('log₁₀(time)', fontsize=10)
    ax.set_title('Scaling Exponent (log-log)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    
    fig.tight_layout()
    fp = os.path.join(output_dir, 'fig9_scalability.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved {os.path.basename(fp)}")
    
    # ── Summary ──
    print(f"\n  Largest graph computed: n={max(ns)}, "
          f"κ_CM = {timing['cm_total'][-1]:.1f}s, "
          f"κ_FR = {timing['fr_total'][-1]:.2f}s, "
          f"κ_OR = {timing['or_total'][-1]:.1f}s")
    
    return timing


# =============================================================================
# MAIN
# =============================================================================

def main():
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("  CAYLEY-MENGER CURVATURE — EXTENDED VALIDATION")
    print("  Three extensions for methodological robustness")
    print("=" * 70)
    
    # Part 1: Known curvature
    known_results = run_known_curvature_analysis(output_dir)
    
    # Part 2: Community detection
    community_scores = run_community_detection_analysis(output_dir)
    
    # Part 3: Scalability
    timing = run_scalability_analysis(output_dir)
    
    print("\n" + "=" * 70)
    print("  ALL EXTENDED ANALYSES COMPLETE")
    print(f"  Figures saved to {output_dir}/")
    print("=" * 70)
    
    return known_results, community_scores, timing


if __name__ == "__main__":
    known_results, community_scores, timing = main()
