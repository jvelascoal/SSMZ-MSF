"""
Optimization models for agricultural segmentation problems.
EXECUTION MODES:
1. Heuristic only: python script.py <data_file> <alpha> h1|h2 [draw] [output] [plot]
2. Lazy model: python script.py <data_file> <alpha> lazy [time] [draw] [output] [plot]
3. Flow model: python script.py <data_file> <alpha> flow [time] [draw] [output] [plot]
4. Hybrid Lazy: python script.py <data_file> <alpha> lazy h1|h2 [time] [draw] [output] [plot]
5. Hybrid Flow: python script.py <data_file> <alpha> flow h1|h2 [time] [draw] [output] [plot]
6. Graph mode: Add 'graph' as second argument: python script.py grid|graph <data_file> <alpha> ...

Note: 
- 'grid' mode uses the original grid-based incidence matrix (default)
- 'graph' mode uses a planar graph loaded from the data file
- The incidence matrix is auto-generated from the grid dimensions
- 'time' is the time limit in seconds for the optimization (default: 1800s)
- 'draw' (optional) - generates LaTeX visualizations
- 'output' (optional) - generates a detailed report file
- 'plot' (optional) - generates convergence plots from Gurobi logs
"""

import os
import networkx as nx
import random as rnd
import sys
import disjoin as dj
import numpy as np
import gurobipy as grb
import time
from copy import deepcopy
from datetime import datetime
import re
import json
import subprocess

# ============================================================================
# 0. OPTIONAL IMPORTS FOR PLOTTING
# ============================================================================

def check_and_import_grblogtools():
    """
    Check if grblogtools is installed.
    Returns True if available, False otherwise.
    """
    try:
        import grblogtools as glt
        return True
    except ImportError:
        return False


def check_and_import_matplotlib():
    """
    Check if matplotlib is installed.
    Returns True if available, False otherwise.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return True
    except ImportError:
        return False


def check_and_import_plotly():
    """
    Check if plotly is installed.
    Returns True if available, False otherwise.
    """
    try:
        import plotly.express as px
        return True
    except ImportError:
        return False


def generate_convergence_plot(logfile, output_filename):
    """
    Generates convergence plots from a Gurobi log file using grblogtools.
    Creates:
        - PDF (vector quality for publications)
        - PNG (for presentations/web)
        - HTML (interactive for exploration)
    
    Inputs:
        logfile (str): Path to the Gurobi log file
        output_filename (str): Base filename for output (without extension)
    
    Returns:
        bool: True if plots were generated successfully, False otherwise
    """
    
    if not check_and_import_grblogtools():
        print("Warning: grblogtools not installed. Please run:")
        print("  python -m pip install grblogtools")
        print("Skipping convergence plot generation.")
        return False
    
    try:
        import grblogtools as glt
        import pandas as pd
    except ImportError as e:
        print(f"Error importing required libraries: {e}")
        return False
    
    try:
        results = glt.parse(logfile)
        
        progress_data = results.progress("nodelog")
        
        if progress_data is None or progress_data.empty:
            print(f"Warning: No nodelog data found in {logfile}. Skipping plot.")
            return False
        
        import os
        log_basename = os.path.basename(logfile)
        
        # ====================================================================
        # 1. GENERATE PDF AND PNG USING MATPLOTLIB
        # ====================================================================
        if check_and_import_matplotlib():
            try:
                import matplotlib.pyplot as plt
                import matplotlib.ticker as ticker
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                
                has_incumbent = 'Incumbent' in progress_data.columns
                has_bestbd = 'BestBd' in progress_data.columns
                has_gap = 'Gap' in progress_data.columns
                
                if has_incumbent:
                    ax1.plot(progress_data['Time'], progress_data['Incumbent'], 
                            'b-', label='Incumbent (Best Solution)', linewidth=2)
                else:
                    ax1.text(0.5, 0.5, 'No Incumbent data available', 
                            transform=ax1.transAxes, ha='center', va='center')
                
                if has_bestbd:
                    ax1.plot(progress_data['Time'], progress_data['BestBd'], 
                            'r--', label='Best Bound', linewidth=2)
                
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Objective Value')
                ax1.set_title(f'Convergence Plot: {log_basename}')
                ax1.legend(loc='best')
                ax1.grid(True, alpha=0.3)
                

                ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: f'{int(x)}' if x < 1000 else f'{x/1000:.1f}k'))
                

                if has_gap:
                    ax2.plot(progress_data['Time'], progress_data['Gap'], 
                            'g-', label='MIP Gap (%)', linewidth=2)
                    ax2.set_ylabel('MIP Gap (%)')
                    ax2.legend(loc='best')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No MIP Gap data available', 
                            transform=ax2.transAxes, ha='center', va='center',
                            fontsize=12)
                    ax2.set_ylabel('MIP Gap (%)')
                
                ax2.set_xlabel('Time (seconds)')
                ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: f'{int(x)}' if x < 1000 else f'{x/1000:.1f}k'))
                

                try:
                    model_name = results.solver_info.get('ModelName', '')
                    if model_name:
                        plt.suptitle(f'Gurobi Convergence Analysis\nModel: {model_name}', fontsize=14)
                    else:
                        plt.suptitle(f'Gurobi Convergence Analysis\n{log_basename}', fontsize=14)
                except:
                    plt.suptitle(f'Gurobi Convergence Analysis\n{log_basename}', fontsize=14)
                
                plt.tight_layout()
                
                plt.savefig(output_filename + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
                
                plt.savefig(output_filename + '.png', dpi=300, bbox_inches='tight')
                
                plt.close()
                
                print(f"  PDF saved to: {output_filename}.pdf")
                print(f"  PNG saved to: {output_filename}.png")
                
            except Exception as e:
                print(f"  Warning: Matplotlib plot failed: {e}")
        else:
            print("  Warning: matplotlib not installed. Skipping PDF/PNG generation.")
            print("  Install with: python -m pip install matplotlib")
        
        # ====================================================================
        # 2. GENERATE INTERACTIVE HTML USING PLOTLY
        # ====================================================================
        if check_and_import_plotly():
            try:
                import plotly.express as px
                

                y_columns = []
                if 'Incumbent' in progress_data.columns:
                    y_columns.append('Incumbent')
                if 'BestBd' in progress_data.columns:
                    y_columns.append('BestBd')
                if 'Gap' in progress_data.columns:
                    y_columns.append('Gap')
                
                if y_columns:
                    fig = px.line(progress_data, 
                                  x="Time", 
                                  y=y_columns,
                                  title=f"Convergence Plot: {os.path.basename(logfile)}",
                                  labels={"value": "Value", "Time": "Time (seconds)", 
                                          "variable": "Metric"})
                    
                    fig.write_html(output_filename + '.html')
                    print(f"  Interactive HTML saved to: {output_filename}.html")
                else:
                    print("  Warning: No data columns available for interactive plot.")
                    
            except Exception as e:
                print(f"  Warning: Plotly HTML generation failed: {e}")
        else:
            print("  Note: plotly not installed. To generate interactive HTML:")
            print("    python -m pip install plotly")
        
        return True
        
    except Exception as e:
        print(f"Error generating convergence plots: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_convergence_plot_matplotlib_only(logfile, output_filename):
    """
    Generates convergence plots using matplotlib only (no grblogtools).
    Parses log file manually.
    
    Inputs:
        logfile (str): Path to the Gurobi log file
        output_filename (str): Base filename for output (without extension)
    
    Returns:
        bool: True if plots were generated successfully, False otherwise
    """
    if not check_and_import_matplotlib():
        print("Warning: matplotlib not installed. Please run:")
        print("  python -m pip install matplotlib")
        return False
    
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        times = []
        incumbents = []
        best_bounds = []
        gaps = []
        
        with open(logfile, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        if parts[0] == 'Nodes' or parts[0] == 'H' or parts[0] == '*':
                            continue
                        
                        numeric_parts = []
                        for p in parts:
                            try:
                                numeric_parts.append(float(p))
                            except ValueError:
                                pass
                        
                        if len(numeric_parts) >= 4:
                            time_val = numeric_parts[0]
                            incumbent = numeric_parts[2] if len(numeric_parts) > 2 else None
                            best_bound = numeric_parts[3] if len(numeric_parts) > 3 else None
                            gap = numeric_parts[4] if len(numeric_parts) > 4 else None
                        
                            gap_str = None
                            for p in parts:
                                if '%' in p:
                                    gap_str = p.replace('%', '')
                                    try:
                                        gap = float(gap_str)
                                    except:
                                        pass
                            
                            if incumbent is not None and best_bound is not None:
                                times.append(time_val)
                                incumbents.append(incumbent)
                                best_bounds.append(best_bound)
                                if gap is not None:
                                    gaps.append(gap)
                    except:
                        pass
        
        if not times:
            print(f"Warning: No convergence data found in {logfile}. Skipping plot.")
            return False
        

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        

        ax1.plot(times, incumbents, 'b-', label='Incumbent (Best Solution)', linewidth=2)
        ax1.plot(times, best_bounds, 'r--', label='Best Bound', linewidth=2)
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Objective Value')
        ax1.set_title(f'Convergence Plot: {os.path.basename(logfile)}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        

        if gaps:
            ax2.plot(times, gaps, 'g-', label='MIP Gap (%)', linewidth=2)
            ax2.set_ylabel('MIP Gap (%)')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No MIP Gap data available', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_ylabel('MIP Gap (%)')
        
        ax2.set_xlabel('Time (seconds)')
        
        plt.suptitle(f'Gurobi Convergence Analysis\n{os.path.basename(logfile)}', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(output_filename + '.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_filename + '.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  PDF saved to: {output_filename}.pdf")
        print(f"  PNG saved to: {output_filename}.png")
        return True
        
    except Exception as e:
        print(f"Error generating convergence plot: {e}")
        return False


def generate_convergence_plot_full(logfile, output_filename, method='auto'):
    """
    Generates convergence plots from a Gurobi log file.
    Tries grblogtools first, falls back to manual parsing if not available.
    
    Inputs:
        logfile (str): Path to the Gurobi log file
        output_filename (str): Base filename for output (without extension)
        method (str): 'auto', 'grblogtools', or 'matplotlib'
    
    Returns:
        bool: True if plots were generated successfully, False otherwise
    """

    if method == 'auto' or method == 'grblogtools':
        if check_and_import_grblogtools():
            success = generate_convergence_plot(logfile, output_filename)
            if success:
                return True
            else:
                print("  grblogtools method failed, trying matplotlib fallback...")
    

    if method == 'auto' or method == 'matplotlib':
        return generate_convergence_plot_matplotlib_only(logfile, output_filename)
    
    return False


# ============================================================================
# 1. GRAPH CONSTRUCTION
# ============================================================================

def create_graph(m, n, c):
    """
    Creates an undirected graph with vertices in an m x n grid.
    Only includes vertices with c > 0.
    Includes a sink node with ID = m*n (original total nodes).
    Used for the flow model which requires a sink for flow conservation.
    
    Inputs:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
        c (dict): Dictionary mapping (i,j) coordinates to data values
    
    Returns:
        G (networkx.Graph): Undirected graph with sink node (ID = m*n)
                            Edges have 'costo' attribute (absolute difference)
        node_map (dict): Dictionary mapping original index to new vertex ID
        reverse_map (dict): Dictionary mapping new vertex ID to original index
        valid_nodes (list): List of valid vertex indices
    """
    N = m * n
    G = nx.Graph()
    
  
    valid_nodes = []
    node_map = {}  
    reverse_map = {} 
    

    for i in range(N):
        if dj.get_datos(c, i, n) > 0:
            valid_nodes.append(i)
            node_map[i] = i  
            reverse_map[i] = i
    
    num_valid = len(valid_nodes)
    print(f"  Valid nodes: {num_valid} out of {N} (removed {N - num_valid} nodes with c <= 0)")
    
    for v in valid_nodes:
        G.add_node(v)
        G.nodes[v]['sample'] = dj.get_datos(c, v, n)
        G.nodes[v]['original_index'] = v
        G.nodes[v]['valid'] = True
    
    sink_id = N
    G.add_node(sink_id)
    G.nodes[sink_id]['sample'] = None
    G.nodes[sink_id]['original_index'] = -1
    G.nodes[sink_id]['valid'] = True
    

    for orig_i in valid_nodes:
        if orig_i + 1 in node_map:
            if (orig_i + 1) % n != 0:  
                j = node_map[orig_i + 1]
                i = orig_i
                G.add_edge(i, j)
                G[i][j]['costo'] = round(abs(
                    dj.get_datos(c, i, n) - 
                    dj.get_datos(c, j, n)
                ), 4)
        
        if orig_i + n in node_map:
            j = node_map[orig_i + n]
            i = orig_i
            G.add_edge(i, j)
            G[i][j]['costo'] = round(abs(
                dj.get_datos(c, i, n) - 
                dj.get_datos(c, j, n)
            ), 4)
    

    for k in valid_nodes:
        G.add_edge(k, sink_id)
        G[k][sink_id]['costo'] = 0  
    data=nx.node_link_data(G)
    with open("salida.json","w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)
    return G

def create_graph_noaux(m, n, c):
    """
    Creates graph without sink node.
    Only includes vertices with c > 0.
    Used for heuristics and the lazy model which don't require a sink.
    Edge costs are based on variance for better MST construction.
    
    Inputs:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
        c (dict): Dictionary mapping (i,j) coordinates to data values
    
    Returns:
        G (networkx.Graph): Undirected graph without sink node
                            Edges have 'costo' attribute (variance)
        node_map (dict): Dictionary mapping original index to new vertex ID
        reverse_map (dict): Dictionary mapping new vertex ID to original index
        valid_nodes (list): List of valid vertex indices
    """
    N = m * n
    G = nx.Graph()
    

    valid_nodes = []
    node_map = {}  
    reverse_map = {}  
    
 
    for i in range(N):
        if dj.get_datos(c, i, n) > 0:
            valid_nodes.append(i)
            node_map[i] = i
            reverse_map[i] = i
    
    num_valid = len(valid_nodes)
    print(f"  Valid nodes: {num_valid} out of {N} (removed {N - num_valid} nodes with c <= 0)")
    
    for v in valid_nodes:
        G.add_node(v)
        G.nodes[v]['sample'] = dj.get_datos(c, v, n)
        G.nodes[v]['original_index'] = v
        G.nodes[v]['valid'] = True
    
    for orig_i in valid_nodes:
        if orig_i + 1 in node_map:
            if (orig_i + 1) % n != 0: 
                j = node_map[orig_i + 1]
                i = orig_i
                G.add_edge(i, j)
                G[i][j]['costo'] = round(abs(dj.get_datos(c, i, n) - dj.get_datos(c, j, n) ), 4)
        
        if orig_i + n in node_map:
            j = node_map[orig_i + n]
            i = orig_i
            G.add_edge(i, j)
            G[i][j]['costo'] = round(abs(dj.get_datos(c, i, n) - dj.get_datos(c, j, n) ), 4)
    
    return G


# ============================================================================
# 1.1 GRAPH LOADING FOR PLANAR GRAPHS
# ============================================================================

def load_data_graph(data_file, model_type, add_sink=True):
    """
    Loads a planar graph from a data file.
    
    File format:
        <n> <m>              # Number of vertices, number of edges
        <vertex_id> <value>  # Vertex sample value (n lines)
        <u> <v>              # Edge (m lines)
    
    Inputs:
        data_file (str): Path to the data file
        model_type (str): 'lazy' or 'flow'
        add_sink (bool): Whether to add a sink node (for flow model)
    
    Returns:
        G (networkx.Graph): Undirected graph
        n (int): Number of vertices
        m (int): Number of edges
    """
    with open(data_file, "r") as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith('#')]
    
    if len(lines) < 2:
        raise ValueError(f"File {data_file} has insufficient data")
    

    parts = lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    
    G = nx.Graph()
    
    line_idx = 1
    for i in range(n):
        if line_idx >= len(lines):
            raise ValueError(f"Expected {n} vertices, got {line_idx - 1}")
        
        parts = lines[line_idx].split()
        if len(parts) < 2:
            raise ValueError(f"Invalid vertex line: {lines[line_idx]}")
        
        v_id = int(parts[0])
        value = float(parts[1])
        G.add_node(v_id, sample=value)
        line_idx += 1
    
    for i in range(m):
        if line_idx >= len(lines):
            raise ValueError(f"Expected {m} edges, got {i}")
        
        parts = lines[line_idx].split()
        if len(parts) < 2:
            raise ValueError(f"Invalid edge line: {lines[line_idx]}")
        
        u = int(parts[0])
        v = int(parts[1])
        
        cost = round(abs(G.nodes[u]['sample'] - G.nodes[v]['sample']), 4)
        G.add_edge(u, v, costo=cost)
        line_idx += 1
    
    if not nx.is_connected(G):
        print(f"  WARNING: Graph is not connected! ({nx.number_connected_components(G)} components)")
    
    if add_sink:
        sink_id = n
        G.add_node(sink_id, sample=0.0)
        for u in G.nodes():
            if u != sink_id:
                G.add_edge(u, sink_id, costo=0.0)
    
    print(f"  Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Connected: {nx.is_connected(G)}")
    
    return G, n, m


# ============================================================================
# 2. HEURISTICS
# ============================================================================

def heuristic_h1(G, VT, alpha):
    """
    HEURISTIC H1: Simple edge removal heuristic.
    Removes edges in descending order of cost until homogeneity constraint is satisfied.
    This is faster but may produce suboptimal results.
    
    Inputs:
        G (networkx.Graph): Input graph
        datos (dict): Data values for each vertex
        m (int): Number of rows
        n (int): Number of columns
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
    
    Returns:
        CC (list): List of components (list of lists of vertices)
        lista_aristas (dict): Dictionary mapping component index to list of edges
        iterations (int): Number of iterations performed
    """
    T = nx.minimum_spanning_tree(G, weight='costo', algorithm='prim')
    CC = list(nx.connected_components(T))
    CC = list(map(list, CC))
    st = dj.sol_graph(G,CC,'lazy')
    st.up_vars()
    H = st.calcH(VT)
    n_edges = len(T.edges())    
    iterations = 0
    while H < alpha and n_edges > 0:
        aristas = sorted(T.edges(data=True), key=lambda t: t[2].get('costo', 1), reverse=True)
        T.remove_edge(aristas[0][0], aristas[0][1])
        CC = list(nx.connected_components(T))
        CC = list(map(list, CC))
        st = dj.sol_graph(G,CC,'lazy')
        st.up_vars()
        H = st.calcH(VT)
        iterations += 1
        n_edges -= 1
    
    CC = sorted(CC, key=len, reverse=True)
    lista_aristas = aristas_component(T, CC)
    return CC, lista_aristas, iterations


def heuristic_h2(G, VT, alpha):
    """
    HEURISTIC H2: Evaluates all edges before removal.
    This heuristic computes homogeneity after temporarily removing each edge
    and selects the edge that yields the highest H value.
    
    Inputs:
        G (networkx.Graph): Input graph
        datos (dict): Data values for each vertex
        m (int): Number of rows
        n (int): Number of columns
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
    
    Returns:
        CC (list): List of components (list of lists of vertices)
        lista_aristas (dict): Dictionary mapping component index to list of edges
        iterations (int): Number of iterations performed
    """
    T = nx.minimum_spanning_tree(G, weight='costo', algorithm='prim')
    CC = list(nx.connected_components(T))
    CC = list(map(list, CC))
    st = dj.sol_graph(G,CC,'lazy')
    st.up_vars()
    H = st.calcH(VT)
    n_edges = len(T.edges())    
    iterations = 0
    while H < alpha and n_edges > 0:
        aristas = list(T.edges(data=True))
        Hlist = []
        Taux = deepcopy(T)
        for e in aristas:
            Taux.remove_edge(e[0], e[1])
            CC_aux = list(nx.connected_components(Taux))
            CC_aux = list(map(list, CC_aux))
            st_aux=dj.sol_graph(G,CC_aux,'lazy')
            st_aux.up_vars()
            Hlist.append([e[0], e[1], st_aux.calcH(VT)])
            Taux.add_edge(e[0], e[1], costo=e[2]['costo'])
        
        Hlist = sorted(Hlist, key=lambda t: t[2], reverse=True)
        T.remove_edge(Hlist[0][0], Hlist[0][1])
        CC = list(nx.connected_components(T))
        CC = list(map(list, CC))
        st = dj.sol_graph(G,CC,'lazy')
        st.up_vars()
        H = st.calcH(VT)
        iterations += 1
        n_edges -= 1
    CC = sorted(CC, key=len, reverse=True)
    lista_aristas = aristas_component(T, CC)
    return CC, lista_aristas, iterations


# ============================================================================
# 2.1 HEURISTICS FOR PLANAR GRAPHS
# ============================================================================

def heuristic_h1_graph(G, VT, alpha):
    """
    HEURISTIC H1 for planar graphs.
    Same logic as heuristic_h1 but works with any graph structure.
    
    Inputs:
        G (networkx.Graph): Input graph (planar or any connected graph)
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
    
    Returns:
        CC (list): List of components (list of lists of vertices)
        lista_aristas (dict): Dictionary mapping component index to list of edges
        iterations (int): Number of iterations performed
    """
    return heuristic_h1(G, VT, alpha)


def heuristic_h2_graph(G, VT, alpha):
    """
    HEURISTIC H2 for planar graphs.
    Same logic as heuristic_h2 but works with any graph structure.
    
    Inputs:
        G (networkx.Graph): Input graph (planar or any connected graph)
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
    
    Returns:
        CC (list): List of components (list of lists of vertices)
        lista_aristas (dict): Dictionary mapping component index to list of edges
        iterations (int): Number of iterations performed
    """
    return heuristic_h2(G, VT, alpha)


# ============================================================================
# 3. AUXILIARY FUNCTIONS
# ============================================================================

def conv_lista(lista):
    """
    Convert list of lists removing empty entries.
    
    Inputs:
        lista (list): List of lists
    
    Returns:
        lista_aux (list): Filtered list with empty lists removed
    """
    lista_aux = []
    for k in range(0, len(lista)):
        if len(lista[k]) > 0:
            lista_aux.append(lista[k])
    return lista_aux


def buscar_arista(lista, e):
    """
    Finds which component an edge belongs to.
    
    Inputs:
        lista (list): List of components (list of lists of vertices)
        e (tuple): Edge (u, v)
    
    Returns:
        int: Component index if both vertices are in same component with size>1, else -1
    """
    for i in range(0, len(lista)):
        if e[0] in lista[i] and e[1] in lista[i]:
            return i if len(lista[i]) > 1 else -1
    return -1


def buscar(lista, elem):
    """
    Checks if element is in list.
    
    Inputs:
        lista (list): List to search in
        elem: Element to search for
    
    Returns:
        int: 1 if element found, 0 otherwise
    """
    return 1 if elem in lista else 0


def aristas_component(G, lista):
    """
    Assigns edges to components and returns dictionary with same indices as lista.
    Uses vertex-to-component mapping for O(1) lookup.
    
    Inputs:
        G (networkx.Graph): Graph containing edges
        lista (list): List of components (list of lists of vertices)
    
    Returns:
        lista_aristas (dict): Dictionary mapping component index to list of edges within that component
    """
    vertex_to_comp = {}
    for idx, comp in enumerate(lista):
        for v in comp:
            vertex_to_comp[v] = idx
    
    lista_aristas = {i: [] for i in range(len(lista))}
    
    for e in G.edges():
        u, v = e
        if u in vertex_to_comp and v in vertex_to_comp:
            comp_idx = vertex_to_comp[u]
            if vertex_to_comp[v] == comp_idx and len(lista[comp_idx]) > 1:
                lista_aristas[comp_idx].append(e)
    
    return lista_aristas


# ============================================================================
# 4. CALLBACK FOR LAZY MODEL
# ============================================================================

def lazy_callback(model, where):
    """
    Callback to add lazy constraints that eliminate cycles.
    This function is called by Gurobi at integer feasible solutions.
    
    Inputs:
        model (gurobipy.Model): The Gurobi model
        where (int): Callback context indicator
    
    Returns:
        None (adds lazy constraints to the model)
    """
    if where == grb.GRB.Callback.MIPSOL:
        y_val = model.cbGetSolution(model._y)
        arboles={}
        
        for u,v,k in y_val:
            if y_val[u,v,k]>0.001:
                try:
                    arboles[k].append([u,v])
                except KeyError:
                    arboles[k]=[]
                    arboles[k].append([u,v])
        for q in arboles:
            auxG=nx.Graph()
            aristaso=[]
            for e in arboles[q]:
                auxG.add_edge(e[0],e[1])
                aristaso.append([e[0],e[1]])
            ciclos=list(nx.simple_cycles(auxG))
            
            if len(ciclos)>0:
                for ciclo in ciclos:
                    res=0
                    cicloa=[]
                    if len(ciclo)>0:
                        for j in range(0,len(ciclo)-1):
                            cicloa.append([ciclo[j],ciclo[j+1]])
                        cicloa.append([ciclo[len(ciclo)-1],ciclo[0]])	
                    for e in cicloa:
                        if [e[0],e[1]] in aristaso:
                            res=res+model._y[e[0],e[1],q]
                        elif [e[1],e[0]] in aristaso:
                            res=res+model._y[e[1],e[0],q]
                    model.cbLazy(res<=len(cicloa)-1)
                    print(cicloa)


# ============================================================================
# 5. WARM START FUNCTIONS
# ============================================================================

def warm_start_trivial_grid(G, x, y, r, z, theta, K, model_type="flow"):
    """
    Trivial warm start: each vertex has its own color.
    Used when no heuristic warm start is specified.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        K (int): Number of possible colors (m*n)
        model_type (str): 'lazy' or 'flow'
    
    Returns:
        None (sets start values for variables)
    """
    print(f"  Applying TRIVIAL warm start (each vertex own color)")
    
    if model_type == "flow":
        n_colors = max(G.nodes())
        K=K-1
    else:
        n_colors = G.number_of_nodes()
        
    theta.start = K
    for k in range(0,K):
        r[k].start = 0.0
        z[k].start = 0.0
        for u in G.nodes():
            x[u, k].start = 0.0
        
    k=0
    nodos=list(G.nodes())
    while k < K:
        r[k].start = 1.0
        z[k].start = 1.0
        u=nodos[k]
        
        if model_type == "flow":
            if u != n_colors:
                x[u, k].start = 1.0
        else:
            x[u, k].start = 1.0
        k += 1
    if model_type == "flow":
        for k in range(0,K):
            x[n_colors, k].start = 1.0
    
    print(f"  Trivial warm start: {K} colors (each vertex independent)")

def warm_start_trivial_graph(G, x, y, r, z, theta, K, model_type="flow"):
    """
    Trivial warm start: each vertex has its own color.
    Used when no heuristic warm start is specified.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        K (int): Number of possible colors (m*n)
        model_type (str): 'lazy' or 'flow'
    
    Returns:
        None (sets start values for variables)
    """
    print(f"  Applying TRIVIAL warm start (each vertex own color)")
    
    if model_type == "flow":
        n_colors = max(G.nodes())
    else:
        n_colors = G.number_of_nodes()
        
    theta.start = K
    for k in range(0,K):
        r[k].start = 0.0
        z[k].start = 0.0
        for u in G.nodes():
            x[u, k].start = 0.0

    k=0
    nodos=list(G.nodes())

    while k < K:
        r[k].start = 1.0
        z[k].start = 1.0
        u=nodos[k]
        
        if model_type == "flow":
            if u != n_colors:
                print(u,k)
                x[u, k].start = 1.0
        else:
            x[u, k].start = 1.0
        k += 1
    if model_type == "flow":
        for k in range(0,K):
            x[n_colors, k].start = 1.0
    
    print(f"  Trivial warm start: {K} colors (each vertex independent)")

def warm_start_trivial(G, x, y, r, z, theta, K, model_type="flow",Itype='grid'):
    if Itype == 'grid':
        warm_start_trivial_grid(G, x, y, r, z, theta, K, model_type)
    else:
        warm_start_trivial_graph(G, x, y, r, z, theta, K, model_type)
    
def warm_start_heuristic(G, x, y, r, z, theta, vertices, aristas, K, model_type, model, VT):
    """
    Applies warm start from heuristic solution.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        vertices (list): List of components (list of lists of vertices)
        aristas (dict): Dictionary mapping component index to edges
        K (int): Number of possible colors
        model_type (str): 'lazy' or 'flow'
        model (gurobipy.Model): Gurobi model
        VT (float): Total variance normalization factor
    
    Returns:
        None (sets start values for variables)
    """
    print(f"\n  Applying warm start with {len(vertices)} components")
    print(f"  Available colors K = {K}")
    graph_sol=dj.sol_graph(G,vertices,model_type)
    graph_sol.up_vars()
    H=graph_sol.calcH(VT)
    fobj=graph_sol.fobj(VT, 0.5)
    if model_type == "flow":
        N = max(G.nodes())
    else:
        N = G.number_of_nodes()
    
    for k in range(0, K):
        r[k].start = 0.0
        z[k].start = 0.0
        
        for u in G.nodes():
            x[u, k].start = 0.0
        
        if model_type == "lazy":
            for e in G.edges():
                y[e[0], e[1], k].start = 0.0
    
    n_components = len(vertices)
    n_colors_to_use = min(n_components, K)
    
    print(f"  Activating {n_colors_to_use} colors")
    
    assigned_x = 0
    assigned_y = 0
   
    for k in range(0, n_components):
        r[k].start = 1.0
        z[k].start = len(vertices[k])
        
  
        for u in vertices[k]:
            x[u, k].start = 1.0
            assigned_x += 1
        if model_type=='flow':
            x[N, k].start = 1.0
            assigned_x += 1
        

        G=nx.Graph()
        if model_type == "lazy":
            for e in aristas[k]:
                y[e[0], e[1], k].start = 1.0
    theta.start = n_components    
    model.update()

    
    print(f"  Warm start complete:")
    print(f"    - {n_colors_to_use} colors activated")
    print(f"    - {assigned_x} x variables assigned")
    print(f"    - {assigned_y} y variables assigned")
    print(f"    - theta.start = {theta.start}")
    
            
def verify_warm_start_feasibility(G,model, x, y, r, z, theta, vertices, aristas, K, model_type):
    """
    Verifies if the warm start solution is feasible.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        model (gurobipy.Model): Gurobi model
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        vertices (list): List of components
        aristas (dict): Dictionary mapping component index to edges
        K (int): Number of possible colors
        model_type (str): 'lazy' or 'flow'
    
    Returns:
        None (prints verification results)
    """
    print("\n  === CHECKING WARM START FEASIBILITY ===")
    
    violations = 0
    for u in G.nodes(): 
        if u != K:
            sum_x = 0
            for k in range(K):
                if (u, k) in x and x[u, k].start == 1.0:
                    sum_x += 1
            if sum_x != 1:
                print(f"  VIOLATION: Vertex {u} has {sum_x} colors (should be 1)")
                violations += 1
                if violations > 5:
                    break
    
    if violations == 0:
        print("  ✓ Each vertex has exactly 1 color")
    
    violations = 0
    for k in range(K):
        if k in z:
            sum_vertices = sum(1 for u in range(K) if u != K and (u, k) in x and x[u, k].start == 1.0)
            z_val = z[k].start if z[k].start is not None else 0
            if abs(z_val - sum_vertices) > 0.001:
                print(f"  VIOLATION: Color {k}: z={z_val}, actual sum={sum_vertices}")
                violations += 1
                if violations > 5:
                    break
    
    if violations == 0:
        print("  ✓ z variables are consistent")
    
    violations = 0
    for k in range(K):
        if k in r and k in z:
            r_val = r[k].start if r[k].start is not None else 0
            z_val = z[k].start if z[k].start is not None else 0
            if (r_val == 1 and z_val <= 0) or (r_val == 0 and z_val > 0):
                print(f"  VIOLATION: Color {k}: r={r_val}, z={z_val}")
                violations += 1
                if violations > 5:
                    break
    
    if violations == 0:
        print("  ✓ r variables are consistent")
    
    if model_type == "lazy":
        violations = 0
        for (i, j) in G.edges():
            if i != K and j != K:
                for k in range(K):
                    if (i, j, k) in y and y[i, j, k].start == 1.0:
                        if (i, k) not in x or x[i, k].start != 1.0:
                            print(f"  VIOLATION: Edge ({i},{j}) color {k} but vertex {i} doesn't have color {k}")
                            violations += 1
                        if (j, k) not in x or x[j, k].start != 1.0:
                            print(f"  VIOLATION: Edge ({i},{j}) color {k} but vertex {j} doesn't have color {k}")
                            violations += 1
                        if violations > 5:
                            break
                if violations > 5:
                    break
        
        if violations == 0:
            print("  ✓ Edge variables are consistent")
    

    if theta.start is not None:
        sum_r = sum(1 for k in range(K) if k in r and r[k].start == 1.0)
        if abs(theta.start - sum_r) > 0.001:
            print(f"  VIOLATION: theta={theta.start}, actual sum_r={sum_r}")
        else:
            print(f"  ✓ theta is consistent (={theta.start})")
    
    print("  === END FEASIBILITY CHECK ===\n")




# ============================================================================
# 6. COLOR GENERATION AND DRAWING
# ============================================================================

def genera_colores2(m, n):
    """
    Generates complementary color pairs for visual distinction.
    
    Inputs:
        m (int): Number of rows
        n (int): Number of columns
    
    Returns:
        colores (list): List of RGB color values as [r,g,b] lists
    """
    N = m * n
    rnd.seed(45679)
    colores = []
    for k in range(0, int(N / 2) + 1):
        p1 = rnd.randint(0, 1)
        r_val = rnd.uniform(0.0, 0.3) if p1 == 0 else rnd.uniform(0.7, 1.0)
        
        p2 = rnd.randint(0, 1)
        g_val = rnd.uniform(0.0, 0.3) if p2 == 0 else rnd.uniform(0.7, 1.0)
        
        b_val = rnd.uniform(0, 1.0)
        colores.append([r_val, g_val, b_val])
        colores.append([1 - r_val, 1 - g_val, 1 - b_val])
    return colores


def genera_colores_graph(n):
    """
    Generates colors for graph visualization.
    
    Inputs:
        n (int): Number of colors needed
    
    Returns:
        colores (list): List of RGB color values as [r,g,b] lists
    """
    rnd.seed(45679)
    colores = []
    n_colors = max(n, 1)
    for k in range(0, n_colors + 1):
        r_val = rnd.uniform(0.2, 0.9)
        g_val = rnd.uniform(0.2, 0.9)
        b_val = rnd.uniform(0.2, 0.9)
        colores.append([r_val, g_val, b_val])
    return colores


def dibujo(nombre, m, n, c_nodos, seed=456):
    """
    Generates a LaTeX diagram of colored nodes (rectangles only).
    
    Inputs:
        nombre (str): Base filename for output
        m (int): Number of rows
        n (int): Number of columns
        c_nodos (dict or list): Components (dictionary or list of lists)
        seed (int): Random seed for colors
    
    Returns:
        None (creates .tex and .pdf files)
    """
    N = m * n
    rnd.seed(seed)
    rgbt = genera_colores2(m, n)
    
    if isinstance(c_nodos, dict):
        nodos_dict = {k: v for k, v in c_nodos.items() if len(v) > 0}
    else:
        nodos_dict = {k: v for k, v in enumerate(c_nodos) if len(v) > 0}
    
    with open(nombre + ".tex", 'w') as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage[usenames,dvipsnames]{xcolor}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\usetikzlibrary{patterns}\n")
        
        for k in range(0, len(rgbt)):
            f.write(f"\\definecolor{{col{str(k)}}}{{rgb}}{{{rgbt[k][0]},{rgbt[k][1]},{rgbt[k][2]}}}\n")
        
        f.write("\\begin{document}\n")
        f.write("\\begin{tikzpicture}[scale=1.5]\n")
        
        for k, nodos in nodos_dict.items():
            for p in nodos:
                if p != N and p is not None:
                    i = int(p / n)
                    j = p % n
                    f.write(f"\\fill[fill=col{str(k)},opacity=0.45]({j},{m-(i+1)}) rectangle ({j+1},{m-i});\n")
        
        for k, nodos in nodos_dict.items():
            for p in nodos:
                if p != N and p is not None:
                    i = int(p / n)
                    j = p % n
                    f.write(f"\\node[text=black, font=\\fontsize{{6}}{{6}},opacity=1,text opacity=1,circle,minimum size=0.8cm](v{p}) at ({j+0.5},{m-i-0.5}){{${k+1}$}};\n")
        
        f.write("\\end{tikzpicture}\n")
        f.write("\\end{document}\n")
    
    _outdir = os.path.dirname(nombre)
    _odflag = f"-output-directory={_outdir}" if _outdir else ""
    os.system(f"pdflatex --interaction=batchmode {_odflag} {nombre}.tex > /dev/null 2>&1")
    for ext in ['.aux', '.log']:
        aux_file = nombre + ext
        if os.path.exists(aux_file):
            os.remove(aux_file)


def dibujo2(nombre, G, m, n, c_nodos, c_arcos, seed=456):
    """
    Draws complete graph with colored nodes and edges.
    
    Inputs:
        nombre (str): Base filename for output
        G (networkx.DiGraph): Directed graph
        m (int): Number of rows
        n (int): Number of columns
        c_nodos (dict): Dictionary mapping component index to list of vertices
        c_arcos (dict): Dictionary mapping component index to list of edges
        seed (int): Random seed for colors
    
    Returns:
        None (creates .tex and .pdf files)
    """
    N = m * n
    rnd.seed(seed)
    
    if isinstance(c_nodos, dict):
        v_list = {k: v for k, v in c_nodos.items() if len(v) > 0}
    else:
        v_list = {k: v for k, v in enumerate(c_nodos) if len(v) > 0}
    
    if isinstance(c_arcos, dict):
        e_list = {k: v for k, v in c_arcos.items() if len(v) > 0}
    else:
        e_list = {k: v for k, v in enumerate(c_arcos) if len(v) > 0}
    
    rgbt = genera_colores2(m, n)
    
    with open(nombre + ".tex", 'w') as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage[usenames,dvipsnames]{xcolor}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\usetikzlibrary{patterns}\n")
        
        for k in range(0, len(rgbt)):
            f.write(f"\\definecolor{{col{str(k)}}}{{rgb}}{{{rgbt[k][0]},{rgbt[k][1]},{rgbt[k][2]}}}\n")
        
        f.write("\\begin{document}\n")
        f.write("\\begin{tikzpicture}[scale=1.5]\n")
        
        all_nodes = set()
        for nodos in v_list.values():
            for p in nodos:
                if p != N and p is not None:
                    all_nodes.add(p)
        
        for p in all_nodes:
            i = int(p / n)
            j = p % n
            f.write(f"\\coordinate (v{p}) at ({j+0.5},{m-i-0.5});\n")
        
        for k, nodos in v_list.items():
            color_idx = k % len(rgbt)
            for p in nodos:
                if p != N and p is not None:
                    i = int(p / n)
                    j = p % n
                    f.write(f"\\fill[fill=col{str(color_idx)},opacity=0.35]({j},{m-(i+1)}) rectangle ({j+1},{m-i});\n")
        
        for k, aristas in e_list.items():
            color_idx = k % len(rgbt)
            for e in aristas:
                if e[0] != N and e[1] != N:
                    f.write(f"\\draw[color=col{str(color_idx)},very thick,opacity=0.7] (v{e[0]}) -- (v{e[1]});\n")
        
        for k, nodos in v_list.items():
            color_idx = k % len(rgbt)
            for p in nodos:
                if p != N and p is not None:
                    i = int(p / n)
                    j = p % n
                    f.write(f"\\node[fill=white, text=black, font=\\fontsize{{8}}{{8}},opacity=1.0,text opacity=1,circle,minimum size=0.6cm,draw=col{str(color_idx)},thick] at (v{p}) {{${k+1}$}};\n")
        
        f.write("\\end{tikzpicture}\n")
        f.write("\\end{document}\n")
    
    _outdir = os.path.dirname(nombre)
    _odflag = f"-output-directory={_outdir}" if _outdir else ""
    os.system(f"pdflatex --interaction=batchmode {_odflag} {nombre}.tex > /dev/null 2>&1")
    for ext in ['.aux', '.log']:
        aux_file = nombre + ext
        if os.path.exists(aux_file):
            os.remove(aux_file)


def dibujo_heuristic(nombre, G, m, n, c_nodos, c_arcos, seed=456):
    """
    Special version for drawing heuristic solutions.
    
    Inputs:
        nombre (str): Base filename for output
        G (networkx.DiGraph): Directed graph
        m (int): Number of rows
        n (int): Number of columns
        c_nodos (dict): Dictionary mapping component index to list of vertices
        c_arcos (dict): Dictionary mapping component index to list of edges
        seed (int): Random seed for colors
    
    Returns:
        None (creates .tex and .pdf files)
    """
    N = m * n
    rnd.seed(seed)
    
    if isinstance(c_nodos, dict):
        v_list = {k: v for k, v in c_nodos.items() if len(v) > 0}
    else:
        v_list = {k: v for k, v in enumerate(c_nodos) if len(v) > 0}
    
    if isinstance(c_arcos, dict):
        e_list = {k: v for k, v in c_arcos.items() if len(v) > 0}
    else:
        e_list = {k: v for k, v in enumerate(c_arcos) if len(v) > 0}
    
    rgbt = genera_colores2(m, n)
    
    with open(nombre + ".tex", 'w') as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage[usenames,dvipsnames]{xcolor}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\usetikzlibrary{patterns}\n")
        
        for k in range(0, len(rgbt)):
            f.write(f"\\definecolor{{col{str(k)}}}{{rgb}}{{{rgbt[k][0]},{rgbt[k][1]},{rgbt[k][2]}}}\n")
        
        f.write("\\begin{document}\n")
        f.write("\\begin{tikzpicture}[scale=1.5]\n")
        
        all_nodes = set()
        for nodos in v_list.values():
            for p in nodos:
                if p != N:
                    all_nodes.add(p)
        
        for p in all_nodes:
            i = int(p / n)
            j = p % n
            f.write(f"\\coordinate (v{p}) at ({j+0.5},{m-i-0.5});\n")
        
        for k, nodos in v_list.items():
            color_idx = k % len(rgbt)
            for p in nodos:
                if p != N:
                    i = int(p / n)
                    j = p % n
                    f.write(f"\\fill[fill=col{str(color_idx)},opacity=0.45]({j},{m-(i+1)}) rectangle ({j+1},{m-i});\n")
        

        for k, aristas in e_list.items():
            color_idx = k % len(rgbt)
            for e in aristas:
                if e[0] != N and e[1] != N:
                    f.write(f"\\draw[color=col{str(color_idx)},line width=1.5pt,opacity=0.9] (v{e[0]}) -- (v{e[1]});\n")
        
        for k, nodos in v_list.items():
            color_idx = k % len(rgbt)
            for p in nodos:
                if p != N:
                    f.write(f"\\node[fill=white, text=black, font=\\fontsize{{8}}{{8}},circle,minimum size=0.6cm,draw=col{str(color_idx)},thick] at (v{p}) {{${k+1}$}};\n")
        
        f.write("\\end{tikzpicture}\n")
        f.write("\\end{document}\n")
    
    _outdir = os.path.dirname(nombre)
    _odflag = f"-output-directory={_outdir}" if _outdir else ""
    os.system(f"pdflatex --interaction=batchmode {_odflag} {nombre}.tex > /dev/null 2>&1")
    for ext in ['.aux', '.log']:
        aux_file = nombre + ext
        if os.path.exists(aux_file):
            os.remove(aux_file)


# ============================================================================
# 6.1 DRAWING FUNCTIONS FOR PLANAR GRAPHS
# ============================================================================
def dibujo_graph_tikz(nombre, G, c_nodos, c_arcos, seed=456):
    """
    Draws a planar graph with colored nodes and edges using TikZ.
    Uses planar layout from networkx for positioning (no edge crossings).
    
    Inputs:
        nombre (str): Base filename for output
        G (networkx.Graph): Input graph (undirected)
        c_nodos (dict or list): Components (dictionary mapping component index to vertices,
                                or list of lists of vertices)
        c_arcos (dict or list): Edges per component (dictionary mapping component index to edges,
                                or list of lists of edges)
        seed (int): Random seed for colors
    
    Returns:
        None (creates .tex and .pdf files)
    """
    try:
        if nx.check_planarity(G)[0]:
            pos = nx.planar_layout(G, scale=2.0)
        else:
            print("  Warning: Graph is not planar, using spring layout")
            pos = nx.spring_layout(G, seed=seed, k=2.0, iterations=150)
    except Exception as e:
        print(f"  Warning: Planar layout failed ({e}), using spring layout")
        pos = nx.spring_layout(G, seed=seed, k=2.0, iterations=150)
    
    if isinstance(c_nodos, list):
        v_list = {k: v for k, v in enumerate(c_nodos) if len(v) > 0}
    else:
        v_list = {k: v for k, v in c_nodos.items() if len(v) > 0}
    
    if isinstance(c_arcos, list):
        e_list = {k: v for k, v in enumerate(c_arcos) if len(v) > 0}
    else:
        e_list = {k: v for k, v in c_arcos.items() if len(v) > 0}

    n_components = len(v_list)
    colores = genera_colores_graph(max(n_components, 1))
    
    with open(nombre + ".tex", 'w') as f:
        f.write("\\documentclass{standalone}\n")
        f.write("\\usepackage[usenames,dvipsnames]{xcolor}\n")
        f.write("\\usepackage{tikz}\n")
        f.write("\\usetikzlibrary{patterns,positioning,shapes.geometric,backgrounds}\n")
        

        for k in range(len(colores)):
            r, g, b = colores[k]
            f.write(f"\\definecolor{{col{str(k)}}}{{rgb}}{{{r},{g},{b}}}\n")
        
        f.write("\\begin{document}\n")
        f.write("\\begin{tikzpicture}[scale=1.8, every node/.style={draw,circle,minimum size=0.7cm,font=\\small,inner sep=1pt}]\n")

        if pos:
            all_x = [pos[v][0] for v in pos if v in pos]
            all_y = [pos[v][1] for v in pos if v in pos]
            
            if all_x and all_y:
                min_x, max_x = min(all_x), max(all_x)
                min_y, max_y = min(all_y), max(all_y)
                range_x = max_x - min_x if max_x - min_x > 0 else 1
                range_y = max_y - min_y if max_y - min_y > 0 else 1
                

                scale = 10
                padding = 0.5
                for v in G.nodes():
                    if v in pos:
                        x = pos[v][0]
                        y = pos[v][1]
                        norm_x = ((x - min_x) / range_x) * (scale - 2 * padding) + padding
                        norm_y = ((y - min_y) / range_y) * (scale - 2 * padding) + padding
                        f.write(f"\\coordinate (v{v}) at ({norm_x},{norm_y});\n")
        

        f.write("\\begin{scope}[on background layer]\n")
        for k, aristas in e_list.items():
            color_idx = k % len(colores)
            for e in aristas:
                u, v = e
                if u in pos and v in pos:
                    f.write(f"\\draw[color=col{str(color_idx)},line width=1.5pt,opacity=0.7] (v{u}) -- (v{v});\n")
        f.write("\\end{scope}\n")
        

        for k, nodos in v_list.items():
            color_idx = k % len(colores)
            for v in nodos:
                if v in pos:
                    sample = G.nodes[v].get('sample', 0)
                    f.write(f"\\node[fill=col{str(color_idx)}!25, draw=col{str(color_idx)}!80, thick, text=black] at (v{v}) {{{v}:{k+1}}};\n")
        
        f.write("\\end{tikzpicture}\n")
        f.write("\\end{document}\n")
    
    _outdir = os.path.dirname(nombre)
    _odflag = f"-output-directory={_outdir}" if _outdir else ""
    os.system(f"pdflatex --interaction=batchmode {_odflag} {nombre}.tex > /dev/null 2>&1")
    for ext in ['.aux', '.log']:
        aux_file = nombre + ext
        if os.path.exists(aux_file):
            os.remove(aux_file)



def dibujo_graph_graphviz(nombre, G, c_nodos, c_arcos, seed=456):
    """
    Draws a planar graph using Graphviz (if available).
    This produces cleaner layouts for planar graphs.
    
    Inputs:
        nombre (str): Base filename for output
        G (networkx.Graph): Input graph (undirected)
        c_nodos (dict or list): Components (dictionary or list of lists)
        c_arcos (dict or list): Edges per component (dictionary or list of lists)
        seed (int): Random seed for colors
    
    Returns:
        None (creates .pdf file)
    """
    try:
        import pygraphviz as pgv
        has_graphviz = True
    except ImportError:
        has_graphviz = False
    
    if not has_graphviz:
        print("  Graphviz not available. Using TikZ fallback.")
        dibujo_graph_tikz(nombre, G, c_nodos, c_arcos, seed)
        return
    
    try:
        if isinstance(c_nodos, list):
            v_list = {k: v for k, v in enumerate(c_nodos) if len(v) > 0}
        else:
            v_list = {k: v for k, v in c_nodos.items() if len(v) > 0}
        
        if isinstance(c_arcos, list):
            e_list = {k: v for k, v in enumerate(c_arcos) if len(v) > 0}
        else:
            e_list = {k: v for k, v in c_arcos.items() if len(v) > 0}
        
        A = nx.nx_agraph.to_agraph(G)
        
        A.graph_attr.update(
            splines="true",
            overlap="false",
            sep="+10",
            nodesep="0.5",
            ranksep="0.5"
        )
        
        n_components = len(v_list)
        colores = genera_colores_graph(n_components + 1)
        
  
        color_map = {}
        for k, nodos in v_list.items():
            if len(nodos) > 0:
                color_idx = k % len(colores)
                r, g, b = colores[color_idx]
                color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                for v in nodos:
                    color_map[v] = color

        for node in A.nodes():
            v_id = int(node)
            if v_id in color_map:
                node.attr['fillcolor'] = color_map[v_id]
                node.attr['style'] = 'filled'
                node.attr['fontcolor'] = 'black'

        for k, aristas in e_list.items():
            if len(aristas) > 0:
                color_idx = k % len(colores)
                r, g, b = colores[color_idx]
                edge_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
                for u, v in aristas:
                    edge = A.get_edge(u, v)
                    if edge:
                        edge.attr['color'] = edge_color
                        edge.attr['penwidth'] = '2'
        

        A.write(nombre + '.gv')
        A.draw(nombre + '_gv.pdf', prog='neato', format='pdf')
        print(f"  Graphviz PDF saved to: {nombre}.pdf")
        
    except Exception as e:
        print(f"  Graphviz drawing failed: {e}")
        print("  Using TikZ fallback.")
        dibujo_graph_tikz(nombre, G, c_nodos, c_arcos, seed)


def infer_grid_from_solution(c_nodos, c_arcos):
    """
    Determines whether a graph-mode solution can be embedded in a rectangular
    grid, so that it can be rendered with the grid-style cell maps instead of a
    node-link layout.

    The node identifiers are interpreted as row-major grid indices: a vertex with
    id p sits at row p // ncol and column p % ncol. The number of columns ncol is
    inferred from the spanning-forest edges (vertical adjacencies have a step of
    ncol, horizontal adjacencies a step of 1). The embedding is accepted only if
    every drawn edge is a valid grid adjacency. Missing ids are left as no-data
    cells, which naturally reproduces irregular field shapes.

    Inputs:
        c_nodos (dict or list): Components (zone -> list of vertices).
        c_arcos (dict or list): Edges per component (zone -> list of edges).

    Returns:
        tuple (nrow, ncol) if the solution is grid-embeddable, else None.
    """
    groups = c_nodos.values() if isinstance(c_nodos, dict) else c_nodos
    nodes = [p for vs in groups for p in vs if p is not None]
    if not nodes or not all(isinstance(p, int) for p in nodes):
        return None

    egroups = c_arcos.values() if isinstance(c_arcos, dict) else c_arcos
    edges = [(e[0], e[1]) for es in egroups for e in es]

    diffs = {abs(u - v) for u, v in edges if isinstance(u, int) and isinstance(v, int)}
    vertical = [d for d in diffs if d > 1]
    if not vertical:
        return None
    ncol = min(vertical)
    if ncol < 2:
        return None

    for u, v in edges:
        d = abs(u - v)
        horizontal = (d == 1 and u // ncol == v // ncol)
        if not (horizontal or d == ncol):
            return None

    nrow = max(nodes) // ncol + 1
    return nrow, ncol


def dibujo_graph(nombre, G, c_nodos, c_arcos, method='tikz', seed=456):
    """
    Main drawing function for planar graphs.

    If the solution is embeddable in a rectangular grid (node ids are row-major
    grid indices and every edge is a grid adjacency), it is rendered with the
    grid-style colored-cell map (reusing dibujo2), matching the grid-mode output
    and leaving missing cells blank for irregular fields. Otherwise it falls back
    to the requested node-link layout (TikZ or Graphviz).

    Inputs:
        nombre (str): Base filename for output
        G (networkx.Graph): Input graph (undirected)
        c_nodos (dict or list): Components (dictionary mapping component index to list of vertices,
                               or list of lists of vertices)
        c_arcos (dict or list): Edges per component (dictionary mapping component index to list of edges,
                               or list of lists of edges)
        method (str): 'tikz' or 'graphviz'
        seed (int): Random seed for colors

    Returns:
        None (creates visualization files)
    """
    dims = infer_grid_from_solution(c_nodos, c_arcos)
    if dims is not None:
        nrow, ncol = dims
        dibujo2(nombre, G, nrow, ncol, c_nodos, c_arcos, seed)
        return

    if method == 'graphviz':
        dibujo_graph_graphviz(nombre, G, c_nodos, c_arcos, seed)
    else:
        dibujo_graph_tikz(nombre, G, c_nodos, c_arcos, seed)


# ============================================================================
# 7. MODEL CREATION
# ============================================================================

def create_model_lazy(G, c, m, n, VT, alpha):
    """
    Creates the lazy model that uses lazy constraints to eliminate cycles.
    This model uses callback functions to add constraints on-the-fly.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        c (dict): Data values for each vertex
        m (int): Number of rows
        n (int): Number of columns
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
    
    Returns:
        mod (gurobipy.Model): Gurobi model
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        media (dict): Mean value per color
        p (dict): Absolute deviation variables
        dif_abs (dict): Absolute difference auxiliary variables
        aux (dict): Auxiliary variables for linearization
        H_num (gurobipy.Var): Numerator of homogeneity constraint
        H_den (gurobipy.Var): Denominator of homogeneity constraint
    """
    K = m * n
    N=G.number_of_nodes()
    x = {}
    y = {}
    z = {}
    r = {}
    media = {}
    p = {}
    dif_abs = {}
    aux = {}
    
    mod = grb.Model("msf_ssmz_lazy")
    
    for u in G.nodes():
        for k in range(0, K):
            x[u, k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"x_{u}_{k}")
            p[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"p_{u}_{k}")
            dif_abs[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"dabs_{u}_{k}")
            aux[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f"aux_{u}_{k}")
    
    for e in G.edges():
        for k in range(0, K):
            y[e[0], e[1], k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"y_{e[0]}_{e[1]}_{k}")
    
    for k in range(0, K):
        r[k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"r_{k}")
        z[k] = mod.addVar(vtype=grb.GRB.INTEGER, name=f"z_{k}")
        media[k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, name=f"media_{k}")
    
    theta = mod.addVar(vtype=grb.GRB.INTEGER, name="theta")
    H_num = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hnum")
    H_den = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hden")
    
    mod.setObjective(theta, grb.GRB.MINIMIZE)
    
    for k in range(0, K):
        mod.addConstr(grb.quicksum([x[u, k] for u in G.nodes()]) == z[k], f"num_vertices_color_{k}")
    
    for u in G.nodes():
        for k in range(0, K):
            mod.addConstr(x[u, k] <= r[k], f"uso_color_{k}_vertice_{u}")
    
    for e in G.edges():
        for k in range(0, K):
            mod.addConstr(y[e[0], e[1], k] <= r[k], f"uso_color_{k}_arista_{e[0]}_{e[1]}")
    
    for u in G.nodes():
        mod.addConstr(grb.quicksum([x[u, k] for k in range(0, K)]) == 1, f"vertice_{u}_un_color")
    
    for k in range(0, K):
        mod.addConstr(r[k] <= grb.quicksum([x[u, k] for u in G.nodes()]), f"color_{k}_tiene_elementos")
    
    mod.addConstr(grb.quicksum([r[k] for k in range(0, K)]) == theta, "colores_usados")
    
    for e in G.edges():
        for k in range(0, K):
            mod.addConstr(y[e[0], e[1], k] + y[e[1], e[0], k] <= x[e[0], k] * x[e[1], k])
    
    mod.addConstr(grb.quicksum(y[e[0], e[1], k] for e in G.edges() for k in range(0, K)) == N - theta, "total_aristas")
    
    for k in range(0, K):
        mod.addConstr(grb.quicksum(y[e[0], e[1], k] for e in G.edges()) == grb.quicksum(x[u, k] for u in G.nodes()) - r[k])

    M = 10000
    for u in G.nodes():
        for k in range(0, K):
            mod.addConstr(aux[u, k] == dj.get_datos(c, u, n) - media[k])
            mod.addGenConstrAbs(dif_abs[u, k], aux[u, k], "abs_" + str(u) + "_" + str(k))
            mod.addGenConstrIndicator(x[u, k], 1, p[u, k] - dif_abs[u, k], grb.GRB.EQUAL, 0, "save_abs_" + str(u) + "_" + str(k))
            mod.addGenConstrIndicator(x[u, k], 0, p[u, k], grb.GRB.EQUAL, 0, "save_abs_b_" + str(u) + "_" + str(k))
    
    for k in range(0, K):
        mod.addConstr(media[k] * z[k] == grb.quicksum([dj.get_datos(c, u, n) * x[u, k] for u in G.nodes()]), f"media_region_{k}")
        mod.addConstr(media[k] <= M * r[k])
    
    mod.addConstr(H_num == grb.quicksum([p[u, k] * p[u, k] for u in G.nodes() for k in range(0, K)]), "homogeneidad_num")
    mod.addConstr(H_den == ((VT * N) - (VT * theta)) * (1 - alpha), "homogeneidad_den")
    mod.addConstr(H_den >= H_num, "homogeneidad_constr")
    
    mod._y = y
    mod._x = x
    
    return mod, x, y, r, z, theta, media, p, dif_abs, aux, H_num, H_den


def create_model_flow(G, c, m, n, VT, alpha, type='grid'):
    """
    Creates the flow model with variables for acyclicity.
    This model uses flow variables V[u,k] to enforce a tree structure.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        c (dict): Data values for each vertex
        m (int): Number of rows
        n (int): Number of columns
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
        type (str): 'grid' or 'graph' - determines how K is calculated
    
    Returns:
        mod (gurobipy.Model): Gurobi model
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        media (dict): Mean value per color
        p (dict): Absolute deviation variables
        dif_abs (dict): Absolute difference auxiliary variables
        V_var (dict): Flow variables for topological ordering
        H_num (gurobipy.Var): Numerator of homogeneity constraint
        H_den (gurobipy.Var): Denominator of homogeneity constraint
    """
    if type == 'grid':
        K = m * n
    else:
        K = G.number_of_nodes() - 1  
    
    N = G.number_of_nodes()
    
    x = {}
    y = {}
    z = {}
    r = {}
    V_var = {}
    media = {}
    p = {}
    dif_abs = {}
    aux = {}
    
    mod = grb.Model("msf_ssmz_flow")
    
    for u in G.nodes():
        for k in range(0, K):
            x[u, k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"x_{u}_{k}")
            V_var[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, name=f"V_{u}_{k}")
            if u != K:
                p[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"p_{u}_{k}")
                dif_abs[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"dabs_{u}_{k}")
                aux[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f"aux_{u}_{k}")
    
    for e in G.edges():
        for k in range(0, K):
            y[e[0], e[1], k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"y_{e[0]}_{e[1]}_{k}")
    
    for k in range(0, K):
        r[k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"r_{k}")
        z[k] = mod.addVar(vtype=grb.GRB.INTEGER, name=f"z_{k}")
        media[k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, name=f"media_{k}")
    
    theta = mod.addVar(vtype=grb.GRB.INTEGER, name="theta")
    H_num = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hnum")
    H_den = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hden")
    
    mod.setObjective(theta, grb.GRB.MINIMIZE)
    
   
    for k in range(0, K):
        mod.addConstr(grb.quicksum([x[u, k] for u in G.nodes() if u != K]) == z[k], f"num_vertices_color_{k}")
    
    for u in G.nodes():
        if u != K:
            mod.addConstr(grb.quicksum([x[u, k] for k in range(0, K)]) == 1, f"vertice_{u}_un_color")
        else:
            mod.addConstr(grb.quicksum([x[u, k] for k in range(0, K)]) == theta, f"sumidero_tiene_theta_colores")
    
    for u in G.nodes():
        for k in range(0, K):
            mod.addConstr(x[u, k] <= r[k], f"uso_color_{k}_vertice_{u}")
    
    for e in G.edges():
        for k in range(0, K):
            mod.addConstr(y[e[0], e[1], k] <= r[k], f"uso_color_{k}_arista_{e[0]}_{e[1]}")
    

    for k in range(0, K):
        mod.addConstr(grb.quicksum([y[u, K, k] for u in G.nodes() if u != K]) == r[k], f"sumidero_entrada_{k}")
        mod.addConstr(grb.quicksum([y[K, u, k] for u in G.nodes() if u != K]) == 0, f"sumidero_salida_{k}")
    
    mod.addConstr(grb.quicksum([r[k] for k in range(0, K)]) == theta, "colores_usados")
    
    for k in range(0, K):
        mod.addConstr(r[k] <= grb.quicksum([x[u, k] for u in G.nodes() if u != K]), f"color_{k}_tiene_elementos")
    
    for e in G.edges():
        if e[0] < e[1]:
            for k in range(0, K):
                mod.addConstr(y[e[0], e[1], k] + y[e[1], e[0], k] <= x[e[0], k] * x[e[1], k])
    
    for e in G.edges():
        mod.addConstr(grb.quicksum([y[e[0], e[1], k] for k in range(0, K)]) <= 1, f"arista_{e[0]}_{e[1]}_un_color_maximo")
    
    M = 10000
    for u in G.nodes():
        if u != K:
            for k in range(0, K):
                mod.addConstr(grb.quicksum([y[u, q, k] for q in G.neighbors(u) if q <= K]) == x[u, k], f"flujo_salida_{u}_{k}")
                mod.addConstr(aux[u, k] == dj.get_datos(c, u, n) - media[k])
                mod.addGenConstrAbs(dif_abs[u, k], aux[u, k], "abs_" + str(u) + "_" + str(k))
                mod.addGenConstrIndicator(x[u, k], 1, p[u, k] - dif_abs[u, k], grb.GRB.EQUAL, 0, "save_abs_" + str(u) + "_" + str(k))
                mod.addGenConstrIndicator(x[u, k], 0, p[u, k], grb.GRB.EQUAL, 0, "save_abs_b_" + str(u) + "_" + str(k))
    
    for u in G.nodes():
        if u != K:
            for e in G.edges():
                if u == e[0]:
                    for k in range(0, K):
                        mod.addConstr(V_var[e[0], k] >= V_var[e[1], k] + y[e[0], e[1], k] - (N) * (1 - y[e[0], e[1], k]))
        else:
            for k in range(0, K):
                mod.addConstr(V_var[u, k] == 0)
    
    for k in range(0, K):
        mod.addConstr(media[k] * z[k] == grb.quicksum([dj.get_datos(c, u, n) * x[u, k] for u in G.nodes() if u != K]), f"media_region_{k}")
        mod.addConstr(media[k] <= M * r[k])
    
    mod.addConstr(H_num == grb.quicksum([p[u, k] * p[u, k] for u in G.nodes() if u != K for k in range(0, K)]), "homogeneidad_num")
    mod.addConstr(H_den == ((VT * N) - (VT * theta)) * (1 - alpha), "homogeneidad_den")
    mod.addConstr(H_den >= H_num, "homogeneidad_constr")
    
    mod.update()
    return mod, x, y, r, z, theta, media, p, dif_abs, V_var, H_num, H_den


# ============================================================================
# 7.1 MODEL CREATION FOR PLANAR GRAPHS
# ============================================================================

def create_model_flow_graph(G, VT, alpha, type='graph'):
    """
    Creates the flow model for planar graphs.
    Similar to create_model_flow but adapted for general graphs.
    
    Inputs:
        G (networkx.DiGraph): Directed graph (with sink node)
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
        type (str): 'graph' for planar graphs
    
    Returns:
        mod (gurobipy.Model): Gurobi model
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        media (dict): Mean value per color
        p (dict): Absolute deviation variables
        dif_abs (dict): Absolute difference auxiliary variables
        V_var (dict): Flow variables for topological ordering
        H_num (gurobipy.Var): Numerator of homogeneity constraint
        H_den (gurobipy.Var): Denominator of homogeneity constraint
    """
    K = G.number_of_nodes() - 1 
    N = max(G.nodes())
    
    x = {}
    y = {}
    z = {}
    r = {}
    V_var = {}
    media = {}
    p = {}
    dif_abs = {}
    aux = {}
    
    mod = grb.Model("msf_ssmz_flow_graph")
    
    for u in G.nodes():
        for k in range(0, K):
            x[u, k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"x_{u}_{k}")
            V_var[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, name=f"V_{u}_{k}")
            if u != N:
                p[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"p_{u}_{k}")
                dif_abs[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"dabs_{u}_{k}")
                aux[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f"aux_{u}_{k}")
    
    for e in G.edges():
        for k in range(0, K):
            y[e[0], e[1], k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"y_{e[0]}_{e[1]}_{k}")
    
    for k in range(0, K):
        r[k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"r_{k}")
        z[k] = mod.addVar(vtype=grb.GRB.INTEGER, name=f"z_{k}")
        media[k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, name=f"media_{k}")
    
    theta = mod.addVar(vtype=grb.GRB.INTEGER, name="theta")
    H_num = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hnum")
    H_den = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hden")
    
    mod.setObjective(theta, grb.GRB.MINIMIZE)
    
    for k in range(0, K):
        mod.addConstr(grb.quicksum([x[u, k] for u in G.nodes() if u != N]) == z[k], f"num_vertices_color_{k}")
    
    for u in G.nodes():
        if u != N:
            mod.addConstr(grb.quicksum([x[u, k] for k in range(0, K)]) == 1, f"vertice_{u}_un_color")
        else:
            mod.addConstr(grb.quicksum([x[u, k] for k in range(0, K)]) == theta, f"sumidero_tiene_theta_colores")
    
    for u in G.nodes():
        for k in range(0, K):
            mod.addConstr(x[u, k] <= r[k], f"uso_color_{k}_vertice_{u}")
    
    for e in G.edges():
        for k in range(0, K):
            mod.addConstr(y[e[0], e[1], k] <= r[k], f"uso_color_{k}_arista_{e[0]}_{e[1]}")
    
    for k in range(0, K):
        mod.addConstr(grb.quicksum([y[u, N, k] for u in G.nodes() if u != N]) == r[k], f"sumidero_entrada_{k}")
        mod.addConstr(grb.quicksum([y[N, u, k] for u in G.nodes() if u != N]) == 0, f"sumidero_salida_{k}")
    
    mod.addConstr(grb.quicksum([r[k] for k in range(0, K)]) == theta, "colores_usados")
    
    for k in range(0, K):
        mod.addConstr(r[k] <= grb.quicksum([x[u, k] for u in G.nodes() if u != N]), f"color_{k}_tiene_elementos")
    
    for e in G.edges():
        if e[0] < e[1]:
            for k in range(0, K):
                mod.addConstr(y[e[0], e[1], k] + y[e[1], e[0], k] <= x[e[0], k] * x[e[1], k])
    
    for e in G.edges():
        mod.addConstr(grb.quicksum([y[e[0], e[1], k] for k in range(0, K)]) <= 1, f"arista_{e[0]}_{e[1]}_un_color_maximo")
    
    M = 10000
    for u in G.nodes():
        if u != N:
            for k in range(0, K):
                mod.addConstr(grb.quicksum([y[u, q, k] for q in G.neighbors(u)]) == x[u, k], f"flujo_salida_{u}_{k}")
                sample_val = G.nodes[u]['sample'] if 'sample' in G.nodes[u] else 0
                mod.addConstr(aux[u, k] == sample_val - media[k])
                mod.addGenConstrAbs(dif_abs[u, k], aux[u, k], "abs_" + str(u) + "_" + str(k))
                mod.addGenConstrIndicator(x[u, k], 1, p[u, k] - dif_abs[u, k], grb.GRB.EQUAL, 0, "save_abs_" + str(u) + "_" + str(k))
                mod.addGenConstrIndicator(x[u, k], 0, p[u, k], grb.GRB.EQUAL, 0, "save_abs_b_" + str(u) + "_" + str(k))
    
    for u in G.nodes():
        if u != N:
            for e in G.edges():
                if u == e[0]:
                    for k in range(0, K):
                        mod.addConstr(V_var[e[0], k] >= V_var[e[1], k] + y[e[0], e[1], k] - (N) * (1 - y[e[0], e[1], k]))
        else:
            for k in range(0, K):
                mod.addConstr(V_var[u, k] == 0)
    
    for k in range(0, K):
        mod.addConstr(media[k] * z[k] == grb.quicksum([G.nodes[u]['sample'] * x[u, k] for u in G.nodes() if u != N]), f"media_region_{k}")
        mod.addConstr(media[k] <= M * r[k])
    
    mod.addConstr(H_num == grb.quicksum([p[u, k] * p[u, k] for u in G.nodes() if u != N for k in range(0, K)]), "homogeneidad_num")
    mod.addConstr(H_den == ((VT * K) - (VT * theta)) * (1 - alpha), "homogeneidad_den")
    mod.addConstr(H_den >= H_num, "homogeneidad_constr")
    
    mod.update()
    return mod, x, y, r, z, theta, media, p, dif_abs, V_var, H_num, H_den


def create_model_lazy_graph(G, VT, alpha):
    """
    Creates the lazy model for planar graphs.
    
    Inputs:
        G (networkx.DiGraph): Directed graph
        VT (float): Total variance normalization factor
        alpha (float): Homogeneity threshold
    
    Returns:
        mod (gurobipy.Model): Gurobi model
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables
        r (dict): Color usage variables
        z (dict): Number of vertices per color
        theta (gurobipy.Var): Number of colors variable
        media (dict): Mean value per color
        p (dict): Absolute deviation variables
        dif_abs (dict): Absolute difference auxiliary variables
        aux (dict): Auxiliary variables for linearization
        H_num (gurobipy.Var): Numerator of homogeneity constraint
        H_den (gurobipy.Var): Denominator of homogeneity constraint
    """
    K = G.number_of_nodes()  
    x = {}
    y = {}
    z = {}
    r = {}
    media = {}
    p = {}
    dif_abs = {}
    aux = {}
    
    mod = grb.Model("msf_ssmz_lazy_graph")

    for u in G.nodes():
        for k in range(0, K):
            x[u, k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"x_{u}_{k}")
            p[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"p_{u}_{k}")
            dif_abs[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name=f"dabs_{u}_{k}")
            aux[u, k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, ub=grb.GRB.INFINITY, name=f"aux_{u}_{k}")
    
    for e in G.edges():
        for k in range(0, K):
            y[e[0], e[1], k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"y_{e[0]}_{e[1]}_{k}")
    
    for k in range(0, K):
        r[k] = mod.addVar(vtype=grb.GRB.BINARY, name=f"r_{k}")
        z[k] = mod.addVar(vtype=grb.GRB.INTEGER, name=f"z_{k}")
        media[k] = mod.addVar(vtype=grb.GRB.CONTINUOUS, name=f"media_{k}")
    
    theta = mod.addVar(vtype=grb.GRB.CONTINUOUS,lb=0, ub=grb.GRB.INFINITY, name="theta")
    H_num = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hnum")
    H_den = mod.addVar(vtype=grb.GRB.CONTINUOUS, lb=0, ub=grb.GRB.INFINITY, name="Hden")
    
    mod.setObjective(theta, grb.GRB.MINIMIZE)
    
    for k in range(0, K):
        mod.addConstr(grb.quicksum([x[u, k] for u in G.nodes()]) == z[k], f"num_vertices_color_{k}")
    
    for u in G.nodes():
        for k in range(0, K):
            mod.addConstr(x[u, k] <= r[k], f"uso_color_{k}_vertice_{u}")
    
    for e in G.edges():
        for k in range(0, K):
            mod.addConstr(y[e[0], e[1], k] <= r[k], f"uso_color_{k}_arista_{e[0]}_{e[1]}")
    
    for u in G.nodes():
        mod.addConstr(grb.quicksum([x[u, k] for k in range(0, K)]) == 1, f"vertice_{u}_un_color")
    
    for k in range(0, K):
        mod.addConstr(r[k] <= grb.quicksum([x[u, k] for u in G.nodes()]), f"color_{k}_tiene_elementos")
    
    mod.addConstr(grb.quicksum([r[k] for k in range(0, K)]) == theta, "colores_usados")
    
    for e in G.edges():
        for k in range(0, K):
            mod.addConstr(y[e[0], e[1], k] + y[e[1], e[0], k] <= x[e[0], k] * x[e[1], k],f"anticolor_{e[0]}_{e[1]}_{k}")
    
    mod.addConstr(grb.quicksum(y[e[0], e[1], k] for e in G.edges() for k in range(0, K)) ==
                  K - theta, "total_aristas")
    
    for k in range(0, K):
        mod.addConstr(grb.quicksum(y[e[0], e[1], k] for e in G.edges()) == 
                      grb.quicksum(x[u, k] for u in G.nodes()) - r[k],f"uso_color_{k}_aristas")
    
    M = 10000
    for u in G.nodes():
        sample_val = G.nodes[u]['sample'] if 'sample' in G.nodes[u] else 0
        for k in range(0, K):
            mod.addConstr(aux[u, k] == sample_val - media[k])
            mod.addGenConstrAbs(dif_abs[u, k], aux[u, k], "abs_" + str(u) + "_" + str(k))
            mod.addGenConstrIndicator(x[u, k], 1, p[u, k] - dif_abs[u, k], grb.GRB.EQUAL, 0, "save_abs_" + str(u) + "_" + str(k))
            mod.addGenConstrIndicator(x[u, k], 0, p[u, k], grb.GRB.EQUAL, 0, "save_abs_b_" + str(u) + "_" + str(k))
    
    for k in range(0, K):
        mod.addConstr(media[k] * z[k] == grb.quicksum([G.nodes[u]['sample'] * x[u, k] for u in G.nodes()]), f"media_region_{k}")
        mod.addConstr(media[k] <= M * r[k])
    
    mod.addConstr(H_num == grb.quicksum([p[u, k] * p[u, k] for u in G.nodes() for k in range(0, K)]), "homogeneidad_num")
    mod.addConstr(H_den == ((VT * K) - (VT * theta)) * (1 - alpha), "homogeneidad_den")
    mod.addConstr(H_den >= H_num, "homogeneidad_constr")
    
    mod._y = y
    mod._x = x
    
    return mod, x, y, r, z, theta, media, p, dif_abs, aux, H_num, H_den


# ============================================================================
# 8. SOLUTION EXTRACTION
# ============================================================================

def extract_solution(mod, G, K, x, y=None, model_type="flow"):
    """
    Extracts the solution from the optimized model.
    
    Inputs:
        mod (gurobipy.Model): Gurobi model
        G (networkx.DiGraph): Directed graph
        K (int): Number of colors
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables (optional)
        model_type (str): 'lazy' or 'flow'
    
    Returns:
        lista_nodos (dict): Dictionary mapping color index to list of vertices
        lista_arcos (dict): Dictionary mapping color index to list of edges
    """
    lista_nodos = {}
    lista_arcos = {}
    
    for k in range(0, K):
        nodos_k = []
        for u in G.nodes():
            if u != K and (u, k) in x:
                try:
                    if hasattr(x[u, k], 'x') and abs(x[u, k].x) >= 0.01:
                        nodos_k.append(u)
                except:
                    pass
        if len(nodos_k) > 0:
            lista_nodos[k] = nodos_k
        else:
            lista_nodos[k] = []
    
    if y is not None:
        for k in range(0, K):
            aristas_k = []
            for e in G.edges():
                if (e[0], e[1], k) in y and e[0] < e[1]:
                    try:
                        if hasattr(y[e[0], e[1], k], 'x') and abs(y[e[0], e[1], k].x) >= 0.01:
                            if e[0] != K and e[1] != K:
                                aristas_k.append(e)
                    except:
                        pass
                if (e[1], e[0], k) in y and e[0] < e[1]:
                    try:
                        if hasattr(y[e[1], e[0], k], 'x') and abs(y[e[1], e[0], k].x) >= 0.01:
                            if e[0] != K and e[1] != K:
                                if e not in aristas_k: 
                                    aristas_k.append(e)
                    except:
                        pass
            if len(aristas_k) > 0 :
                lista_arcos[k] = aristas_k
            else:
                lista_arcos[k] = []
    
    return lista_nodos, lista_arcos


def extract_solution_graph(mod, G, K, x, y=None, model_type="flow"):
    """
    Extracts the solution from the optimized model for planar graphs.
    
    Inputs:
        mod (gurobipy.Model): Gurobi model
        G (networkx.DiGraph): Directed graph
        K (int): Number of colors
        x (dict): Vertex-color assignment variables
        y (dict): Edge-color variables (optional)
        model_type (str): 'lazy' or 'flow'
    
    Returns:
        lista_nodos (dict): Dictionary mapping color index to list of vertices
        lista_arcos (dict): Dictionary mapping color index to list of edges
    """
    lista_nodos = {}
    lista_arcos = {}
    if model_type == "lazy":
        N=max(G.nodes() )
    else:
        N=max(G.nodes())
    for k in range(0, K):
        nodos_k = []
        for u in G.nodes():
            if model_type=="flow":
                if u != N:
                    try:
                        if abs(x[u, k].x) >= 0.01:
                            nodos_k.append(u)
                    except:
                        pass
            else:
                try:
                    if abs(x[u, k].x) >= 0.01:
                        nodos_k.append(u)
                except:
                    pass
        if len(nodos_k) > 0:
            lista_nodos[k] = nodos_k
        else:
            lista_nodos[k] = []
 
    if y is not None:
        for k in range(0, K):
            aristas_k = []
            for e in G.edges():
                if model_type=="flow":
                    if e[0] != N and e[1] != N:
                        if (e[0], e[1], k) in y:
                            try:
                                if hasattr(y[e[0], e[1], k], 'x') and abs(y[e[0], e[1], k].x) >= 0.01:
                                    aristas_k.append((e[0], e[1]))
                            except:
                                pass
                        if (e[1], e[0], k) in y:
                            try:
                                if hasattr(y[e[1], e[0], k], 'x') and abs(y[e[1], e[0], k].x) >= 0.01:
                                    if (e[0], e[1]) not in aristas_k and (e[1], e[0]) not in aristas_k:
                                        aristas_k.append((e[0], e[1]))
                            except:
                                pass
                else:
                    if (e[0], e[1], k) in y:
                            try:
                                if hasattr(y[e[0], e[1], k], 'x') and abs(y[e[0], e[1], k].x) >= 0.01:
                                    aristas_k.append((e[0], e[1]))
                            except:
                                pass
                    if (e[1], e[0], k) in y:
                            try:
                                if hasattr(y[e[1], e[0], k], 'x') and abs(y[e[1], e[0], k].x) >= 0.01:
                                    if (e[0], e[1]) not in aristas_k and (e[1], e[0]) not in aristas_k:
                                        aristas_k.append((e[0], e[1]))
                            except:
                                pass

            if len(aristas_k) > 0:
                lista_arcos[k] = aristas_k
            else:
                lista_arcos[k] = []
    return lista_nodos, lista_arcos


# ============================================================================
# 9. REPORT GENERATION
# ============================================================================

def generate_report(execution_info, heuristic_results, model_results, total_time, output_file):
    """
    Generates a detailed report of the execution.
    
    Inputs:
        execution_info (dict): Execution parameters and metadata
        heuristic_results (dict): Heuristic solution results (or None)
        model_results (dict): Optimization model results
        total_time (float): Total execution time
        output_file (str): Path to output report file
    
    Returns:
        None (writes report to file)
    """
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SITE SPECIFIC MANAGEMENT ZONE REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("EXECUTION INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Date: {execution_info['date']}\n")
        f.write(f"  Data file: {execution_info['data_file']}\n")
        f.write(f"  Alpha: {execution_info['alpha']}\n")
        f.write(f"  Graph type: {execution_info.get('graph_type', 'grid')}\n")
        if execution_info.get('graph_type') == 'grid':
            f.write(f"  Grid dimensions: {execution_info['m']} x {execution_info['n']} = {execution_info['m'] * execution_info['n']} nodes\n")
        f.write(f"  VT (variance): {execution_info['VT']:.6f}\n")
        f.write(f"  Model type: {execution_info['model_type'].upper()}\n")
        f.write(f"  Heuristic: {execution_info['heuristic_name'] if execution_info['heuristic_name'] else 'None'}\n")
        f.write(f"  Time limit: {execution_info['time_limit']}s\n")
        f.write("\n")
        
        if heuristic_results:
            f.write("HEURISTIC RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Heuristic type: {heuristic_results['name'].upper()}\n")
            f.write(f"  Number of components: {heuristic_results['components']}\n")
            f.write(f"  Theta value: {heuristic_results['theta']}\n")
            f.write(f"  H value: {heuristic_results['h_value']:.6f}\n")
            f.write(f"  Iterations: {heuristic_results['iterations']}\n")
            f.write(f"  Execution time: {heuristic_results['time']:.2f}s\n")
            f.write("\n")
        
        f.write("OPTIMIZATION RESULTS\n")
        f.write("-" * 40 + "\n")
        if model_results:
            f.write(f"  Theta value: {model_results['theta']}\n")
            f.write(f"  H value: {model_results['h_value']:.6f}\n")
            f.write(f"  Number of components: {model_results['num_components']}\n")
            f.write(f"  Execution time: {model_results['time']:.2f}s\n")
            if model_results.get('gap') is not None:
                f.write(f"  MIP Gap: {model_results['gap']:.6f}\n")
            if model_results.get('status') is not None:
                f.write(f"  Solver status: {model_results['status']}\n")
            f.write("\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total execution time: {total_time:.2f}s\n")
        
        if heuristic_results and model_results:
            improvement = ((heuristic_results['theta'] - model_results['theta']) / heuristic_results['theta']) * 100 if heuristic_results['theta'] > 0 else 0
            f.write(f"  Improvement (Heuristic -> Model): {heuristic_results['theta']} -> {model_results['theta']} ({improvement:.1f}%)\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n  Report saved to: {output_file}")


# ============================================================================
# 10. RUN FUNCTIONS
# ============================================================================

def run_heuristic_only(m, n, c, VT, alpha, heuristic_name, data_file, draw_flag, output_flag, plot_flag, graph_type='grid'):
    """
    Runs only the heuristic without optimization.
    
    Inputs:
        m (int): Number of rows
        n (int): Number of columns
        c (dict): Data values
        VT (float): Total variance
        alpha (float): Homogeneity threshold
        heuristic_name (str): 'h1' or 'h2'
        data_file (str): Path to data file
        draw_flag (bool): Whether to generate drawings
        output_flag (bool): Whether to generate report
        plot_flag (bool): Whether to generate convergence plots
        graph_type (str): 'grid' or 'graph'
    
    Returns:
        sol (list): List of components
        aristas (dict): Edges per component
        theta (int): Objective value
        valorH (float): Homogeneity value
        heur_time (float): Execution time
    """
    print("\n" + "=" * 60)
    print(f"RUNNING HEURISTIC {heuristic_name.upper()} ONLY")
    print(f"Graph type: {graph_type.upper()}")
    print("=" * 60)
    
    resultados_dir = os.environ.get("SSMZ_OUTDIR") or ("/results" if os.path.isdir("/results") else "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)
    
    if graph_type == 'grid':
        G_heuristic = create_graph_noaux(m, n, c)
        heur_func = heuristic_h1 if heuristic_name == "h1" else heuristic_h2
    else:
        G_heuristic, n_graph, m_graph = load_data_graph(data_file, 'lazy', add_sink=False)
        if heuristic_name == "h1":
            heur_func = heuristic_h1_graph
        else:
            heur_func = heuristic_h2_graph
    
    inicio = time.time()
    sol, aristas, iterations = heur_func(G_heuristic, VT, alpha)
    heur_time = time.time() - inicio
    
    st = dj.sol_no_lineas(sol, m, n)
    st.up_vars(c)
    theta = st.fobj(VT, alpha)
    valorH = st.calcH(VT)
    
    print(f"\nHeuristic {heuristic_name.upper()} Results:")
    print(f"  - Theta: {theta}")
    print(f"  - H Value: {valorH}")
    print(f"  - VT: {VT}")
    print(f"  - Iterations: {iterations}")
    print(f"  - Components: {len(sol)}")
    print(f"  - Time: {heur_time:.2f}s")
    
    basename = data_file.split("/")[-1].split(".")[0]
    
    if draw_flag:
        dibname = os.path.join(resultados_dir, f"{basename}_{alpha}_{heuristic_name}_{graph_type}")
        if graph_type == 'grid':
            dibujo_heuristic(f"{dibname}_graph", G_heuristic, m, n, sol, aristas, 789123)
            dibujo(f"{dibname}_rec", m, n, sol, 789123)
        else:
            dibujo_graph(f"{dibname}_graph", G_heuristic, sol, aristas, method='tikz', seed=789123)
        print(f"\n  Graphics saved to: {dibname}_graph.pdf")
    
    if output_flag:
        output_file = os.path.join(resultados_dir, f"{basename}_{alpha}_{heuristic_name}_{graph_type}_report.txt")
        execution_info = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_file': data_file,
            'alpha': alpha,
            'm': m,
            'n': n,
            'VT': VT,
            'model_type': 'heuristic',
            'heuristic_name': heuristic_name,
            'time_limit': 0,
            'graph_type': graph_type
        }
        heuristic_results = {
            'name': heuristic_name,
            'components': len(sol),
            'theta': theta,
            'h_value': valorH,
            'iterations': iterations,
            'time': heur_time
        }
        generate_report(execution_info, heuristic_results, None, heur_time, output_file)
    
    if plot_flag:
        print("  Note: 'plot' flag is only valid for optimization models with Gurobi logs.")
        print("  Skipping plot generation for heuristic-only run.")
    
    return sol, aristas, theta, valorH, heur_time


def run_heuristic_only_graph(G, VT, alpha, heuristic_name, data_file, draw_flag, output_flag, plot_flag):
    """
    Runs heuristic on a planar graph.
    
    Inputs:
        G (networkx.Graph): Input graph
        VT (float): Total variance
        alpha (float): Homogeneity threshold
        heuristic_name (str): 'h1' or 'h2'
        data_file (str): Path to data file
        draw_flag (bool): Whether to generate drawings
        output_flag (bool): Whether to generate report
        plot_flag (bool): Whether to generate convergence plots
    
    Returns:
        sol (list): List of components
        aristas (dict): Edges per component
        theta (int): Objective value
        valorH (float): Homogeneity value
        heur_time (float): Execution time
    """
    print("\n" + "=" * 60)
    print(f"RUNNING HEURISTIC {heuristic_name.upper()} ON PLANAR GRAPH")
    print("=" * 60)
    
    resultados_dir = os.environ.get("SSMZ_OUTDIR") or ("/results" if os.path.isdir("/results") else "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)
    
    inicio = time.time()
    if heuristic_name == "h1":
        sol, aristas, iterations = heuristic_h1_graph(G, VT, alpha)
    else:
        sol, aristas, iterations = heuristic_h2_graph(G, VT, alpha)
    heur_time = time.time() - inicio
    
    n_components = len(sol)
    theta = n_components
    
    st = dj.sol_graph(G, sol, 'lazy')
    st.up_vars()
    valorH = st.calcH(VT)
    
    print(f"\nHeuristic {heuristic_name.upper()} Results:")
    print(f"  - Theta: {theta}")
    print(f"  - H Value: {valorH}")
    print(f"  - VT: {VT}")
    print(f"  - Iterations: {iterations}")
    print(f"  - Components: {len(sol)}")
    print(f"  - Time: {heur_time:.2f}s")
    
    basename = data_file.split("/")[-1].split(".")[0]
    
    if draw_flag:
        dibname = os.path.join(resultados_dir, f"{basename}_{alpha}_{heuristic_name}_graph")
        dibujo_graph(dibname, G, sol, aristas, method='graphviz', seed=789123)
        print(f"\n  Graphics saved to: {dibname}.pdf")
    
    if output_flag:
        output_file = os.path.join(resultados_dir, f"{basename}_{alpha}_{heuristic_name}_graph_report.txt")
        execution_info = {
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_file': data_file,
            'alpha': alpha,
            'm': 0,
            'n': 0,
            'VT': VT,
            'model_type': 'heuristic_graph',
            'heuristic_name': heuristic_name,
            'time_limit': 0,
            'graph_type': 'graph'
        }
        heuristic_results = {
            'name': heuristic_name,
            'components': len(sol),
            'theta': theta,
            'h_value': valorH,
            'iterations': iterations,
            'time': heur_time
        }
        generate_report(execution_info, heuristic_results, None, heur_time, output_file)
    
    return sol, aristas, theta, valorH, heur_time


def run_model(model_type, heuristic_name, time_limit, m, n, c, VT, alpha, 
              data_file, draw_flag, output_flag, plot_flag, graph_type='grid'):
    """
    Runs an optimization model (lazy or flow) with optional heuristic warm start.
    
    Inputs:
        model_type (str): 'lazy' or 'flow'
        heuristic_name (str): None, 'h1', or 'h2'
        time_limit (int): Time limit in seconds
        m (int): Number of rows
        n (int): Number of columns
        c (dict): Data values
        VT (float): Total variance
        alpha (float): Homogeneity threshold
        data_file (str): Path to data file
        draw_flag (bool): Whether to generate drawings
        output_flag (bool): Whether to generate report
        plot_flag (bool): Whether to generate convergence plots
        graph_type (str): 'grid' or 'graph'
    
    Returns:
        mod (gurobipy.Model): Solved model
        lista_nodos (dict): Final components
        lista_arcos (dict): Final edges per component
        theta (int): Final objective value
        valorH (float): Final homogeneity value
        total_time (float): Total execution time
    """
    
    print("\n" + "=" * 60)
    print(f"RUNNING MODEL: {model_type.upper()}")
    print(f"Graph type: {graph_type.upper()}")
    print(f"Heuristic: {heuristic_name.upper() if heuristic_name else 'None'}")
    print(f"Time limit: {time_limit}s")
    print(f"Draw visualizations: {draw_flag}")
    print(f"Generate report: {output_flag}")
    print(f"Generate convergence plot: {plot_flag}")
    print("=" * 60)
    
    resultados_dir = os.environ.get("SSMZ_OUTDIR") or ("/results" if os.path.isdir("/results") else "resultados")
    if not os.path.exists(resultados_dir):
        os.makedirs(resultados_dir)
    
    basename = data_file.split("/")[-1].split(".")[0]
    
    # ========================================================================
    # GRID MODE (Original functionality)
    # ========================================================================
    if graph_type == 'grid':
        if model_type == "lazy":
            G_und = create_graph_noaux(m, n, c)
            K = m * n
            G = G_und.to_directed()
        else:
            G_und = create_graph(m, n, c)
            K = m * n
            G = G_und.to_directed()
        
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if model_type == "lazy":
            mod, x, y, r, z, theta, media, p, dif_abs, aux, H_num, H_den = create_model_lazy(G, c, m, n, VT, alpha)
            use_callback = True
            callback_func = lazy_callback
        else:
            mod, x, y, r, z, theta, media, p, dif_abs, V_var, H_num, H_den = create_model_flow(G, c, m, n, VT, alpha, type='grid')
            use_callback = False
            callback_func = None
        
        heur_time = 0
        sol = None
        aristas_heur = None
        iterations = 0
        heuristic_theta = 0
        heuristic_h = 0
        
        if heuristic_name:
            print("\nGenerating initial solution with heuristic...")
            G_heuristic = create_graph_noaux(m, n, c)
            inicio = time.time()
            
            if heuristic_name == "h1":
                sol, aristas_heur, iterations = heuristic_h1(G_heuristic, VT, alpha)
            else:
                sol, aristas_heur, iterations = heuristic_h2(G_heuristic, VT, alpha)
            
            heur_time = time.time() - inicio
            print(f"  Heuristic completed in {heur_time:.2f}s")
            print(f"  Components: {len(sol)}, Iterations: {iterations}")
            
            st_heur = dj.sol_no_lineas(sol, m, n)
            st_heur.up_vars(c)
            heuristic_theta = st_heur.fobj(VT, alpha)
            heuristic_h = st_heur.calcH(VT)
            
            if draw_flag:
                dibname_hs = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_hs_{heuristic_name}_grid")
                dibujo_heuristic(dibname_hs, G_heuristic, m, n, sol, aristas_heur, 789123)
                os.system("rm *.aux")
                os.system("rm *.log")
            
            warm_start_heuristic(G, x, y, r, z, theta, sol, aristas_heur, K, model_type, mod, VT)
            verify_warm_start_feasibility(G, mod, x, y, r, z, theta, sol, aristas_heur, K, model_type)
        
        else:
            print("\nNo heuristic specified. Applying trivial warm start...")
            warm_start_trivial(G, x, y, r, z, theta, len(G.nodes()), model_type,graph_type)
        
        logfile = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_{heuristic_name if heuristic_name else 'no_ws'}_grid.log")
        dibname = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_{heuristic_name if heuristic_name else 'no_ws'}_grid")
        
        available_time = time_limit - heur_time
        if available_time < 60:
            available_time = 60
        

        print("\nSolving optimization model...")
        modelo_inicio = time.time()
        mod = configure_and_solve(mod, available_time, logfile, use_callback, callback_func)
        modelo_time = time.time() - modelo_inicio
        

        lista_nodos, lista_arcos = extract_solution(mod, G, K, x, y, model_type)
        if len(lista_nodos) > 0:
            st = dj.sol_no_lineas(lista_nodos, m, n)
            st.up_vars(c)
            theta = st.fobj(VT, alpha)
            valorH = st.calcH(VT)
        else:
            theta = 0
            valorH = 0
        

        gap = mod.MIPGap if hasattr(mod, 'MIPGap') else None
        status = mod.Status
        
        print(f"\nResults after optimization:")
        print(f"  - Theta: {theta}")
        print(f"  - H Value: {valorH}")
        print(f"  - Components: {len(lista_nodos)}")
        print(f"  - Time: {modelo_time:.2f}s")
        if gap is not None:
            print(f"  - Gap: {gap:.6f}")

        if draw_flag and len(lista_nodos) > 0:
            dibujo2(f"{dibname}_graph", G, m, n, lista_nodos, lista_arcos, 789123)
            dibujo(f"{dibname}_rec", m, n, lista_nodos, 789123)
            print(f"  Graphics: {dibname}_graph.pdf and {dibname}_rec.pdf")
            os.system("rm *.aux")
            os.system("rm *.log")
        

        if plot_flag:
            print("\nGenerating convergence plots...")
            plot_basename = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_{heuristic_name if heuristic_name else 'no_ws'}_grid_convergence")
            generate_convergence_plot_full(logfile, plot_basename, method='auto')
        
        total_time = heur_time + modelo_time
        
        print(f"\nSummary:")
        if heuristic_name:
            print(f"  - Heuristic ({heuristic_name.upper()}): {len(sol)} components, {heur_time:.2f}s")
        print(f"  - Optimization: {theta} components, {modelo_time:.2f}s")
        print(f"  - Total time: {total_time:.2f}s")
        
        if heuristic_name and sol and theta > 0:
            improvement = ((len(sol) - theta) / len(sol)) * 100
            print(f"  - Improvement: {len(sol)} -> {theta} ({improvement:.1f}%)")
        
        if output_flag:
            if heuristic_name:
                report_name = f"{basename}_{alpha}_{heuristic_name}_{model_type}_grid"
            else:
                report_name = f"{basename}_{alpha}_{model_type}_no_ws_grid"
            
            output_file = os.path.join(resultados_dir, f"{report_name}.txt")
            
            execution_info = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_file': data_file,
                'alpha': alpha,
                'm': m,
                'n': n,
                'VT': VT,
                'model_type': model_type,
                'heuristic_name': heuristic_name if heuristic_name else None,
                'time_limit': time_limit,
                'graph_type': 'grid'
            }
            
            heuristic_results = None
            if heuristic_name:
                heuristic_results = {
                    'name': heuristic_name,
                    'components': len(sol),
                    'theta': heuristic_theta,
                    'h_value': heuristic_h,
                    'iterations': iterations,
                    'time': heur_time
                }
            
            model_results = {
                'theta': theta,
                'h_value': valorH,
                'num_components': len(lista_nodos),
                'time': modelo_time,
                'gap': gap,
                'status': status
            }
            
            generate_report(execution_info, heuristic_results, model_results, total_time, output_file)
        
        return mod, lista_nodos, lista_arcos, theta, valorH, total_time
    
    # ========================================================================
    # GRAPH MODE (New functionality for planar graphs)
    # ========================================================================
    else:
        print("\nLoading planar graph...")
        G_und, n_graph, m_graph = load_data_graph(data_file, model_type, add_sink=(model_type == "flow"))
        G = G_und.to_directed()
        
        if model_type == "lazy":
            K = G.number_of_nodes()
        else:
            K = G.number_of_nodes() - 1  
        
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"K (colors): {K}")
        
  
        if model_type == "lazy":
            mod, x, y, r, z, theta, media, p, dif_abs, aux, H_num, H_den = create_model_lazy_graph(G, VT, alpha)
            use_callback = True
            callback_func = lazy_callback
        else:
            mod, x, y, r, z, theta, media, p, dif_abs, V_var, H_num, H_den = create_model_flow_graph(G, VT, alpha, type='graph')
            use_callback = False
            callback_func = None
        
        heur_time = 0
        sol = None
        aristas_heur = None
        iterations = 0
        heuristic_theta = 0
        heuristic_h = 0
        

        if heuristic_name:
            print("\nGenerating initial solution with heuristic...")
            G_heuristic, _, _ = load_data_graph(data_file, 'lazy', add_sink=False)
            inicio = time.time()
            
            if heuristic_name == "h1":
                sol, aristas_heur, iterations = heuristic_h1_graph(G_heuristic, VT, alpha)
            else:
                sol, aristas_heur, iterations = heuristic_h2_graph(G_heuristic, VT, alpha)
            
            heur_time = time.time() - inicio
            print(f"  Heuristic completed in {heur_time:.2f}s")
            print(f"  Components: {len(sol)}, Iterations: {iterations}")
            
            st_heur = dj.sol_graph(G_heuristic, sol, 'lazy')
            st_heur.up_vars()
            heuristic_theta = len(sol)
            heuristic_h = st_heur.calcH(VT)
            
            if draw_flag:
                dibname_hs = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_hs_{heuristic_name}_graph")
                dibujo_graph(dibname_hs, G_heuristic, sol, aristas_heur, method='tikz', seed=789123)
                dibujo_graph(dibname_hs, G_heuristic, sol, aristas_heur, method='graphviz', seed=789123)
                os.system("rm *.aux")
                os.system("rm *.log")
            
            warm_start_heuristic(G, x, y, r, z, theta, sol, aristas_heur, K, model_type, mod, VT)
            verify_warm_start_feasibility(G, mod, x, y, r, z, theta, sol, aristas_heur, K, model_type)
        
        else:
            print("\nNo heuristic specified. Applying trivial warm start...")
            warm_start_trivial(G, x, y, r, z, theta, K, model_type,graph_type)
        
        logfile = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_{heuristic_name if heuristic_name else 'no_ws'}_graph.log")
        dibname = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_{heuristic_name if heuristic_name else 'no_ws'}_graph")
        
        available_time = time_limit - heur_time
        if available_time < 60:
            available_time = 60
        

        print("\nSolving optimization model...")
        modelo_inicio = time.time()
        mod = configure_and_solve(mod, available_time, logfile, use_callback, callback_func)
        modelo_time = time.time() - modelo_inicio
        

        lista_nodos, lista_arcos = extract_solution_graph(mod, G, K, x, y, model_type)
        
        if len(lista_nodos) > 0:
            theta = sum(1 for k in lista_nodos.values() if len(k) > 0)
            st = dj.sol_graph(G, list(lista_nodos.values()), 'lazy')
            st.up_vars()
            valorH = st.calcH(VT)
        else:
            theta = 0
            valorH = 0
        
        gap = mod.MIPGap if hasattr(mod, 'MIPGap') else None
        status = mod.Status
        ncomponents=sum(1 for k in lista_nodos.values() if len(k) > 0)
        totalv=sum(len(k) for k in lista_nodos.values())
        totale=sum(len(k) for k in lista_arcos.values())
        print(f"\nResults after optimization:")
        print(f"  - Theta: {theta}")
        print(f"  - H Value: {valorH}")
        print(f"  - Components: {ncomponents}")
        print(f"  - Time: {modelo_time:.2f}s")      
        print(f"  - Total vertices: {totalv}")     
        print(f"  - Total edges: {totale}")
        if gap is not None:
            print(f"  - Status: {status}")
        if gap is not None:
            print(f"  - Gap: {gap:.6f}")
        if draw_flag and len(lista_nodos) > 0:
            G_draw, _, _ = load_data_graph(data_file, 'lazy', add_sink=False)
            dibujo_graph(f"{dibname}_graph", G_draw, lista_nodos, lista_arcos, method='tikz', seed=789123)
            dibujo_graph(f"{dibname}_graph", G_draw, lista_nodos, lista_arcos, method='graphviz', seed=789123)
            
            print(f"  Graphics: {dibname}_graph.pdf")
            os.system("rm *.aux")
            os.system("rm *.log")
        
        if plot_flag:
            print("\nGenerating convergence plots...")
            plot_basename = os.path.join(resultados_dir, f"{basename}_{alpha}_{model_type}_{heuristic_name if heuristic_name else 'no_ws'}_graph_convergence")
            generate_convergence_plot_full(logfile, plot_basename, method='auto')
        
        total_time = heur_time + modelo_time
        
        print(f"\nSummary:")
        if heuristic_name:
            print(f"  - Heuristic ({heuristic_name.upper()}): {len(sol)} components, {heur_time:.2f}s")
        print(f"  - Optimization: {theta} components, {modelo_time:.2f}s")
        print(f"  - Total time: {total_time:.2f}s")
        
        if heuristic_name and sol and theta > 0:
            improvement = ((len(sol) - theta) / len(sol)) * 100
            print(f"  - Improvement: {len(sol)} -> {theta} ({improvement:.1f}%)")

        if output_flag:
            if heuristic_name:
                report_name = f"{basename}_{alpha}_{heuristic_name}_{model_type}_graph"
            else:
                report_name = f"{basename}_{alpha}_{model_type}_no_ws_graph"
            
            output_file = os.path.join(resultados_dir, f"{report_name}.txt")
            
            execution_info = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_file': data_file,
                'alpha': alpha,
                'm': m,
                'n': n,
                'VT': VT,
                'model_type': model_type,
                'heuristic_name': heuristic_name if heuristic_name else None,
                'time_limit': time_limit,
                'graph_type': 'graph'
            }
            
            heuristic_results = None
            if heuristic_name:
                heuristic_results = {
                    'name': heuristic_name,
                    'components': len(sol),
                    'theta': heuristic_theta,
                    'h_value': heuristic_h,
                    'iterations': iterations,
                    'time': heur_time
                }
            
            model_results = {
                'theta': theta,
                'h_value': valorH,
                'num_components': len(lista_nodos),
                'time': modelo_time,
                'gap': gap,
                'status': status
            }
            
            generate_report(execution_info, heuristic_results, model_results, total_time, output_file)
        
        return mod, lista_nodos, lista_arcos, theta, valorH, total_time


def configure_and_solve(mod, time_limit, logfile, use_callback=False, callback_func=None):
    """
    Configures model parameters and solves it.
    
    Inputs:
        mod (gurobipy.Model): Gurobi model
        time_limit (float): Time limit in seconds
        logfile (str): Path to log file
        use_callback (bool): Whether to use lazy callback
        callback_func (function): Callback function for lazy constraints
    
    Returns:
        mod (gurobipy.Model): Solved model
    """
    mod.params.NonConvex = 2
    mod.setParam("VarBranch", 1)
    mod.setParam("PreDual", 1)
    mod.setParam("Heuristics", 0.01)
    mod.setParam("Cuts", 1)
    mod.setParam("PreQLinearize", 0)
    mod.setParam("PrePasses", 1)
    mod.setParam("ScaleFlag", 2)
    mod.setParam('TimeLimit', time_limit)
    mod.setParam('OutputFlag', 1)
    mod.setParam("LogFile", logfile)
    mod.write("model.lp")
    
    if use_callback and callback_func:
        mod.Params.LazyConstraints = 1
        mod.optimize(callback_func)
    else:
        mod.optimize()
    
    return mod


# ============================================================================
# 11. LOAD DATA
# ============================================================================

def load_data(data_file):
    """
    Loads data from input file and auto-generates incidence matrix.
    
    Inputs:
        data_file (str): Path to data file (first line: m n, then m lines of values)
    
    Returns:
        m (int): Number of rows
        n (int): Number of columns
        c (dict): Dictionary mapping (i,j) coordinates to data values
        VT (float): Total variance normalization factor
        incident (list): Incidence samples (auto-generated from grid)
    """
    with open(data_file, "r") as fh:
        l = 0
        c = {}
        for linea in fh:
            l_list = linea.split(" ")
            if l == 0:
                m = int(l_list[0])
                n = int(l_list[1])
            else:
                for j in range(0, n):
                    if j < len(l_list):
                        c[l - 1, j] = float(l_list[j])
            if l == m:
                break
            l = l + 1
    
    A, L, S = dj.leerInc(None, m, n)
    print(f"  Incidence matrix auto-generated for {m}x{n} grid ({L} edges, {S} vertices)")
    
    incident = dj.get_samples(A)
    VT = dj.calcVT(c, m, n)
    
    return m, n, c, VT, incident


def load_data_graph(data_file, model_type, add_sink=True):
    """
    Loads a planar graph from a data file.
    
    File format:
        <n> <m>              # Number of vertices, number of edges
        <vertex_id> <value>  # Vertex sample value (n lines)
        <u> <v>              # Edge (m lines)
    
    Inputs:
        data_file (str): Path to the data file
        model_type (str): 'lazy' or 'flow'
        add_sink (bool): Whether to add a sink node (for flow model)
    
    Returns:
        G (networkx.Graph): Undirected graph
        n (int): Number of vertices
        m (int): Number of edges
    """
    with open(data_file, "r") as fh:
        lines = [line.strip() for line in fh if line.strip() and not line.startswith('#')]
    
    if len(lines) < 2:
        raise ValueError(f"File {data_file} has insufficient data")
   
    parts = lines[0].split()
    n = int(parts[0])
    m = int(parts[1])
    
    G = nx.Graph()
    

    line_idx = 1
    for i in range(n):
        if line_idx >= len(lines):
            raise ValueError(f"Expected {n} vertices, got {line_idx - 1}")
        
        parts = lines[line_idx].split()
        if len(parts) < 2:
            raise ValueError(f"Invalid vertex line: {lines[line_idx]}")
        
        v_id = int(parts[0])
        value = float(parts[1])
        G.add_node(v_id, sample=value)
        line_idx += 1

    for i in range(m):
        if line_idx >= len(lines):
            raise ValueError(f"Expected {m} edges, got {i}")
        
        parts = lines[line_idx].split()
        if len(parts) < 2:
            raise ValueError(f"Invalid edge line: {lines[line_idx]}")
        
        u = int(parts[0])
        v = int(parts[1])
  
        cost = round(abs(G.nodes[u]['sample'] - G.nodes[v]['sample']), 4)
        G.add_edge(u, v, costo=cost)
        line_idx += 1
    

    if not nx.is_connected(G):
        print(f"  WARNING: Graph is not connected! ({nx.number_connected_components(G)} components)")
    

    if add_sink:
        sink_id = max(G.nodes()) + 1
        G.add_node(sink_id, sample=0.0)
        for u in G.nodes():
            if u != sink_id:
                G.add_edge(u, sink_id, costo=0.0)
    
    print(f"  Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Connected: {nx.is_connected(G)}")
    
    return G, n, m


# ============================================================================
# 12. MAIN FUNCTION
# ============================================================================

def print_usage():
    """
    Prints program usage instructions.
    
    Inputs:
        None
    
    Returns:
        None (prints to console)
    """
    print("""
PROGRAM USAGE
================================================================================

1. Run HEURISTIC only:
   python ssmz.py grid <data_file> <alpha> h1 [draw] [output] [plot]
   python ssmz.py grid <data_file> <alpha> h2 [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> h1 [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> h2 [draw] [output] [plot]

2. Run LAZY model (no heuristic):
   python ssmz.py grid <data_file> <alpha> lazy [time] [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> lazy [time] [draw] [output] [plot]

3. Run LAZY model with heuristic warm start:
   python ssmz.py grid <data_file> <alpha> lazy h1 [time] [draw] [output] [plot]
   python ssmz.py grid <data_file> <alpha> lazy h2 [time] [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> lazy h1 [time] [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> lazy h2 [time] [draw] [output] [plot]

4. Run FLOW model (no heuristic):
   python ssmz.py grid <data_file> <alpha> flow [time] [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> flow [time] [draw] [output] [plot]

5. Run FLOW model with heuristic warm start:
   python ssmz.py grid <data_file> <alpha> flow h1 [time] [draw] [output] [plot]
   python ssmz.py grid <data_file> <alpha> flow h2 [time] [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> flow h1 [time] [draw] [output] [plot]
   python ssmz.py graph <data_file> <alpha> flow h2 [time] [draw] [output] [plot]

PARAMETERS:
  graph_type      : grid | graph
  data_file       : File with data
  alpha           : Alpha parameter (float)
  model           : lazy | flow | h1 | h2
  h1|h2 (optional): Heuristic to use for warm start
  time (optional) : Time limit in seconds (default: 1800)
  draw (optional) : Generate visualizations
  output (optional): Generate detailed report file
  plot (optional) : Generate convergence plots from Gurobi logs

NOTES:
  - 'grid' mode: Uses original grid-based incidence matrix
  - 'graph' mode: Loads planar graph from file
  - The incidence matrix is auto-generated from the grid dimensions for 'grid' mode

EXAMPLES:
  # Grid mode - Heuristic only with draw and output
  python ssmz.py grid datos.txt 0.5 h1 draw output
  
  # Grid mode - Lazy model without heuristic (time limit 3600s)
  python ssmz.py grid datos.txt 0.5 lazy 3600
  
  # Graph mode - Flow model with warm start (h1, time limit 3600s)
  python ssmz.py graph instance.txt 0.5 flow h1 3600 draw output
  
  # Graph mode - Lazy with warm start and convergence plot
  python ssmz.py graph instance.txt 0.5 lazy h2 1800 draw plot
================================================================================
""")


def main():
    """
    Main function - entry point of the program.
    
    Inputs:
        Command line arguments (see print_usage)
    
    Returns:
        None (executes the program)
    """
    if len(sys.argv) < 4:
        print_usage()
        sys.exit(1)
    
    graph_type = sys.argv[1].lower()
    if graph_type not in ['grid', 'graph']:
        print(f"ERROR: Unknown graph type '{graph_type}'. Use 'grid' or 'graph'.")
        print_usage()
        sys.exit(1)
    
    data_file = sys.argv[2]
    alpha = float(sys.argv[3])
    
    if len(sys.argv) >= 5:
        mode = sys.argv[4].lower()
    else:
        print_usage()
        sys.exit(1)
    
    print(f"\nLoading data...")
    print(f"  Graph type: {graph_type.upper()}")
    print(f"  Data: {data_file}")
    print(f"  Alpha: {alpha}")
    
    draw_flag = False
    output_flag = False
    plot_flag = False
    time_limit = 1800  
    heuristic_name = None
    

    idx = 5
    while idx < len(sys.argv):
        arg = sys.argv[idx].lower()
        if arg == "draw":
            draw_flag = True
        elif arg == "output":
            output_flag = True
        elif arg == "plot":
            plot_flag = True
        elif arg in ["h1", "h2"]:
            heuristic_name = arg
        elif arg.isdigit():
            time_limit = int(arg)
        idx += 1
    
    # ========================================================================
    # GRID MODE (Original functionality)
    # ========================================================================
    if graph_type == 'grid':
        m, n, c, VT, incident = load_data(data_file)
        print(f"  Grid: {m} x {n} = {m * n} nodes")
        print(f"  VT: {VT:.6f}")
        

        if mode in ["h1", "h2"]:
            run_heuristic_only(m, n, c, VT, alpha, mode, data_file, draw_flag, output_flag, plot_flag, graph_type='grid')
            return

        if mode not in ["lazy", "flow"]:
            print(f"\nERROR: Unknown mode '{mode}'")
            print_usage()
            sys.exit(1)
        
        run_model(mode, heuristic_name, time_limit, m, n, c, VT, alpha,
                  data_file, draw_flag, output_flag, plot_flag, graph_type='grid')
    
    # ========================================================================
    # GRAPH MODE (New functionality for planar graphs)
    # ========================================================================
    else:
        G, n_graph, m_graph = load_data_graph(data_file, 'lazy' if mode in ['h1', 'h2', 'lazy'] else 'flow', 
                                               add_sink=(mode == 'flow'))
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if mode == "flow":
            sink_id = max(G.nodes())
            samples = [G.nodes[u]['sample'] for u in G.nodes() if u!=sink_id]  
        else:
            samples = [G.nodes[u]['sample'] for u in G.nodes()]
        mean = np.mean(samples)
        print(f"  Mean: {mean:.6f}")
        print(f"  n_graph: {n_graph}")
        if mode == "flow":
            VT = np.sum([(s - mean) ** 2 for s in samples])/(n_graph)
        else:
            VT = np.sum([(s - mean) ** 2 for s in samples])/n_graph
        print(f"  VT: {VT:.6f}")
        if mode in ["h1", "h2"]:
            run_heuristic_only_graph(G, VT, alpha, mode, data_file, draw_flag, output_flag, plot_flag)
            return
        
        if mode not in ["lazy", "flow"]:
            print(f"\nERROR: Unknown mode '{mode}'")
            print_usage()
            sys.exit(1)
        
        m, n = 0, 0
        c = {}
        
        run_model(mode, heuristic_name, time_limit, m, n, c, VT, alpha,
                  data_file, draw_flag, output_flag, plot_flag, graph_type='graph')
    
    print("\n" + "=" * 60)
    print("EXECUTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()