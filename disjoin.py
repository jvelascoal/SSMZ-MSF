"""
Disjoint set utilities for graph segmentation problems.
Includes incidence matrix generation for grid graphs.
"""

from copy import deepcopy
import random
import os


def findSet(p, i):
    """
    Find with path compression.
    
    Inputs:
        p (list): Parent array
        i (int): Element to find
    
    Returns:
        int: Root of the set containing i
    """
    if p[i] == i:
        return i
    else:
        result = findSet(p, p[i])
        p[i] = result
        return result


def unionSet(p, rank, u, v):
    """
    Union by rank.
    
    Inputs:
        p (list): Parent array
        rank (list): Rank array
        u (int): First element
        v (int): Second element
    """
    x = findSet(p, u)
    y = findSet(p, v)
    if rank[x] > rank[y]:
        p[y] = x
    elif rank[x] < rank[y]:
        p[x] = y
    else:
        p[x] = y
        rank[y] += 1


def get_index_p(subs, elem):
    """
    Find index of element in dictionary.
    
    Inputs:
        subs (dict): Dictionary mapping parent to list of children
        elem (int): Element to find
    
    Returns:
        int: Index if found, -1 otherwise
    """
    for e in subs:
        if elem == e:
            return e
            break
    return -1


def get_subsets(p):
    """
    Get subsets from parent array.
    
    Inputs:
        p (list): Parent array
    
    Returns:
        dict: Dictionary mapping root to list of elements
    """
    vs = {}
    PS = len(p)
    for k in range(0, PS):
        parent = findSet(p, k)
        idx = get_index_p(vs, parent)
        if idx < 0:
            vs[parent] = [k]
        else:
            vs[idx].append(k)
    return vs


def get_samples(A):
    """
    Extract samples from incidence matrix.
    
    Inputs:
        A (list): Incidence matrix (list of lists)
    
    Returns:
        list: List of pairs of connected vertices
    """
    SA = len(A)
    aux = []
    for i in range(0, SA):
        v = []
        for j in range(0, len(A[i])):
            if A[i][j] == 1:
                v.append(j)
            if len(v) >= 2:
                break
        aux.append(v)
    return aux


def calcVT(datos, m, n):
    """
    Calculate total variance (VT) from data.
    
    Inputs:
        datos (dict): Dictionary mapping (i,j) to value
        m (int): Number of rows
        n (int): Number of columns
    
    Returns:
        float: Total variance
    """
    total_samples =sum(1 for i in range(0, m) for j in range(0, n) if datos[i,j]>0)
    s_mt = sum(datos[i, j] for i in range(0, m) for j in range(0, n) if datos[i,j]>0) / total_samples
    s_vt = sum((datos[i, j] - s_mt) ** 2 for i in range(0, m) for j in range(0, n) if datos[i,j]>0) / total_samples
    return s_vt


def generate_incidence_matrix(m, n):
    """
    Generates the incidence matrix for an m x n grid graph.
    Each row represents an edge between two adjacent cells.
    
    Inputs:
        m (int): Number of rows in the grid
        n (int): Number of columns in the grid
    
    Returns:
        A (list): Incidence matrix (list of lists)
        L (int): Number of edges (rows in matrix)
        S (int): Number of vertices (columns in matrix)
    """
    total_cells = m * n
    
    # Calculate total number of edges (adjacent pairs)
    # Horizontal edges: (n-1) per row * m rows
    # Vertical edges: (m-1) per column * n columns
    total_edges = (n - 1) * m + (m - 1) * n
    
    incidence = []
    
    # Horizontal edges: between cells in the same row
    for i in range(m):
        for j in range(n - 1):
            row = [0] * total_cells
            row[i * n + j] = 1          # Current cell
            row[i * n + j + 1] = 1      # Right neighbor
            incidence.append(row)
    
    # Vertical edges: between cells in the same column
    for i in range(m - 1):
        for j in range(n):
            row = [0] * total_cells
            row[i * n + j] = 1          # Current cell
            row[(i + 1) * n + j] = 1    # Bottom neighbor
            incidence.append(row)
    
    return incidence, total_edges, total_cells


def leerInc(nombre=None, m=None, n=None):
    """
    Reads incidence matrix from file or generates it if dimensions are provided.
    
    Inputs:
        nombre (str, optional): Path to incidence file
        m (int, optional): Number of rows (if generating)
        n (int, optional): Number of columns (if generating)
    
    Returns:
        A (list): Incidence matrix (list of lists)
        L (int): Number of edges (rows in matrix)
        S (int): Number of vertices (columns in matrix)
    
    Note:
        If nombre is provided, reads from file.
        If nombre is None and m,n are provided, generates matrix.
        If both are None, returns empty matrix.
    """
    # Generate incidence matrix if dimensions are provided
    if nombre is None and m is not None and n is not None:
        return generate_incidence_matrix(m, n)
    
    # Read from file if nombre is provided
    if nombre is not None:
        A = []
        i = 0
        with open(nombre, "r") as fh_matrix:
            for linea in fh_matrix:
                if i == 0:
                    s_linea = linea.strip()
                    s1 = s_linea.split(" ")
                    L = int(s1[0])
                    S = int(s1[1])
                    i = i + 1
                else:
                    A.append([])
                    s_linea = linea.strip()
                    s1 = s_linea.split(" ")
                    for j in range(0, len(s1)):
                        A[i - 1].append(int(s1[j]))
                    i = i + 1
        return [A, L, S]
    
    # If no arguments provided, return empty
    return [[], 0, 0]


def dibujo(nombre, sol, lineas, m, n, seed=456):
    """
    Generates LaTeX diagram of the solution with grid lines.
    
    Inputs:
        nombre (str): Output filename (without extension)
        sol (list): List of components (list of lists of vertices)
        lineas (list): Binary list indicating which edges are active
        m (int): Number of rows
        n (int): Number of columns
        seed (int): Random seed for colors
    
    Returns:
        None (creates .tex and .pdf files)
    """
    N = m * n
    random.seed(seed)
    rgbt = [((random.uniform(0, 1) + random.uniform(0, 1)) / 2.0,
             (random.uniform(0, 1) + random.uniform(0, 1) + random.uniform(0, 1)) / 3.0,
             (random.uniform(0, 1) + random.uniform(0, 1)) / 2.0) for x in range(N)]
    
    f = open(nombre + ".tex", 'w')
    f.write("\\documentclass{standalone} \n")
    f.write("\\usepackage[usenames,dvipsnames]{xcolor} \n")
    f.write("\\usepackage{tikz}\n")
    f.write("\\usetikzlibrary{patterns} \n")
    for k in range(0, len(rgbt)):
        f.write("\\definecolor{col" + str(k) + "}{rgb}{" + str(rgbt[k][0]) + "," + str(rgbt[k][1]) + "," + str(rgbt[k][2]) + "}\n")
    f.write("\\begin{document} \n")
    f.write("\\begin{tikzpicture} \n")
    
    for k in range(0, len(sol)):
        for p in sol[k]:
            i = int(p / n)
            j = p % n
            f.write("\\fill[fill=col" + str(k) + ",opacity=0.3](" + str(j) + "," + str(m - (i + 1)) + ") rectangle ( " + str(j + 1) + "," + str(m - i) + ");\n")
            f.write("\\node[font=\\fontsize{8}{8}] at (" + str(j + 0.5) + "," + str(m - i - 0.5) + "){$" + str(k + 1) + "$};\n")
            
            if i < m - 1:
                if j < n - 1:
                    k1 = 2 * i * (n - 1) + (i + j)
                    k2 = 2 * i * (n - 1) + ((n - 1) + i + j)
                    if lineas[k1] == 1:
                        f.write("\\draw[color=DarkOrchid, line width=0.55mm,opacity=0.9] (" + str(j + 1) + "," + str(m - i - 1) + ") -- ( " + str(j + 1) + "," + str(m - i) + ");\n")
                    if lineas[k2] == 1:
                        f.write("\\draw[color=DarkOrchid, line width=0.55mm,opacity=0.9] (" + str(j) + "," + str(m - (i + 1)) + ") -- ( " + str(j + 1) + "," + str(m - (i + 1)) + ");\n")
                else:
                    k2 = 2 * i * (n - 1) + ((n - 1) + i + j)
                    if lineas[k2] == 1:
                        f.write("\\draw[color=DarkOrchid, line width=0.55mm,opacity=0.9] (" + str(j) + "," + str(m - (i + 1)) + ") -- ( " + str(j + 1) + "," + str(m - (i + 1)) + ");\n")
            else:
                if j < n - 1:
                    k1 = 2 * i * (n - 1) + (i + j)
                    if lineas[k1] == 1:
                        f.write("\\draw[color=DarkOrchid, line width=0.55mm,opacity=0.9] (" + str(j + 1) + "," + str(m - i - 1) + ") -- ( " + str(j + 1) + "," + str(m - i) + ");\n")
    
    f.write("\\end{tikzpicture}\n")
    f.write("\\end{document}\n")
    f.close()
    os.system("pdflatex --interaction=batchmode " + nombre + ".tex")


class solucion:
    """
    Solution class for line-based segmentation.
    """
    def __init__(self, lineas, incidencias, m, n, s, rank):
        self.lineas = deepcopy(lineas)
        self.vars = []
        self.m = m
        self.n = n
        self.L = len(lineas)
        s_c = deepcopy(s)
        rank_c = deepcopy(rank)
        for k in range(0, self.L):
            kdx = incidencias[k]
            if self.lineas[k] == 0:
                unionSet(s_c, rank_c, kdx[0], kdx[1])
        aux = get_subsets(s_c)
        lista = []
        for pi in aux:
            lista.append(aux[pi])
        self.z = lista
        self.Z = len(self.z)

    def get_data(self, datos, k):
        i = int(k / self.n)
        j = k % self.n
        return datos[i, j]

    def up_vars(self, datos):
        lista = []
        for k in range(0, self.Z):
            suma_m = 0
            for p in self.z[k]:
                suma_m += self.get_data(datos, p)
            sm = suma_m / len(self.z[k])
            suma_v = 0.0
            for p in self.z[k]:
                suma_v += (self.get_data(datos, p) - sm) ** 2
            if len(self.z[k]) > 1:
                sv = suma_v / (len(self.z[k]) - 1.0)
            else:
                sv = 0
            lista.append(sv)
        self.vars = lista

    def calcH(self, VT):
        suma = 0
        for k in range(0, self.Z):
            suma += (len(self.z[k]) - 1) * self.vars[k]
        den = VT * ((self.m * self.n) - self.Z)
        if den == 0.0:
            return 1.0
        else:
            return 1.0 - ((suma) / den)
    
    def fobj(self, VT, alpha):
        H = self.calcH(VT)
        if H < alpha:
            return self.Z + 1000 - abs(H)
        else:
            return self.Z


def get_datos(datos, k, n):
    i = int(k / n)
    j = k % n
    return datos[i, j]


class sol_no_lineas:
    """
    Solution class for non-line-based segmentation.
    """
    def __init__(self, z, m, n):
        self.m = m
        self.n = n
        self.z = []
        for k in range(0, len(z)):
            lista = []
            if len(z[k]) > 0:
                for p in z[k]:
                    if p != m * n:
                        lista.append(p)
                self.z.append(lista)
        self.Z = len(self.z)
        self.vars = []
    
    def get_data(self, datos, k):
        i = int(k / self.n)
        j = k % self.n
        return datos[i, j]
    
    def up_vars(self, datos):
        lista = []
        for k in range(0, self.Z):
            suma_m = 0
            for p in self.z[k]:
                suma_m += self.get_data(datos, p)
            sm = suma_m / len(self.z[k])
            suma_v = 0.0
            for p in self.z[k]:
                suma_v += (self.get_data(datos, p) - sm) ** 2
            if len(self.z[k]) > 1:
                sv = suma_v / (len(self.z[k]) - 1.0)
            else:
                sv = 0
            lista.append(sv)
        self.vars = lista
    
    def calcH(self, VT):
        suma = 0
        for k in range(0, self.Z):
            suma += (len(self.z[k]) - 1) * self.vars[k]
        den = VT * ((self.m * self.n) - self.Z)
        if den == 0.0:
            return 1.0
        else:
            return 1.0 - ((suma) / den)
    
    def fobj(self, VT, alpha):
        H = self.calcH(VT)
        if H < alpha:
            return self.Z + 1000 - abs(H)
        else:
            return self.Z


class sol_graph:
    """
    Solution class for graph-based segmentation.
    """
    def __init__(self,G,z,type):
        self.G = G
        self.z = z
        self.Z = len(self.z)
        self.vars = []
        self.type = type
    def get_data(self, v):
        return self.G.nodes[v]['sample']
    def up_vars(self):
        lista = []
        for k in range(0, self.Z):
            suma_m = 0
            if len(self.z[k]) > 0:
                for v in self.z[k]:
                    suma_m += self.get_data(v)
                sm = suma_m / len(self.z[k])
                suma_v = 0.0
                for v in self.z[k]:
                    suma_v += (self.get_data(v) - sm) ** 2
                if len(self.z[k]) > 1:
                    sv = suma_v / (len(self.z[k]) - 1.0)
                else:
                    sv = 0
                lista.append(sv)
            else:
                lista.append(0)
        self.vars = lista
    def calcH(self, VT):
        suma = 0
        for k in range(0, self.Z):
            if len(self.z[k]) > 0:
                suma += (len(self.z[k]) - 1) * self.vars[k]
        if self.type=='flow':
            N=len(self.G.nodes())-1
        else:
            N=len(self.G.nodes())
        auxZ=sum(1 for k in range(0, self.Z) if len(self.z[k]) > 0)
        den = VT * (N - auxZ)
        if den == 0.0:
            return 1.0
        else:
            return 1.0 - ((suma) / den)
    
    def fobj(self, VT, alpha):
        H = self.calcH(VT)
        if H < alpha:
            return self.Z + 1000 - abs(H)
        else:
            return self.Z
        