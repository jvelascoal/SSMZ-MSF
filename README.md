# SSMZ-MSF

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.8-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.dajour.2026.100707-blue.svg)](https://doi.org/10.1016/j.dajour.2026.100707)

**An open-source solver for optimal site-specific management zone delineation via minimum spanning forests.**

SSMZ-MSF partitions an agricultural field, sampled on a rectangular grid, into the
**minimum number of spatially contiguous management zones** such that each zone meets a
user-defined within-zone homogeneity threshold `α`. The problem is modeled as a
graph-partitioning problem in which every zone induces a tree, so the whole partition is a
*minimum spanning forest* of the field.

The solver provides two exact mixed-integer formulations, two fast spanning-tree
heuristics, and hybrid strategies that warm-start the exact models with a heuristic
solution, together with reporting and publication-quality map generation.

This repository accompanies the *Software Impacts* article and implements the framework
introduced and validated in Urbán-Rivero & Velasco, *Decision Analytics Journal* (2026);
see [Citation](#citation).

---

## Features

- **Exact models** (solved with Gurobi):
  - `lazy` — subtour (cycle) elimination via lazy constraints generated on demand.
  - `flow` — acyclicity enforced a priori by Miller–Tucker–Zemlin-type flow constraints.
- **Heuristics** (operate on a minimum spanning tree, parameter-free):
  - `h1` — greedy: removes the highest-cost edge at each step.
  - `h2` — best-improvement: removes the edge that maximizes homogeneity.
- **Hybrid mode** — use `h1`/`h2` to warm-start the `lazy` or `flow` model.
- **Two input modes** — `grid` (rectangular lattice) and `graph` (general planar
  adjacency graph), with masking of no-data cells for irregular field shapes.
- **Outputs** — detailed text reports, TikZ/Graphviz zone maps, and Gurobi convergence plots.
- **Deterministic** results for a fixed configuration.

---

## Requirements

- **Python ≥ 3.8**
- **Gurobi** with a valid license (academic licenses are free). Install via
  `gurobipy`; see https://www.gurobi.com.
- Python packages: `gurobipy`, `networkx`, `numpy` (see `requirements.txt`).
- *Optional:* `matplotlib`, `plotly`, `grblogtools` (convergence plots); a LaTeX
  distribution providing `pdflatex` (TikZ zone maps in `grid` mode, `draw` flag);
  and `pygraphviz` + Graphviz (node-link layouts in `graph` mode, `draw` flag).

## Installation

```bash
# (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Verify that Gurobi is licensed:

```bash
python -c "import gurobipy as g; m=g.Model(); print('Gurobi OK', g.gurobi.version())"
```

---

## Usage

```bash
python ssmz_msf.py grid|graph <data_file> <alpha> <mode> [h1|h2] [time] [draw] [output] [plot]
```

| Argument     | Description                                                               |
|--------------|--------------------------------------------------------------------------|
| `grid` \| `graph` | Input type: `grid` for a rectangular lattice, `graph` for a general planar graph (see [Input format](#input-format)). |
| `data_file`  | Path to the data file.                                                    |
| `alpha`      | Homogeneity threshold `α ∈ (0,1)` (e.g. `0.5`, `0.7`, `0.9`).             |
| `mode`       | `h1` \| `h2` \| `lazy` \| `flow`.                                         |
| `h1`/`h2`    | *(optional)* heuristic warm start, only with `lazy`/`flow`.              |
| `time`       | *(optional)* solver time limit in seconds (default `1800`).              |
| `draw`       | *(optional)* generate zone maps (TikZ in `grid` mode; Graphviz in `graph` mode). |
| `output`     | *(optional)* write a detailed text report.                               |
| `plot`       | *(optional)* generate Gurobi convergence plots (PDF/PNG/HTML).           |

All results are written to a `resultados/` directory created in the working folder.

### Examples

```bash
# Grid, heuristic H2 only (sub-second), with maps and a report
python ssmz_msf.py grid examples/field_6x7.txt 0.5 h2 draw output

# Grid, lazy exact model, warm-started with H2, 300 s budget, full output
python ssmz_msf.py grid examples/field_6x7.txt 0.5 lazy h2 300 draw output plot

# Grid, flow exact model without warm start, 600 s budget
python ssmz_msf.py grid examples/field_6x7.txt 0.7 flow 600

# Graph mode on a general planar instance, heuristic H2
python ssmz_msf.py graph examples/graph_irregular.txt 0.7 h2 draw output
```

Running `python ssmz_msf.py` without arguments prints the full usage help.

---

### `grid` mode

A plain-text file whose **first line** gives the grid dimensions `m n` (rows, columns),
followed by `m` lines of `n` space-separated numeric values (one per grid cell):

```
6 7
16 14.5 14.9 12.2 12.3 16.7 9.7
16.5 17 16.1 10 13.4 17.4 16.8
13 18.4 17.5 10.1 18.2 12.9 14.4
16.6 10.1 15 13.9 12.3 15.3 15.5
12 12.1 17.2 17.1 15 18.6 17.9
11.7 15.9 18.5 11.8 14.5 10.6 18.7
```

The grid adjacency (incidence) structure is generated automatically; no separate
incidence file is required. Example: [`examples/field_6x7.txt`](examples/field_6x7.txt).

**Irregular fields (no-data cells).** Cells with a non-positive value (e.g. `0`)
are treated as no-data: they are removed from the graph and excluded from the
homogeneity computation, so the remaining cells form an irregular field. For
example, [`examples/field_6x7_masked.txt`](examples/field_6x7_masked.txt) masks
five cells of the 6×7 field (the grid equivalent of
[`examples/graph_irregular.txt`](examples/graph_irregular.txt)):

```
6 7
10.7 15.4 13.9 16.1 15.6 16.6 16.6
11.7 16.0 13.8 12.6 14.4 15.4 11.2
15.1 12.6 0 16.8 10.5 18.7 10.4
16.3 12.7 0 11.4 11.5 16.7 13.5
14.1 11.1 14.5 15.0 14.13 9.6 12.5
11.8 12.8 14.9 14.0 0 0 0
```

```bash
python ssmz_msf.py grid examples/field_6x7_masked.txt 0.7 h2 draw output
```

### `graph` mode

A plain-text file describing a general planar adjacency graph. The **first line**
gives `n m` (number of vertices, number of edges), followed by `n` lines
`vertex_id value` and then `m` lines `u v` listing the edges:

```
37 58
0 10.7
1 15.4
...
0 1
0 8
...
```

Edge costs are derived automatically as the absolute difference of the incident
vertex values. Example: [`examples/graph_irregular.txt`](examples/graph_irregular.txt)
(an irregular field); [`examples/graph_6x7.txt`](examples/graph_6x7.txt) encodes the
same field as `field_6x7.txt` in graph form and reproduces the grid-mode result.

## Output

For each run, `resultados/` may contain:

- `*_report.txt` / `*.txt` — number of zones `θ`, homogeneity `H`, optimality gap, runtime.
- `*_rec.pdf` — zone map (colored cells labeled by zone index).
- `*_graph.pdf` — zone map overlaid with the spanning-forest edges.
- `*_log.log` — Gurobi solver log.
- `*_convergence.{pdf,png,html}` — convergence plots (with `plot`).

---

## Reproducing the illustrative example

The 6×7 field in `examples/field_6x7.txt` (benchmark Class 1 from Cid-Garcia et al., 2013)
reproduces the *Software Impacts* article example. For `α = 0.5`, heuristic `h2` returns
**6 contiguous zones** with homogeneity **H ≈ 0.566** in well under a second:

```bash
python ssmz_msf.py grid examples/field_6x7.txt 0.5 h2 draw output
```

Repeating with `α = 0.7` and `α = 0.9` yields progressively finer partitions
(9 and 19 zones, respectively).

---

## License

- **Software:** [MIT License](LICENSE).
- **Documentation, example data, and figures:** Creative Commons Attribution 4.0
  International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## Citation

If you use SSMZ-MSF in your research, please cite the validating study:

> L. E. Urbán-Rivero and J. Velasco. *A hybrid analytics-driven optimization framework
> for site-specific zone management in precision agriculture.* Decision Analytics Journal,
> 19 (2026) 100707. https://doi.org/10.1016/j.dajour.2026.100707

The underlying greedy spanning-tree heuristic was introduced in:

> L. E. Urbán Rivero, J. Velasco, and J. Ramírez Rodríguez. *A simple greedy heuristic for
> the site-specific management zone problem.* Axioms, 11(7) (2022) 318.
> https://doi.org/10.3390/axioms11070318

## Authors

- **Luis E. Urbán-Rivero** — Instituto Tecnológico Autónomo de México (ITAM), Mexico City, Mexico. <luis.urban@itam.mx>
- **Jonás Velasco** — SECIHTI / Centro de Investigación en Matemáticas (CIMAT),
  Aguascalientes, Mexico. <jvelasco@cimat.mx>

## Support

For questions, please open an issue or contact <jvelasco@cimat.mx>.
