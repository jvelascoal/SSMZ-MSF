#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reproduce the LICENSE-FREE results of the SSMZ-MSF Software Impacts article:
#   - Table 2: heuristics H1/H2 over alpha in {0.5, 0.7, 0.9}
#   - Fig. 1 : H2 zone maps for the same thresholds
# Uses only networkx/numpy (no Gurobi license required).
# ---------------------------------------------------------------------------
set -euo pipefail

# Locate the example instance: Code Ocean mounts it under /data, otherwise use examples/.
if [ -f /data/field_6x7.txt ]; then
    DATA=/data/field_6x7.txt
else
    DATA="$(dirname "$0")/examples/field_6x7.txt"
fi

PY=${PYTHON:-python}
SCRIPT="$(dirname "$0")/ssmz_msf.py"

# Render TikZ maps only if a LaTeX engine is available.
DRAW=""
if command -v pdflatex >/dev/null 2>&1; then DRAW="draw"; fi

echo "============================================================"
echo " SSMZ-MSF reproducible example (license-free heuristics)"
echo " Instance: $DATA"
echo "============================================================"
printf "\n%-6s | %-18s | %-18s\n" "alpha" "H1 (theta, H)" "H2 (theta, H)"
echo "-------+--------------------+--------------------"

for a in 0.5 0.7 0.9; do
    line=""
    for h in h1 h2; do
        out=$("$PY" "$SCRIPT" grid "$DATA" "$a" "$h" $DRAW output 2>/dev/null || true)
        th=$(printf '%s\n' "$out" | grep "Theta:"   | grep -oE '[0-9]+' | head -1)
        H=$(printf  '%s\n' "$out" | grep "H Value:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
        printf -v cell "(%s, %.3f)" "${th:-NA}" "${H:-0}"
        line="$line| $(printf '%-18s' "$cell") "
    done
    printf "%-6s %s\n" "$a" "$line"
done

echo
echo "Reports and (if LaTeX is available) zone maps were written to ./resultados/"
echo "Done."
