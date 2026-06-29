# Contributing to SSMZ-MSF

Thanks for your interest in improving SSMZ-MSF.

## Reporting issues
Please open a GitHub issue using the bug-report template, including the exact
command line, the input instance (if not one of the bundled examples), and your
environment (OS, Python, Gurobi, networkx/numpy versions).

## Development setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Before submitting a pull request
- Keep the code style consistent with the surrounding code.
- Make sure the license-free smoke test passes (it does not require Gurobi):
  ```bash
  bash reproduce_example.sh
  python ssmz_msf.py graph examples/graph_irregular.txt 0.7 h2
  ```
- The continuous-integration workflow runs the same checks on every push and
  pull request.

## License
By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE) (software) and CC BY 4.0 (documentation).
