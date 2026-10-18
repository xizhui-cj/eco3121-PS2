# Collaborative assignment workflow

Scope currently implemented: Question 1 only. The assistant maintains Python,
generated numerical outputs, and LaTeX mechanics. The student writes the
interpretation in `sections/q1.tex`. Do not edit generated files in `results/`;
rerun Python when a numerical change is needed.

## Local reproduction

Use Python 3.12 and an isolated virtual environment:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/q1.py
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

In this cloud instance the prepared interpreter is
`/workspace/.venvs/eco3121-ps2/bin/python`.

The script checks all 200 observations, agreement between the uploaded CSV and
Stata files, full rank, and independent coefficient/standard-error calculations.
It includes an intercept and uses nonrobust OLS standard errors as requested.

## GitHub and Overleaf

GitHub repository: `xizhui-cj/eco3121-PS2`.
Overleaf project: https://www.overleaf.com/project/6ad45f6e9bc2ef659d583a88
The Overleaf connection and synchronization have not been verified yet.

If your Overleaf account provides GitHub synchronization, import the GitHub
repository into Overleaf and choose `main.tex` as the main document. The exact
menu and availability depend on the account. Overleaf synchronization is a
separate action; a GitHub push alone does not prove the Overleaf copy updated.
After each GitHub update, pull it into Overleaf using its GitHub synchronization
control and recompile. Before pulling, send any Overleaf edits back to GitHub
and reconcile conflicts to preserve your writing.

If GitHub synchronization is unavailable, upload a project ZIP to Overleaf as
an initial import. That provides a compiled document but does not create live
GitHub synchronization. Never place an Overleaf token in the repository/chat.

## Current checkpoint

- Q1 coding and numerical table: generated and locally checked.
- Q1 interpretation: waiting for student.
- Questions 2--9: not started.
- Overleaf linking: pending account-side setup and verification.
