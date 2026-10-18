"""Question 2: natural-log wage on education and ability, with an intercept."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'data' / 'PS2_data.csv'
df = pd.read_csv(source)
cols = ['education', 'ability', 'wage']
if len(df) != 200 or not np.isfinite(df[cols].to_numpy()).all():
    raise ValueError('Expected 200 complete, finite observations.')
if not (df['wage'] > 0).all():
    raise ValueError('Natural log requires strictly positive wages.')
stata = pd.read_stata(ROOT / 'data' / 'PS2_data.dta')
np.testing.assert_allclose(df[cols], stata[cols], rtol=0, atol=1e-12)
y = np.log(df['wage'])
X = sm.add_constant(df[['education', 'ability']], has_constant='add')
if np.linalg.matrix_rank(X) != 3:
    raise ValueError('Design matrix must have full column rank.')
model = sm.OLS(y, X, missing='raise').fit(cov_type='nonrobust')
b, *_ = np.linalg.lstsq(X.to_numpy(), y.to_numpy(), rcond=None)
residual = y.to_numpy() - X.to_numpy() @ b
cov = (residual @ residual / (len(df) - X.shape[1])) * np.linalg.inv(X.T @ X)
np.testing.assert_allclose(model.params, b, rtol=1e-10, atol=1e-10)
np.testing.assert_allclose(model.bse, np.sqrt(np.diag(cov)), rtol=1e-10)
out = ROOT / 'results'
out.mkdir(exist_ok=True)
pd.DataFrame({'coefficient': model.params, 'std_error': model.bse}).to_csv(
    out / 'q2_coefficients.csv', index_label='term')
metadata = {'question': 2, 'n': int(model.nobs), 'dependent_variable': 'ln(wage)',
            'intercept': True, 'standard_errors': 'usual OLS (nonrobust)',
            'education_coefficient': float(model.params['education']),
            'education_std_error': float(model.bse['education']),
            'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
(out / 'q2_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
equation = '\n'.join([
    r'\[',
    rf'\widehat{{\ln(\mathrm{{wage}}_i)}} = {model.params["const"]:.3f}'
    rf' + {model.params["education"]:.3f}\,\mathrm{{education}}_i'
    rf' + {model.params["ability"]:.3f}\,\mathrm{{ability}}_i.',
    r'\]', ''])
(out / 'q2_equation.tex').write_text(equation)
table = '\n'.join([
    r'\begin{center}', r'\begin{tabular}{lr}', r'\hline',
    r'Quantity & Estimate \\', r'\hline',
    rf'Education coefficient ($\hat{{\beta}}_1$) & {model.params["education"]:.3f} \\',
    rf'Usual OLS standard error & {model.bse["education"]:.3f} \\',
    rf'Observations & {int(model.nobs)} \\',
    r'\hline', r'\end{tabular}', r'\end{center}', ''])
(out / 'q2_table.tex').write_text(table)
print(json.dumps(metadata, indent=2))
print('Verified positive wages, CSV/Stata agreement, and independent OLS calculations.')
