"""Questions 1 and 3. Run from any directory; output paths are project-relative."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'data' / 'PS2_data.csv'
df = pd.read_csv(source)
cols = ['education', 'ability', 'wage']
if len(df) != 200 or not np.isfinite(df[cols].to_numpy()).all():
    raise ValueError('Expected 200 complete, finite observations.')
stata = pd.read_stata(ROOT / 'data' / 'PS2_data.dta')
np.testing.assert_allclose(df[cols], stata[cols], rtol=0, atol=1e-12)
X = sm.add_constant(df[['education', 'ability']], has_constant='add')
if np.linalg.matrix_rank(X) != 3:
    raise ValueError('Design matrix must have full column rank.')
model = sm.OLS(df['wage'], X, missing='raise').fit(cov_type='nonrobust')
# Independently verify coefficients and conventional standard errors.
b, *_ = np.linalg.lstsq(X.to_numpy(), df['wage'].to_numpy(), rcond=None)
residual = df['wage'].to_numpy() - X.to_numpy() @ b
cov = (residual @ residual / (len(df) - X.shape[1])) * np.linalg.inv(X.T @ X)
np.testing.assert_allclose(model.params, b, rtol=1e-10, atol=1e-10)
np.testing.assert_allclose(model.bse, np.sqrt(np.diag(cov)), rtol=1e-10)
out = ROOT / 'results'
out.mkdir(exist_ok=True)
pd.DataFrame({'coefficient': model.params, 'std_error': model.bse}).to_csv(out / 'q1_coefficients.csv', index_label='term')
metadata = {'question': 1, 'n': int(model.nobs), 'intercept': True,
            'standard_errors': 'usual OLS (nonrobust)',
            'education_coefficient': float(model.params['education']),
            'education_std_error': float(model.bse['education']),
            'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
(out / 'q1_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
table = '\n'.join([
    r'\begin{center}', r'\begin{tabular}{lr}', r'\hline',
    r'Quantity & Estimate \\', r'\hline',
    rf'Education coefficient ($\hat{{\beta}}_1$) & {model.params["education"]:.3f} \\',
    rf'Usual OLS standard error & {model.bse["education"]:.3f} \\',
    rf'Observations & {int(model.nobs)} \\',
    r'\hline', r'\end{tabular}', r'\end{center}', ''])
(out / 'q1_table.tex').write_text(table)
equation = '\n'.join([
    r'\[',
    rf'\widehat{{\mathrm{{wage}}}}_i = {model.params["const"]:.3f}'
    rf' + {model.params["education"]:.3f}\,\mathrm{{education}}_i'
    rf' + {model.params["ability"]:.3f}\,\mathrm{{ability}}_i.',
    r'\]', ''])
(out / 'q1_equation.tex').write_text(equation)
# Question 3 uses this same levels regression, not the log-wage model.
ssr = float(model.resid @ model.resid)
sst = float(((df['wage'] - df['wage'].mean()) ** 2).sum())
r_squared = 1 - ssr / sst
ser = float(np.sqrt(ssr / model.df_resid))
np.testing.assert_allclose([r_squared, ssr, ser],
                           [model.rsquared, model.ssr, np.sqrt(model.mse_resid)],
                           rtol=1e-10)
q3 = {'question': 3, 'source_model': 'Q1: wage on education and ability with intercept',
      'n': int(model.nobs), 'residual_df': int(model.df_resid),
      'r_squared': r_squared, 'residual_sum_of_squares': ssr,
      'regression_standard_error': ser}
(out / 'q3_results.json').write_text(json.dumps(q3, indent=2) + '\n')
q3_table = '\n'.join([
    r'\begin{center}', r'\begin{tabular}{lr}', r'\hline',
    r'Quantity & Estimate \\', r'\hline',
    rf'$R^2$ & {r_squared:.3f} \\',
    rf'Residual sum of squares (SSR) & {ssr:.3f} \\',
    rf'Standard error of the regression & {ser:.3f} \\',
    r'\hline', r'\end{tabular}', r'\end{center}', ''])
(out / 'q3_table.tex').write_text(q3_table)
print(json.dumps(q3, indent=2))
print(json.dumps(metadata, indent=2))
print('Verified CSV/Stata agreement and independent OLS calculations.')
