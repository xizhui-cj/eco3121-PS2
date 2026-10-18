"""Question 6: compare R-squared and explained sums of squares on one sample."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'PS2_data.csv')
if len(df) != 200 or not np.isfinite(df[['wage', 'education', 'ability']]).all().all():
    raise ValueError('Expected 200 complete, finite observations.')
sst = float(((df['wage'] - df['wage'].mean()) ** 2).sum())
rows = []
for label, columns in [('Q5: education only', ['education']),
                       ('Q1: education and ability', ['education', 'ability'])]:
    fit = sm.OLS(df['wage'], sm.add_constant(df[columns], has_constant='add'),
                 missing='raise').fit(cov_type='nonrobust')
    explained = float(((fit.fittedvalues - df['wage'].mean()) ** 2).sum())
    residual = float((fit.resid ** 2).sum())
    np.testing.assert_allclose(explained + residual, sst, rtol=1e-10)
    np.testing.assert_allclose([explained, explained / sst],
                               [fit.ess, fit.rsquared], rtol=1e-10)
    rows.append({'model': label, 'n': int(fit.nobs), 'r_squared': float(fit.rsquared),
                 'explained_sum_of_squares_SSE': explained,
                 'residual_sum_of_squares_SSR': residual})
if rows[1]['explained_sum_of_squares_SSE'] < rows[0]['explained_sum_of_squares_SSE'] - 1e-8:
    raise AssertionError('Adding a regressor decreased explained sum of squares.')
out = ROOT / 'results'
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'q6_comparison.csv', index=False)
(out / 'q6_results.json').write_text(json.dumps({'question': 6, 'total_sum_of_squares': sst,
                                               'models': rows}, indent=2) + '\n')
table = [r'\begin{center}', r'\begin{tabular}{lrr}', r'\hline',
         r'Model & $R^2$ & Explained sum of squares (SSE) \\', r'\hline']
for row in rows:
    table.append(rf'{row["model"]} & {row["r_squared"]:.3f} & {row["explained_sum_of_squares_SSE"]:.3f} \\')
table.extend([r'\hline', r'\end{tabular}', r'\end{center}', ''])
(out / 'q6_table.tex').write_text('\n'.join(table))
print(json.dumps(rows, indent=2))
print('Verified common sample and total = explained + residual sums of squares.')
