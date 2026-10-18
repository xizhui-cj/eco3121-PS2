"""Question 7(a): auxiliary regression and omitted-variable identity."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'PS2_data.csv')
if len(df) != 200 or not np.isfinite(df[['wage', 'education', 'ability']]).all().all():
    raise ValueError('Expected 200 complete, finite observations.')
def ols(outcome, regressors):
    return sm.OLS(df[outcome], sm.add_constant(df[regressors], has_constant='add'),
                  missing='raise').fit(cov_type='nonrobust')
full = ols('wage', ['education', 'ability'])
short = ols('wage', ['education'])
aux = ols('ability', ['education'])
delta = float(aux.params['education'])
lhs = float(short.params['education'] - full.params['education'])
rhs = float(full.params['ability'] * delta)
np.testing.assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-10)
x = df['education'] - df['education'].mean()
z = df['ability'] - df['ability'].mean()
np.testing.assert_allclose(delta, (x @ z) / (x @ x), rtol=1e-10)
out = ROOT / 'results'
out.mkdir(exist_ok=True)
pd.DataFrame({'coefficient': aux.params, 'std_error': aux.bse}).to_csv(
    out / 'q7_ability_on_education.csv', index_label='term')
metadata = {'question': 7, 'n': len(df), 'intercept_in_each_regression': True,
            'delta_tilde_1': delta,
            'beta_tilde_1': float(short.params['education']),
            'beta_hat_1': float(full.params['education']),
            'beta_hat_2': float(full.params['ability']),
            'lhs_difference': lhs, 'rhs_product': rhs, 'identity_error': lhs - rhs}
(out / 'q7_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
latex = '\n'.join([
    r'\[',
    rf'\widehat{{\mathrm{{ability}}}}_i = {aux.params["const"]:.3f}'
    rf' + {delta:.3f}\,\mathrm{{education}}_i,',
    rf'\qquad \tilde{{\delta}}_1={delta:.3f}.',
    r'\]',
    r'Using unrounded estimates for the calculations,',
    r'\[',
    rf'\tilde{{\beta}}_1-\hat{{\beta}}_1={lhs:.3f}',
    rf'\quad\text{{and}}\quad \hat{{\beta}}_2\tilde{{\delta}}_1={rhs:.3f}.',
    r'\]',
    r'Thus $\tilde{\beta}_1-\hat{\beta}_1=\hat{\beta}_2\tilde{\delta}_1$',
    r'holds up to numerical precision. All displayed estimates are rounded to three decimals.', ''])
(out / 'q7_output.tex').write_text(latex)
print(json.dumps(metadata, indent=2))
print('Verified auxiliary slope independently and coefficient identity before rounding.')
