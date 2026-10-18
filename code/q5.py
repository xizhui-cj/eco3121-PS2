"""Question 5: wage on education alone, including an intercept."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'PS2_data.csv')
if len(df) != 200 or not np.isfinite(df[['wage', 'education']]).all().all():
    raise ValueError('Expected 200 complete, finite observations.')
X = sm.add_constant(df[['education']], has_constant='add')
model = sm.OLS(df['wage'], X, missing='raise').fit(cov_type='nonrobust')
# Independently check the simple-regression slope using centered variables.
x = df['education'] - df['education'].mean()
y = df['wage'] - df['wage'].mean()
slope = (x @ y) / (x @ x)
np.testing.assert_allclose(model.params['education'], slope, rtol=1e-10)
out = ROOT / 'results'
out.mkdir(exist_ok=True)
pd.DataFrame({'coefficient': model.params, 'std_error': model.bse}).to_csv(
    out / 'q5_coefficients.csv', index_label='term')
metadata = {'question': 5, 'n': int(model.nobs), 'intercept': True,
            'education_coefficient_tilde_beta_1': float(model.params['education'])}
(out / 'q5_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
latex = '\n'.join([
    r'\[',
    rf'\widehat{{\mathrm{{wage}}}}_i = {model.params["const"]:.3f}'
    rf' + {model.params["education"]:.3f}\,\mathrm{{education}}_i.',
    r'\]',
    rf'The estimated education coefficient is $\tilde{{\beta}}_1={model.params["education"]:.3f}$.',
    ''])
(out / 'q5_output.tex').write_text(latex)
print(json.dumps(metadata, indent=2))
print('Verified slope independently using centered-variable formula.')
