"""Question 4: partial education out of ability, then regress wage on res."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / 'data' / 'PS2_data.csv')
if len(df) != 200 or not np.isfinite(df[['education', 'ability', 'wage']]).all().all():
    raise ValueError('Expected 200 complete, finite observations.')
first = sm.OLS(df['education'], sm.add_constant(df[['ability']]),
               missing='raise').fit(cov_type='nonrobust')
df['res'] = first.resid
second = sm.OLS(df['wage'], sm.add_constant(df[['res']]),
                missing='raise').fit(cov_type='nonrobust')
full = sm.OLS(df['wage'], sm.add_constant(df[['education', 'ability']]),
              missing='raise').fit(cov_type='nonrobust')
# Verify residual orthogonality and the full-precision slope equality.
np.testing.assert_allclose(np.column_stack([np.ones(len(df)), df['ability']]).T
                           @ df['res'], 0, atol=1e-9)
np.testing.assert_allclose(second.params['res'], full.params['education'],
                           rtol=1e-10, atol=1e-10)
out = ROOT / 'results'
out.mkdir(exist_ok=True)
df.to_csv(out / 'q4_data_with_res.csv', index=False)
for name, model in [('q4_education_on_ability', first), ('q4_wage_on_res', second)]:
    pd.DataFrame({'coefficient': model.params, 'std_error': model.bse}).to_csv(
        out / f'{name}.csv', index_label='term')
metadata = {'question': 4, 'n': len(df), 'intercept_in_each_regression': True,
            'res_coefficient': float(second.params['res']),
            'q1_education_coefficient': float(full.params['education']),
            'difference': float(second.params['res'] - full.params['education'])}
(out / 'q4_results.json').write_text(json.dumps(metadata, indent=2) + '\n')
latex = '\n'.join([
    r'\[',
    rf'\widehat{{\mathrm{{education}}}}_i = {first.params["const"]:.3f}'
    rf' + {first.params["ability"]:.3f}\,\mathrm{{ability}}_i,',
    r'\qquad \mathrm{res}_i=\mathrm{education}_i-\widehat{\mathrm{education}}_i.',
    r'\]',
    r'The second regression gives',
    r'\[',
    rf'\widehat{{\mathrm{{wage}}}}_i = {second.params["const"]:.3f}'
    rf' + {second.params["res"]:.3f}\,\mathrm{{res}}_i.',
    r'\]',
    r'\begin{center}', r'\begin{tabular}{lr}', r'\hline',
    r'Coefficient & Estimate \\', r'\hline',
    rf'On res in Question 4 & {second.params["res"]:.3f} \\',
    rf'On education in Question 1 & {full.params["education"]:.3f} \\',
    r'\hline', r'\end{tabular}', r'\end{center}', ''])
(out / 'q4_output.tex').write_text(latex)
print(json.dumps(metadata, indent=2))
print('Verified residual orthogonality and slope equality before rounding.')
