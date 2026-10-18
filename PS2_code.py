"""ECO3121 Problem Set 2: all computational questions in one script.

Run: python PS2_code.py
Input: data/PS2_data.csv (relative to this script)
Output: code_output/ (regression results, tables, and Q4 residuals)
Questions 8 and 9 are conceptual and are answered in the report.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / 'data' / 'PS2_data.csv')
    columns = ['wage', 'education', 'ability']
    if len(data) != 200 or not np.isfinite(data[columns].to_numpy()).all():
        raise ValueError('Expected 200 observations with no missing or nonfinite values.')
    if not (data['wage'] > 0).all():
        raise ValueError('All wages must be positive for the natural logarithm.')

    out = root / 'code_output'
    out.mkdir(exist_ok=True)
    lines = []

    def show(text):
        lines.append(str(text))
        print(text)

    def regress(y, regressors):
        X = sm.add_constant(data[regressors], has_constant='add')
        if np.linalg.matrix_rank(X) != X.shape[1]:
            raise ValueError('Regressors are not of full column rank.')
        return sm.OLS(y, X, missing='raise').fit(cov_type='nonrobust')

    def coefficients(model, filename):
        table = pd.DataFrame({'coefficient': model.params, 'std_error': model.bse})
        table.to_csv(out / filename, index_label='term')
        show(table.to_string(float_format=lambda x: f'{x:.3f}'))

    # Q1: wage on education and ability, including an intercept.
    show('Question 1')
    full = regress(data['wage'], ['education', 'ability'])
    coefficients(full, 'q1_coefficients.csv')

    # Q2: natural log wage on education and ability.
    show('\nQuestion 2')
    log_model = regress(np.log(data['wage']), ['education', 'ability'])
    coefficients(log_model, 'q2_coefficients.csv')
    show(f'Education percentage change: {100 * np.expm1(log_model.params["education"]):.3f}%')

    # Q3: fit statistics for the Q1 levels regression.
    show('\nQuestion 3')
    show(f'R-squared: {full.rsquared:.3f}')
    show(f'Residual sum of squares (SSR): {full.ssr:.3f}')
    show(f'Standard error of regression: {np.sqrt(full.mse_resid):.3f}')

    # Q4: residualize education against ability, then regress wage on res.
    show('\nQuestion 4: education on ability')
    education_model = regress(data['education'], ['ability'])
    coefficients(education_model, 'q4_education_on_ability.csv')
    data['res'] = education_model.resid
    data.to_csv(out / 'q4_data_with_res.csv', index=False)
    show('\nQuestion 4: wage on res')
    residual_model = regress(data['wage'], ['res'])
    coefficients(residual_model, 'q4_wage_on_res.csv')
    np.testing.assert_allclose(residual_model.params['res'], full.params['education'],
                               rtol=1e-10, atol=1e-10)
    show(f'Q4 res slope = Q1 education slope: {residual_model.params["res"]:.3f}')

    # Q5: wage on education alone, including an intercept.
    show('\nQuestion 5')
    short = regress(data['wage'], ['education'])
    coefficients(short, 'q5_coefficients.csv')

    # Q6: use SSE for explained and SSR for residual sums, as in the assignment.
    show('\nQuestion 6')
    total_ss = float(((data['wage'] - data['wage'].mean()) ** 2).sum())
    rows = []
    for label, model in [('Q5: education only', short), ('Q1: education and ability', full)]:
        explained_ss = float(((model.fittedvalues - data['wage'].mean()) ** 2).sum())
        np.testing.assert_allclose(explained_ss + model.ssr, total_ss, rtol=1e-10)
        rows.append({'model': label, 'n': int(model.nobs), 'R_squared': model.rsquared,
                     'explained_SS_SSE': explained_ss})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / 'q6_comparison.csv', index=False)
    show(comparison.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    # Q7: ability on education and the omitted-variable coefficient identity.
    show('\nQuestion 7: ability on education')
    ability_model = regress(data['ability'], ['education'])
    coefficients(ability_model, 'q7_ability_on_education.csv')
    delta = ability_model.params['education']
    difference = short.params['education'] - full.params['education']
    product = full.params['ability'] * delta
    np.testing.assert_allclose(difference, product, rtol=1e-10, atol=1e-10)
    show(f'delta_tilde_1: {delta:.3f}')
    show(f'beta_tilde_1 - beta_hat_1: {difference:.3f}')
    show(f'beta_hat_2 * delta_tilde_1: {product:.3f}')
    (out / 'PS2_output.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
