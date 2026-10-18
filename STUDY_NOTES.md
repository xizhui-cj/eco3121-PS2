# Learning guide: the ideas behind this assignment

## Q1 and Q2: the dependent variable determines the interpretation

Q1 uses wage itself. Its education slope, 3.690, is a difference in hourly
wage units for one additional year of schooling, holding ability fixed.

Q2 uses natural log wage. Its education slope is about 0.065 log points per
additional year. Exponentiation turns an addition in logs into a multiplication:

`exp(log(wage) + b) = wage * exp(b)`.

With the displayed rounded coefficient, the factor is exp(0.065), about
1.0672. Subtract 1 and multiply by 100 to get approximately 6.72%.
The quick approximation is 100*b, or 6.5%. These percentages use the rounded
coefficient. The final answer uses the full coefficient and reports 6.751%.
More precisely, the regression describes conditional mean log wages;
predicting conditional mean wage levels also requires addressing the log
transformation of the error. These associations are not automatically causal.

## Q3 and Q6: model fit

Total sum of squares measures the observed wage variation around mean wage.
With OLS and an intercept, it splits into explained and residual sums of
squares. The assignment calls these SSE and SSR respectively. Other sources
sometimes reverse those labels, so check the definitions.

`R-squared = explained / total = 1 - residual / total`.

The standard error of the regression is `sqrt(SSR / (n-k))`, where k includes
the intercept. For Q1, k=3 and n-k=197. It measures residual dispersion in wage
units. It is different from the standard error of an estimated coefficient.

Adding a regressor to a nested OLS model on the same sample cannot increase
the residual sum of squares, because the old model is still available by
setting the new coefficient to zero. Consequently, ordinary R-squared and
explained sum of squares cannot fall. This is about in-sample fit; it does
not prove that the new model is causal or better at predicting new data.

## Q4: controlling for ability through residuals

The education-on-ability regression predicts schooling from ability. A
residual of +2 means two more years of schooling than that linear prediction.
These residuals are orthogonal to ability in the sample. The wage-on-residual
slope therefore isolates the same variation in education as the full Q1
regression. The matching slopes illustrate the Frisch-Waugh-Lovell theorem.
The residual need not be independent of ability: orthogonality is weaker.
Matching slopes do not imply that both regressions have identical standard
errors or goodness of fit.

## Q5, Q7, and Q8: omitted-variable bias

Two ingredients produce confounding here:

1. Ability is associated with wages even after controlling for education.
   Q1's ability coefficient is about 2.086 wage units per ability unit.
2. Ability is associated with education. Q7's auxiliary slope is about
   0.512 ability units per additional year of education.

The product has units `(wage/ability)*(ability/year) = wage/year`.
Using full-precision estimates, this product is about 1.069. It equals the
difference between the education slopes with and without ability. Multiplying
the displayed rounded numbers need not reproduce the last decimal.

The sample identity is algebraic. Calling it causal bias additionally
requires assumptions about the full model. Under exogeneity of that model,
omitting ability puts it into the error. If ability is correlated with
education, education is now correlated with the error: zero conditional
mean can fail. Positive ability-wage and ability-education relationships
produce upward omitted-variable bias in the education coefficient.

Direction check: if one of these relationships were negative, the bias would
be negative. If either were zero, this particular population bias term would
be zero.

## Q9: bias and uncertainty are different issues

Zero conditional mean concerns the *average error* at given explanatory
variables. It is central to the usual unbiasedness argument.

Homoskedasticity concerns the *spread of the error* at given explanatory
variables. Constant spread is not required for unbiasedness. Its failure can
make the usual standard errors unreliable even when coefficient estimates
remain unbiased under the other assumptions.

Robust standard errors change estimated uncertainty, not the fitted OLS
coefficients, and do not remove omitted-variable bias. The assignment
explicitly requests usual OLS standard errors, so the submitted output uses
those throughout.
