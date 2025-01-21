# Ensemble

### M-complete problem
 - DGP: AR1 & MA1 & WN
 - Fitted models: AR1 & MA1 

20241219: y-based ensemble
20241230: eta-based ensemble

### M-open problem
 - DGP: ARMA(2,3) & MA(2) & WN
 - Fitted models: AR1 & MA1 

20241218: y-based ensemble
20250109: eta-based ensemble

### next step 

Modify fit_BPS2.R
alpha[n, t] ~ normal(alpha[n, t - 1] + pred_e[n, train_Nt + t - 1] .* delta[n], tau_a);
beta[n, t] ~ normal(beta[n, t - 1] + pred_e[n, train_Nt + t - 1] .* theta[n], tau_b);

Right now, each model weight only depends on the prediciton of a single model (agent). 
In order to fully utilize BPS, each model weight should be regressed on all model (agent) predictions 



# Application of RS models

- Premenstrual Syndrome (PMS) [1]
- Abrupt Mood Switching in Depressive Disorder [1]
- The hot hand of Vinnie Johnson [1]
- Rapid cycling bipolar disorder [2]
- Dynamic Model of Activation [3]


[1] Hamaker, E.L., Grasman, R.P.P.P. Regime Switching State-Space Models Applied to Psychological Processes: Handling Missing Data and Making Inferences. Psychometrika 77, 400–422 (2012). https://doi.org/10.1007/s11336-012-9254-8
[2] Hamaker, Ellen & Grasman, Raoul & Kamphuis, Jan. (2010). Regime-switching models to study psychological process. 
[3] Chow, S.-M., & Zhang, G. (2013). Nonlinear Regime-Switching State-Space (RSSS) models. Psychometrika, 78(4), 740–768. https://doi.org/10.1007/s11336-013-9330-8
