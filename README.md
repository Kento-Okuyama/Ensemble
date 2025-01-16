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