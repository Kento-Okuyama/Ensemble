# ===========================
#  Clear Workspace and Setup
# ===========================
rm(list = ls())  # Clear all objects from the workspace

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241203')

# ===========================
#   Load External Scripts
# ===========================
# Load necessary R scripts for custom functions and models
source('library.R')       # Contains library imports and setups
source('DGP.R')           # Function for data generation process
source('fit_apriori.R')   # Model fitting function: Apriori
source('fit_BMA.R')       # Model fitting function: Bayesian Model Averaging
source('fit_BPS.R')       # Model fitting function: Bayesian Predictive Stacking
source('fit_BPS2.R')      # Model fitting function: Alternative Bayesian Predictive Stacking

# Load additional libraries using a custom function
library_load()

# ===========================
#    Set Parameters
# ===========================
seed <- 123  # Set the random seed for reproducibility

# Parameters for data simulation
Nt <- 50  # Length of each time series

# ===========================
#    Simulate Data
# ===========================
# Generate data using the custom DGP function
df <- DGP(Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)

# Summarize the generated data's regimes
table(c(df$train_regime, df$val_regime, df$test_regime))

# Visualize the regimes across train, validation, and test sets
plot(c(df$train_regime, df$val_regime, df$test_regime), type = "l", 
     main = "Regime Visualization", xlab = "Time", ylab = "Regime")

# ===========================
#   Model Fitting Parameters
# ===========================
n_iter <- 2000   # Number of iterations for Stan model
n_chains <- 4    # Number of chains for Stan model

# ===========================
#      Fit Models
# ===========================
# Fit initial Apriori model
res_apriori <- fit_apriori(data = df, iter = n_iter, chains = n_chains)

# Fit Bayesian Model Averaging (BMA) model
res_BMA <- fit_BMA(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)

# Fit Bayesian Predictive Stacking (BPS) models
res_BPS <- fit_BPS(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
res_BPS2 <- fit_BPS2(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)

# Extract results for AR and MA models from Apriori
res_ar <- res_apriori$res_ar
res_ma <- res_apriori$res_ma

# ===========================
#   Check Convergence (Rhat)
# ===========================
# Uncomment the lines below for manual inspection of Rhat values
# sort(summary(res_ar$fit)$summary[, "Rhat"])
# sort(summary(res_ma$fit)$summary[, "Rhat"])
# sort(summary(res_BMA$fit)$summary[, "Rhat"])
# sort(summary(res_BPS$fit)$summary[, "Rhat"])
# sort(summary(res_BPS2$fit)$summary[, "Rhat"])

# ===========================
#     Compare Models
# ===========================
# Compare models using Leave-One-Out Cross-Validation (LOO) estimates and RMSE
print(data.frame(
  Model = "elpd_loo",
  AR = res_ar$loo_result$estimates["elpd_loo", "Estimate"],
  MA = res_ma$loo_result$estimates["elpd_loo", "Estimate"],
  BMA = res_BMA$loo_result$estimates["elpd_loo", "Estimate"],
  BPS = res_BPS$loo_result$estimates["elpd_loo", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["elpd_loo", "Estimate"]
))

print(data.frame(
  Model = "p_loo",
  AR = res_ar$loo_result$estimates["p_loo", "Estimate"],
  MA = res_ma$loo_result$estimates["p_loo", "Estimate"],
  BMA = res_BMA$loo_result$estimates["p_loo", "Estimate"],
  BPS = res_BPS$loo_result$estimates["p_loo", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["p_loo", "Estimate"]
))

print(data.frame(
  Model = "looic",
  AR = res_ar$loo_result$estimates["looic", "Estimate"],
  MA = res_ma$loo_result$estimates["looic", "Estimate"],
  BMA = res_BMA$loo_result$estimates["looic", "Estimate"],
  BPS = res_BPS$loo_result$estimates["looic", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["looic", "Estimate"]
))

print(data.frame(
  Model = "test_rmse",
  AR = res_ar$test_rmse,
  MA = res_ma$test_rmse,
  BMA = res_BMA$test_rmse,
  BPS = res_BPS$test_rmse,
  BPS2 = res_BPS2$test_rmse
))

# ===========================
#   Model Weight Estimation
# ===========================
# Prepare models and extract log likelihoods
model_list <- list(res_ar$fit, res_ma$fit)
log_lik_list <- lapply(model_list, extract_log_lik)

# Compute relative efficiency for stacking method
r_eff_list <- lapply(model_list, function(x) {
  ll_array <- extract_log_lik(x, merge_chains = FALSE)
  relative_eff(exp(ll_array))
})

# Stacking method weights
wts1 <- loo_model_weights(log_lik_list, 
                          method = "stacking", 
                          r_eff_list = r_eff_list, 
                          optim_control = list(reltol = 1e-10))
print(wts1)

# Prepare LOO objects to avoid redundant computations
loo_list <- lapply(1:length(log_lik_list), function(j) {
  loo(log_lik_list[[j]], r_eff = r_eff_list[[j]])
})

# Confirm weights consistency
wts2 <- loo_model_weights(loo_list, 
                          method = "stacking", 
                          r_eff_list = r_eff_list, 
                          optim_control = list(reltol = 1e-10))
all.equal(wts1, wts2)

# Assign model names for easier interpretation
loo_model_weights(setNames(loo_list, c("AR", "MA")))

# Compute weights using alternative methods (pseudo-BMA)
loo_model_weights(loo_list, method = "pseudobma")
loo_model_weights(loo_list, method = "pseudobma", BB = FALSE)

# Extract customized BMA results
colMeans(extract(res_BMA$fit, "w")$w)

