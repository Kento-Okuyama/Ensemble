# Clear all objects from the workspace
rm(list=ls())

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241127')

# Source external R scripts
source('library.R')
source('observed/DGP_observed.R')
source('observed/fit_apriori_observed.R')
source('fit_BMA.R')
source('fit_BMAxTP.R')
source('fit_BPS.R')

# Load additional libraries from a custom function
library_load()

# Set the seed for reproducibility
seed <- 123

# Parameters for data simulation
N <- 5    # Number of time series
Nt <- 30  # Length of each time series

# Simulate the data
df_obs <- DGP_obs(N = N, Nt = Nt, seed = seed, train_ratio = 0.8)

##################
# Fit stan model #
##################

# model stacking
res_apriori_obs <- fit_apriori_obs(data = df_obs, iter = 1000, chains = 1)
res_BMA <- fit_BMA(data = res_apriori_latent$data_fit, iter = 1000, chains = 1)
res_BMAxTP <- fit_BMAxTP(data = res_apriori_latent$data_fit, iter = 1000, chains = 1)
res_BPS <- fit_BPS(data = res_apriori_latent$data_fit, iter = 1000, chains = 1)

loo_compare(res_BMA$loo_result, res_BMAxTP$loo_result, res_BPS$loo_result)

print(data.frame(
  Model = "elpd_loo",
  BMA = res_BMA$loo_result$estimates["elpd_loo", "Estimate"], 
  BMAxTP = res_BMAxTP$loo_result$estimates["elpd_loo", "Estimate"], 
  BPS = res_BPS$loo_result$estimates["elpd_loo", "Estimate"]))

print(data.frame(
  Model = "p_loo",
  BMA = res_BMA$loo_result$estimates["p_loo", "Estimate"], 
  BMAxTP = res_BMAxTP$loo_result$estimates["p_loo", "Estimate"], 
  BPS = res_BPS$loo_result$estimates["p_loo", "Estimate"]))

print(data.frame(
  Model = "looic",
  BMA = res_BMA$loo_result$estimates["looic", "Estimate"],
  BMAxTP = res_BMAxTP$loo_result$estimates["looic", "Estimate"],
  BPS = res_BPS$loo_result$estimates["looic", "Estimate"]))

print(data.frame(
  Model = "test_rmse",
  BMA = res_BMA$test_rmse, 
  BMAxTP = res_BMAxTP$test_rmse, 
  BPS = res_BPS$test_rmse))

