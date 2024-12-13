# Clear all objects from the workspace
rm(list=ls())

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241130')

# Source external R scripts
source('library.R')
source('DGP_latent_dynamic.R')
source('fit_apriori_latent.R')
source('fit_BMA.R')
source('fit_BPS.R')
source('fit_BPS2.R')

# Load additional libraries from a custom function
library_load()

# Set the seed for reproducibility
seed <- 123

# Parameters for data simulation
Nt <- 100  # Length of each time series

# Simulate the data
df_latent <- DGP_latent(Nt = Nt, seed = seed, train_ratio = 0.8)

##################
# Fit stan model #
##################

n_iter <- 1000
n_chains <- 1

# model stacking
res_apriori_latent <- fit_apriori_latent(data = df_latent, iter = n_iter, chains = n_chains)
res_BMA <- fit_BMA(data = res_apriori_latent$data_fit, iter = n_iter, chains = n_chains)
res_BPS <- fit_BPS(data = res_apriori_latent$data_fit, iter = n_iter, chains = n_chains)
res_BPS2 <- fit_BPS2(data = res_apriori_latent$data_fit, iter = n_iter, chains = n_chains)

res_ar <- res_apriori_latent$res_ar
res_ma <- res_apriori_latent$res_ma
res_wn <- res_apriori_latent$res_wn

# sort(summary(res_ar$fit)$summary[, "Rhat"])
# sort(summary(res_ma$fit)$summary[, "Rhat"])
# sort(summary(res_wn$fit)$summary[, "Rhat"])
# sort(summary(res_BMA$fit)$summary[, "Rhat"])
# sort(summary(res_BPS$fit)$summary[, "Rhat"])
# sort(summary(res_BPS2$fit)$summary[, "Rhat"])

print(data.frame(
  Model = "elpd_loo",
  AR = res_ar$loo_result$estimates["elpd_loo", "Estimate"], 
  MA = res_ma$loo_result$estimates["elpd_loo", "Estimate"], 
  WN = res_wn$loo_result$estimates["elpd_loo", "Estimate"], 
  BMA = res_BMA$loo_result$estimates["elpd_loo", "Estimate"],
  BPS = res_BPS$loo_result$estimates["elpd_loo", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["elpd_loo", "Estimate"]))

print(data.frame(
  Model = "p_loo",
  AR = res_ar$loo_result$estimates["p_loo", "Estimate"], 
  MA = res_ma$loo_result$estimates["p_loo", "Estimate"],
  WN = res_ma$loo_result$estimates["p_loo", "Estimate"],
  BMA = res_BMA$loo_result$estimates["p_loo", "Estimate"], 
  BPS = res_BPS$loo_result$estimates["p_loo", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["p_loo", "Estimate"]))

print(data.frame(
  Model = "looic",
  AR = res_ar$loo_result$estimates["looic", "Estimate"], 
  MA = res_ma$loo_result$estimates["looic", "Estimate"],
  WN = res_ma$loo_result$estimates["looic", "Estimate"],
  BMA = res_BMA$loo_result$estimates["looic", "Estimate"],
  BPS = res_BPS$loo_result$estimates["looic", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["looic", "Estimate"]))

print(data.frame(
  Model = "test_rmse",
  AR = res_ar$test_rmse,
  MA = res_ma$test_rmse,
  WN = res_wn$test_rmse,
  BMA = res_BMA$test_rmse, 
  BPS = res_BPS$test_rmse, 
  BPS2 = res_BPS2$test_rmse))

