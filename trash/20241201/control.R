# Clear all objects from the workspace
rm(list=ls())

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241201')

# Source external R scripts
source('library.R')
source('DGP.R')
source('fit_apriori.R')
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
df <- DGP(Nt = Nt, seed = seed, train_ratio = 0.8)

##################
# Fit stan model #
##################

n_iter <- 2000
n_chains <- 4

# model stacking
res_apriori <- fit_apriori(data = df, iter = n_iter, chains = n_chains)
res_BMA <- fit_BMA(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
res_BPS <- fit_BPS(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
res_BPS2 <- fit_BPS2(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)

res_ar <- res_apriori$res_ar
res_ma <- res_apriori$res_ma

# sort(summary(res_ar$fit)$summary[, "Rhat"])
# sort(summary(res_ma$fit)$summary[, "Rhat"])
# sort(summary(res_BMA$fit)$summary[, "Rhat"])
# sort(summary(res_BPS$fit)$summary[, "Rhat"])
# sort(summary(res_BPS2$fit)$summary[, "Rhat"])

print(data.frame(
  Model = "elpd_loo",
  AR = res_ar$loo_result$estimates["elpd_loo", "Estimate"], 
  MA = res_ma$loo_result$estimates["elpd_loo", "Estimate"], 
  BMA = res_BMA$loo_result$estimates["elpd_loo", "Estimate"],
  BPS = res_BPS$loo_result$estimates["elpd_loo", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["elpd_loo", "Estimate"]))

print(data.frame(
  Model = "p_loo",
  AR = res_ar$loo_result$estimates["p_loo", "Estimate"], 
  MA = res_ma$loo_result$estimates["p_loo", "Estimate"],
  BMA = res_BMA$loo_result$estimates["p_loo", "Estimate"], 
  BPS = res_BPS$loo_result$estimates["p_loo", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["p_loo", "Estimate"]))

print(data.frame(
  Model = "looic",
  AR = res_ar$loo_result$estimates["looic", "Estimate"], 
  MA = res_ma$loo_result$estimates["looic", "Estimate"],
  BMA = res_BMA$loo_result$estimates["looic", "Estimate"],
  BPS = res_BPS$loo_result$estimates["looic", "Estimate"],
  BPS2 = res_BPS2$loo_result$estimates["looic", "Estimate"]))

print(data.frame(
  Model = "test_rmse",
  AR = res_ar$test_rmse,
  MA = res_ma$test_rmse,
  BMA = res_BMA$test_rmse, 
  BPS = res_BPS$test_rmse, 
  BPS2 = res_BPS2$test_rmse))


model_list <- list(res_ar$fit, res_ma$fit)
log_lik_list <- lapply(model_list, extract_log_lik)

# optional but recommended
r_eff_list <- lapply(model_list, function(x) {
  ll_array <- extract_log_lik(x, merge_chains = FALSE)
  relative_eff(exp(ll_array))
}) 

# stacking method
wts1 <- loo_model_weights(log_lik_list, 
                          method = "stacking", 
                          r_eff_list = r_eff_list,
                          optim_control = list(reltol=1e-10)
)

# can also pass a list of psis_loo objects to avoid recomputing loo 
loo_list <- lapply(1:length(log_lik_list), function(j) {
  loo(log_lik_list[[j]], r_eff = r_eff_list[[j]])
})

wts2 <- loo_model_weights(loo_list, 
                          method = "stacking", 
                          r_eff_list = r_eff_list,
                          optim_control = list(reltol=1e-10)
)
all.equal(wts1, wts2)

# can provide names to be used in the results
loo_model_weights(setNames(loo_list, c("AR", "MA")))

# preudo-BMA+ method:
loo_model_weights(loo_list, method = "pseudobma")

# pseudo-BMA method (set BB = FALSE):
loo_model_weights(loo_list, method = "pseudobma", BB = FALSE)

# customized BMA results
colMeans(extract(res_BMA$fit, "w")$w)

