# ===========================
#  Clear Workspace and Setup
# ===========================
rm(list = ls())  # Clear all objects from the workspace

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241219')

# ===========================
#  Load External Scripts
# ===========================
source('library.R')
source('DGP.R')
source('fit_apriori.R')
source('fit_BMA.R')
source('fit_BPS.R')
source('fit_BPS2.R')

library_load()

# ===========================
#  Set Parameters
# ===========================
seed <- 123  
N <- 10       
Nt <- 50     

# ===========================
#  Simulate Data
# ===========================
df <- DGP(N = N, Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)

table(c(df$train_regime[1,], df$val_regime[1,], df$test_regime[1,]))

plot(c(df$train_regime[1,], df$val_regime[1,], df$test_regime[1,]), 
     type = "l", 
     main = "Regime Visualization", 
     xlab = "Time", 
     ylab = "Regime")

# ===========================
#  Model Fitting Parameters
# ===========================
n_iter <- 2000
n_chains <- 4

# ===========================
#  Fit Models
# ===========================
res_apriori <- fit_apriori(data = df, iter = n_iter, chains = n_chains)
res_BMA <- fit_BMA(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
res_BPS <- fit_BPS(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
res_BPS2 <- fit_BPS2(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)

res_ar <- res_apriori$res_ar
res_ma <- res_apriori$res_ma

# ===========================
#  Check Convergence (Rhat)
# ===========================
rhat_values <- list(
  AR = summary(res_ar$fit)$summary[, "Rhat"],
  MA = summary(res_ma$fit)$summary[, "Rhat"],
  BMA = summary(res_BMA$fit)$summary[, "Rhat"],
  BPS = summary(res_BPS$fit)$summary[, "Rhat"],
  BPS2 = summary(res_BPS2$fit)$summary[, "Rhat"]
)

lapply(rhat_values, sort)

# ===========================
#  Compare Models
# ===========================
comparison <- data.frame(
  Metric = c("elpd_loo", "p_loo", "looic", "test_rmse"),
  AR = c(res_ar$loo_result$estimates["elpd_loo", "Estimate"],
         res_ar$loo_result$estimates["p_loo", "Estimate"],
         res_ar$loo_result$estimates["looic", "Estimate"],
         res_ar$test_rmse),
  MA = c(res_ma$loo_result$estimates["elpd_loo", "Estimate"],
         res_ma$loo_result$estimates["p_loo", "Estimate"],
         res_ma$loo_result$estimates["looic", "Estimate"],
         res_ma$test_rmse),
  BMA = c(res_BMA$loo_result$estimates["elpd_loo", "Estimate"],
          res_BMA$loo_result$estimates["p_loo", "Estimate"],
          res_BMA$loo_result$estimates["looic", "Estimate"],
          res_BMA$test_rmse),
  BPS = c(res_BPS$loo_result$estimates["elpd_loo", "Estimate"],
          res_BPS$loo_result$estimates["p_loo", "Estimate"],
          res_BPS$loo_result$estimates["looic", "Estimate"],
          res_BPS$test_rmse),
  BPS2 = c(res_BPS2$loo_result$estimates["elpd_loo", "Estimate"],
           res_BPS2$loo_result$estimates["p_loo", "Estimate"],
           res_BPS2$loo_result$estimates["looic", "Estimate"],
           res_BPS2$test_rmse)
)
print(comparison)

