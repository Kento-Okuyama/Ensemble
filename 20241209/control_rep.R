# ===========================
#  Clear Workspace and Setup
# ===========================
rm(list = ls())  # Clear all objects from the workspace

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241209')

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
#  Set Parameters
# ===========================
N <- 10       
Nt <- 50    

# ===========================
#   Multiple Runs Setup
# ===========================
n_runs <- 100          # Number of iterations
result_list <- list()  # Store results for each run

# ===========================
#   Model Fitting Parameters
# ===========================
n_iter <- 2000   # Number of iterations for Stan model
n_chains <- 4    # Number of chains for Stan model

# Progress bar setup
pb <- txtProgressBar(min = 0, max = n_runs, style = 3)

for (i in 1:n_runs) {
  # Update seed
  seed <- 123 + i  # Change seed for each iteration
  
  # Generate data
  df <- DGP(N = N, Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)
  
  # Fit models
  res_apriori <- fit_apriori(data = df, iter = n_iter, chains = n_chains)
  res_BMA <- fit_BMA(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
  res_BPS <- fit_BPS(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
  res_BPS2 <- fit_BPS2(data = res_apriori$data_fit, iter = n_iter, chains = n_chains)
  
  # Extract results
  results <- data.frame(
    run = i,
    elpd_loo_AR = res_apriori$res_ar$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_MA = res_apriori$res_ma$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_BMA = res_BMA$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_BPS = res_BPS$loo_result$estimates["elpd_loo", "Estimate"],
    elpd_loo_BPS2 = res_BPS2$loo_result$estimates["elpd_loo", "Estimate"],
    test_rmse_AR = res_apriori$res_ar$test_rmse,
    test_rmse_MA = res_apriori$res_ma$test_rmse,
    test_rmse_BMA = res_BMA$test_rmse,
    test_rmse_BPS = res_BPS$test_rmse,
    test_rmse_BPS2 = res_BPS2$test_rmse
  )
  
  # Append to list
  result_list[[i]] <- results
  
  # Update progress bar
  setTxtProgressBar(pb, i)
}

# Combine results into a single data frame
final_results <- do.call(rbind, result_list)

# Save results to a CSV file
write.csv(final_results, "model_comparison_results.csv", row.names = FALSE)

# ===========================
#   Analyze Results
# ===========================
library(dplyr)

# Calculate summary statistics
summary_stats <- final_results %>%
  summarise(
    across(starts_with("elpd_loo"), list(mean = mean, sd = sd)),
    across(starts_with("test_rmse"), list(mean = mean, sd = sd))
  )

# Print summary statistics
print(summary_stats)

# Optionally, visualize results
rmse_columns <- final_results %>% select(starts_with("test_rmse"))

boxplot(rmse_columns, 
        main = "Test RMSE across models", 
        las = 2, 
        xlab = "Models", 
        ylab = "Test RMSE",
        names = c("AR", "MA", "BMA", "BPS", "BPS2"))


