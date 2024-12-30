# ===========================
#  Clear Workspace and Setup
# ===========================
rm(list = ls())  # Clear all objects from the workspace

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/working')

# ===========================
#   Load External Scripts
# ===========================
source('library.R')       # Contains library imports and setups
source('DGP.R')           # Function for data generation process
source('fit_apriori.R')   # Model fitting function: Apriori
source('fit_BMA.R')       # Model fitting function: Bayesian Model Averaging
source('fit_BPS.R')       # Model fitting function: Bayesian Predictive Stacking
source('fit_BPS2.R')      # Model fitting function: Alternative Bayesian Predictive Stacking

library_load()  # Load libraries using custom function

# ===========================
#  Set Parameters
# ===========================
N <- 10      
Nt <- 50    
i <- 1

# ===========================
#   Multiple Runs Setup
# ===========================
n_runs <- 1          # Number of iterations
result_list <- list()  # Store results for each run

# ===========================
#   Model Fitting Parameters
# ===========================
n_iter <- 100   # Number of iterations for Stan model
n_chains <- 1    # Number of chains for Stan model

# Progress bar setup
pb <- txtProgressBar(min = 0, max = n_runs, style = 3)

for (i in 1:n_runs) {
  # Update seed
  seed <- 123 + i  # Change seed for each iteration
  
  # Generate data
  df <- DGP(N = N, Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)
  
  # Fit models
  res_apriori <- fit_apriori(data = df, iter = n_iter, chains = n_chains, refresh = 0)
  res_BMA <- fit_BMA(data = res_apriori$data_fit, iter = n_iter, chains = n_chains, refresh = 0)
  res_BPS <- fit_BPS(data = res_apriori$data_fit, iter = n_iter, chains = n_chains, refresh = 0)
  res_BPS2 <- fit_BPS2(data = res_apriori$data_fit, iter = n_iter, chains = n_chains, refresh = 0)
  
  # Extract results
  results <- data.frame(
    run = i,
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

# Calculate summary statistics
summary_stats <- final_results %>%
  summarise(
    across(starts_with("test_rmse"), list(mean = mean, sd = sd))
  )

# Print summary statistics
print(summary_stats)

# Optionally, visualize results
rmse_columns <- final_results %>% select(starts_with("test_rmse"))

# Specify the file path and format 
png("test_rmse_across_models.png", width = 800, height = 600)

# Create the plot
boxplot(rmse_columns, 
        main = "Test RMSE across models", 
        las = 2, 
        xlab = "Models", 
        ylab = "Test RMSE",
        names = c("AR", "MA", "BMA", "BPS", "BPS2")) 

# Close the file
dev.off()

# ===========================
#   Visualize Weights Across Runs
# ===========================
all_weights <- do.call(rbind, lapply(result_list, function(res) res$weights))
all_weights_long <- reshape2::melt(all_weights, variable.name = "Model", value.name = "Weight")

# Boxplot for model weights
library(ggplot2)
ggplot(all_weights_long, aes(x = Model, y = Weight)) +
  geom_boxplot() +
  labs(title = "Distribution of Model Weights Across Runs",
       x = "Model",
       y = "Weight") +
  theme_minimal()

