fit_BMA <- function(data, iter = 2000, chains = 4, refresh = 0) {
  # ===========================
  #    Stan Model Code
  # ===========================
  # Bayesian Model Averaging (BMA) Stan model
  stan_code <- "
  data {
    int<lower=1> N;                // Number of individuals
    int<lower=1> val_Nt;           // Number of validation data points
    int<lower=1> test_Nt;          // Number of test data points
    int<lower=1> J;                // Number of models
    real val_y[N, val_Nt];         // Validation observed data
    real val_f[N, val_Nt, J];      // Validation model predictions
    real test_y[N, test_Nt];       // Test observed data
    real test_f[N, test_Nt, J];    // Test model predictions
  }
  
  parameters {
    simplex[J] w[N];               // Model weights for each individual (Dirichlet prior)
    real<lower=0> sigma;           // Observation noise standard deviation (shared across individuals)
  }
  
  model {
    // Priors
    sigma ~ normal(0, 1e-1);               // Prior for noise (positive)
    for (n in 1:N) {
      w[n] ~ dirichlet(rep_vector(1.0, J));  // Dirichlet prior for model weights for each individual
      
      // Likelihood for validation data
      for (t in 1:val_Nt) {
        real weighted_prediction = 0;
        for (j in 1:J) {
          weighted_prediction += w[n, j] * val_f[n, t, j];
        }
        val_y[n, t] ~ normal(weighted_prediction, sigma);
      }
    }
  }
  
  generated quantities {
    real log_lik[N, val_Nt];  // Log-likelihood for validation data (PSIS-LOO)
    real test_y_pred[N, test_Nt]; // Predicted values for test data
    real SSE;                   // Sum of Squared Errors (single scalar)
    real RMSE;                  // Root Mean Square Error (single scalar)
    real pi_t;
    real gamma;

    SSE = 0;
    gamma = 0.1;
    for (n in 1:N) {
      // Log-likelihood for validation data
      for (t in 1:val_Nt) {
        real weighted_prediction = 0;
        for (j in 1:J) {
          weighted_prediction += w[n, j] * val_f[n, t, j];
        }
        pi_t = 1 + gamma - square(1 - 1.0 * t / val_Nt);
        log_lik[n, t] = pi_t * normal_lpdf(val_y[n, t] | weighted_prediction, sigma);
      }
  
      // Predictions for test data and calculate RMSE
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = 0;
        for (j in 1:J) {
          test_y_pred[n, t] += normal_rng(w[n, j] * test_f[n, t, j], sigma);
        }
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }
    RMSE = sqrt(SSE / (N * test_Nt));
  }
  "
  # ===========================
  #    Fit the Stan Model
  # ===========================
  fit <- tryCatch({
    stan(model_code = stan_code, data = data, iter = iter, chains = chains, refresh = refresh)
  }, error = function(e) {
    stop("Stan model fitting failed: ", e)
  })
  
  # ===========================
  #    Extract and Process Results
  # ===========================
  # Extract model weights
  weights <- extract(fit, pars = "w")$w
  
  # Convert weights to a data frame for visualization
  weights_df <- as.data.frame(apply(weights, c(2, 3), mean))  
  colnames(weights_df) <- paste0("Model_", 1:dim(weights)[3])  
  
  # Average weights across individuals for each model
  average_weights <- colMeans(weights_df)
  
  # Print the average weights
  cat("\nAverage weights across individuals:\n")
  print(average_weights)
  
  # Optional: Visualize the weights
  barplot(average_weights,
          main = "Average Weights Across Models",
          xlab = "Models",
          ylab = "Average Weight",
          names.arg = colnames(weights_df))
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    fit = fit,
    log_lik = log_lik,
    loo_result = loo_result,
    test_rmse = test_rmse,
    post_plot = post_plot,
    weights = weights_df
  ))
}

# ===========================
#  Clear Workspace and Setup
# ===========================
rm(list = ls())  # Clear all objects from the workspace

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241219')

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

# ===========================
#   Multiple Runs Setup
# ===========================
n_runs <- 50          # Number of iterations
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
