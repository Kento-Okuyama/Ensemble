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
    int<lower=1> S;                // Number of samples per model
    real val_y[N, val_Nt];         // Validation observed data
    real val_f[S, N, val_Nt, J];   // Validation model predictions (samples)
    real test_y[N, test_Nt];       // Test observed data
    real test_f[S, N, test_Nt, J]; // Test model predictions (samples)
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
          for (s in 1:S) {
            weighted_prediction += w[n, j] * val_f[s, n, t, j];
          }
        }
        weighted_prediction /= S; // Take the average over samples
        val_y[n, t] ~ normal(weighted_prediction, sigma);
      }
    }
  }
  
  generated quantities {
    real log_lik[N, val_Nt];       // Log-likelihood for validation data
    real test_y_pred[N, test_Nt];  // Predicted values for test data
    real SSE;                      // Sum of Squared Errors (single scalar)
    real RMSE;                     // Root Mean Square Error (single scalar)
    real pi_t;
    real gamma;

    SSE = 0;
    gamma = 0.1;
    
    for (n in 1:N) {
      // Log-likelihood for validation data
      for (t in 1:val_Nt) {
        real weighted_prediction = 0;
        for (j in 1:J) {
          for (s in 1:S) {
            weighted_prediction += w[n, j] * val_f[s, n, t, j];
          }
        }
        weighted_prediction /= S; // Average over samples
        pi_t = 1 + gamma - square(1 - 1.0 * t / val_Nt);
        log_lik[n, t] = pi_t * normal_lpdf(val_y[n, t] | weighted_prediction, sigma);
      }
  
      // Predictions for test data and calculate RMSE
      for (t in 1:test_Nt) {
        real weighted_prediction = 0;
        for (j in 1:J) {
          for (s in 1:S) {
            weighted_prediction += w[n, j] * test_f[s, n, t, j];
          }
        }
        weighted_prediction /= S; // Average over samples
        test_y_pred[n, t] = normal_rng(weighted_prediction, sigma);
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
  weights_df <- weights 

  # Average weights across samples for each model and person
  average_weights <- apply(weights_df, c(2, 3), mean)
  colnames(average_weights) <- paste0("Model_", 1:dim(weights)[3])  
  
  # Print the average weights
  cat("\nAverage weights across individuals:\n")
  print(average_weights)
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    fit = fit,
    weights = weights_df,
    log_lik = extract_log_lik(fit, parameter_name = "log_lik"),
    test_rmse = mean(extract(fit, pars = "RMSE")$RMSE)
  ))
}
