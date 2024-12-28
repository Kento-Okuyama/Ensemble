fit_BPS <- function(data, iter = 2000, chains = 4, refresh = 0) {
  # ===========================
  #    Stan Model Code
  # ===========================
  stan_code <- "
  data {
  int<lower=1> N;                  // Number of individuals
  int<lower=1> val_Nt;             // Number of validation data points
  int<lower=1> test_Nt;            // Number of test data points
  int<lower=1> J;                  // Number of models
  real val_y[N, val_Nt];           // Validation observed data
  real val_f[N, val_Nt, J];        // Validation model predictions
  real test_y[N, test_Nt];         // Test observed data
  real test_f[N, test_Nt, J];      // Test model predictions
  }
  
  parameters {
    simplex[J] mu[N];                // Initial model weights for each individual
    vector[val_Nt + test_Nt] alpha[N];  // Time-varying intercept for each individual
    vector[J] beta[N, val_Nt + test_Nt]; // Time-varying weights for each time step and individual
    real<lower=0> sigma[N];          // Observation noise standard deviation for each individual
    real<lower=0, upper=1> tau;      // Random walk noise for alpha
    real<lower=0, upper=1> eta;      // Random walk noise for beta
  }
  
  model {
    // Priors
    for (n in 1:N) {
      mu[n] ~ dirichlet(rep_vector(1.0, J));  // Dirichlet prior for initial model weights
      sigma[n] ~ normal(0, 1e-1);
    }
    tau ~ normal(0, 1e-1);
    eta ~ normal(0, 1e-1);
  
    for (n in 1:N) {
      // Random walk prior for alpha (intercept)
      alpha[n, 1] ~ normal(0, tau);
      for (t in 2:(val_Nt + test_Nt)) {
        alpha[n, t] ~ normal(alpha[n, t - 1], tau);
      }
  
      // Random walk prior for beta (weights)
      beta[n, 1] ~ normal(mu[n], eta);
      for (t in 2:(val_Nt + test_Nt)) {
        beta[n, t] ~ normal(beta[n, t - 1], eta);
      }
  
      // Likelihood for validation data
      for (t in 1:val_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = val_f[n, t, j];
        }
        val_y[n, t] ~ normal(alpha[n, t] + dot_product(beta[n, t], model_predictions), sigma[n]);
      }
    }
  }
  
  generated quantities {
    real log_lik[N, val_Nt];         // Log-likelihood for PSIS-LOO
    real test_y_pred[N, test_Nt];    // Predicted values for test data
    real SSE;                        // Sum of Squared Errors (single scalar)
    real RMSE;                       // Root Mean Square Error (single scalar)
    real pi_t;
    real gamma;

    SSE = 0;
    gamma = 0.1;
    for (n in 1:N) {
      // Log-likelihood for validation data
      for (t in 1:val_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = val_f[n, t, j];
        }
        pi_t = 1 + gamma - square(1 - 1.0 * t / val_Nt);
        log_lik[n, t] = pi_t * normal_lpdf(val_y[n, t] | alpha[n, t] + dot_product(beta[n, t], model_predictions), sigma[n]);
      }
  
      // Predictions for test data
      for (t in 1:test_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = test_f[n, t, j];
        }
        test_y_pred[n, t] = normal_rng(alpha[n, val_Nt + t] + dot_product(beta[n, val_Nt + t], model_predictions), sigma[n]);
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
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  
  # Extract time-varying weights
  weights <- extract(fit, pars = "beta")$beta
  
  # Convert weights to a data frame for visualization
  weights_df <- as.data.frame(weights)
  dimnames(weights_df) <- list(
    iterations = NULL,
    individuals = 1:dim(weights)[1],
    time_steps = 1:dim(weights)[2],
    models = paste0("Model_", 1:dim(weights)[3])
  )
  
  # Compute average weights across all iterations
  average_weights <- apply(weights, c(2, 3), mean)  # Average over iterations
  
  # Print the average weights (per individual and model)
  cat("\nAverage weights for each individual and model:\n")
  print(average_weights)
  
  # Reshape weights for plotting
  library(reshape2)
  weights_long <- melt(weights, varnames = c("Individual", "Time", "Model"), value.name = "Weight")
  
  # Plot time-varying weights
  library(ggplot2)
  ggplot(weights_long, aes(x = Time, y = Weight, color = as.factor(Model))) +
    geom_line() +
    facet_wrap(~ Individual, scales = "free_y") +
    labs(title = "Time-Varying Weights for Each Individual",
         x = "Time",
         y = "Weight",
         color = "Model") +
    theme_minimal()
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    fit = fit,
    log_lik = log_lik,
    loo_result = loo_result,
    test_rmse = mean(extract(fit, pars = "RMSE")$RMSE),
    weights = weights_df
  ))
}
