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
    real val_e[N, val_Nt, J];      // Validation model predictions (latent)
    real test_y[N, test_Nt];       // Test observed data
    real test_e[N, test_Nt, J];    // Test model predictions (latent)
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
          weighted_prediction += w[n, j] * val_e[n, t, j];
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
          weighted_prediction += w[n, j] * val_e[n, t, j];
        }
        pi_t = 1 + gamma - square(1 - 1.0 * t / val_Nt);
        log_lik[n, t] = pi_t * normal_lpdf(val_y[n, t] | weighted_prediction, sigma);
      }
  
      // Predictions for test data and calculate RMSE
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = 0;
        for (j in 1:J) {
          test_y_pred[n, t] += normal_rng(w[n, j] * test_e[n, t, j], sigma);
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
  # Extract log-likelihood for PSIS-LOO cross-validation
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  
  # Print LOO results
  # print(loo_result)
  
  # Extract test RMSE
  test_rmse <- mean(extract(fit, pars = "RMSE")$RMSE)
  # cat("\n", "Average RMSE for each individual:", test_rmse, "\n")
  
  # ===========================
  #    Visualize Posterior Distributions
  # ===========================
  # Posterior distributions of stacking weights for all individuals
  post_w <- bayesplot::mcmc_areas(as.array(fit), regex_pars = "w\\[\\d+\\,\\d+\\]") +
    labs(title = "Posterior Distributions of Stacking Weights for All Individuals",
         x = "Weight",
         y = "Density")
  
  # Prepare plot list
  post_plot <- list(post_w = post_w)
  
  # ===========================
  #    Return Results
  # ===========================
  return(list(
    fit = fit,
    log_lik = log_lik,
    loo_result = loo_result,
    test_rmse = test_rmse,
    post_plot = post_plot
  ))
}
