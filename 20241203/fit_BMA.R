fit_BMA <- function(data, iter = 2000, chains = 4) {
  
  # Stan model for transition matrix and weights
  stan_code <- "
  
  data {
    int<lower=1> val_Nt;
    int<lower=1> test_Nt;
    int<lower=1> J;                   // Number of models
    real val_y[val_Nt];        // Validation observed data
    real val_f[val_Nt, J];     // Validation model predictions
    real test_y[test_Nt];          // Test observed data
    real test_f[test_Nt, J];       // Test model predictions
  }
  
  parameters {
    simplex[J] w;                     // Model weights
    real<lower=0> sigma;              // Observation noise
  }

  model {
    // Priors
    w ~ dirichlet(rep_vector(1.0, J));
    sigma ~ normal(0, 1);
  
    // Likelihood
    for (t in 1:val_Nt) {
      real weighted_prediction = 0;
      for (j in 1:J) {
        weighted_prediction += w[j] * val_f[t, j];
      }
      val_y[t] ~ normal(weighted_prediction, sigma);
    }

  }
  
  generated quantities {
    real log_lik[val_Nt];
    real test_y_pred[test_Nt]; // Predicted values for test data
    real SSE;
    real RMSE;
    
    SSE = 0;
    for (t in 1:val_Nt) {
      real weighted_prediction = 0;
      for (j in 1:J) {
        weighted_prediction += w[j] * val_f[t, j];
      }
      log_lik[t] = normal_lpdf(val_y[t] | weighted_prediction, sigma);
    }
    
    for (t in 1:test_Nt) {
      test_y_pred[t] = 0;
      for (j in 1:J) {
        test_y_pred[t] += normal_rng(w[j] * test_f[t, j], sigma);
      }
      SSE += square(test_y[t] - test_y_pred[t]);
    }
    RMSE = sqrt(SSE / test_Nt);
  }
  "
  
  # Fit the transition model
  fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
  
  # Extract log-likelihood for PSIS-LOO cross-validation
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  print(loo_result)
  
  # Extract RMSE
  test_rmse <- mean(extract(fit, "RMSE")$RMSE)
  cat("\n", "Average RMSE across all samples:", test_rmse, "\n")
  
  # Plot posterior distributions of stacking weights
  post_w <- bayesplot::mcmc_areas(as.array(fit), pars = c("w[1]", "w[2]")) +
    labs(title = "Posterior Distributions of initial Weights")
  
  post_plot = list(post_w = post_w)
  
  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, test_rmse = test_rmse, post_plot = post_plot))
}