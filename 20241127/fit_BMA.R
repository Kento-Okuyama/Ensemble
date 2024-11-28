fit_BMA <- function(data, iter = 2000, chains = 4) {
  
  # Stan model for transition matrix and weights
  stan_code <- "
  
  data {
    int<lower=1> N;                   // Number of individuals
    int<lower=1> train_Nt;            // Time series length for training
    int<lower=1> test_Nt;             // Time series length for testing
    int<lower=1> J;                   // Number of models
    real train_y[N, train_Nt];        // Training observed data
    real train_f[N, train_Nt, J];     // Training model predictions
    real test_y[N, test_Nt];          // Test observed data
    real test_f[N, test_Nt, J];       // Test model predictions
  }
  
  parameters {
    simplex[J] w;                     // Model weights
    real<lower=0> sigma;              // Observation noise
  }

  model {
    // Priors
    w ~ dirichlet(rep_vector(1.0, J));
    sigma ~ cauchy(0, 2);
  
    // Likelihood
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        real weighted_prediction = 0;
        for (j in 1:J) {
          weighted_prediction += w[j] * train_f[n, t, j];
        }
        train_y[n, t] ~ normal(weighted_prediction, sigma);
      }
    }
  }
  
  generated quantities {
    matrix[N, train_Nt - 1] log_lik;
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE;    
    real pi_t;
    real gamma;

    SSE = 0;
    gamma = 0.1;
    
    for (n in 1:N) {
      for (t in 2:train_Nt) {
        real weighted_prediction = 0;
        for (j in 1:J) {
          weighted_prediction += w[j] * train_f[n, t, j];
        }
        pi_t = 1 + gamma - square(1 - 1.0 * t / train_Nt); 
        log_lik[n, t-1] = pi_t * normal_lpdf(train_y[n, t] | weighted_prediction, sigma);
      }
      
      for (t in 1:test_Nt) {
        test_y_pred[n, t] = 0;
        for (j in 1:J) {
          test_y_pred[n, t] += normal_rng(w[j] * test_f[n, t, j], sigma);
        }
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }
    RMSE = sqrt(SSE / (N * test_Nt));
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
  post_w <- bayesplot::mcmc_areas(as.array(fit), pars = c("w[1]", "w[2]", "w[3]")) +
    labs(title = "Posterior Distributions of initial Weights")
  
  post_plot = list(post_w = post_w)
  
  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, test_rmse = test_rmse, post_plot = post_plot))
}