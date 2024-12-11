fit_BPS <- function(data, iter = 2000, chains = 4) {
  
  # Stan model code for BPS
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
    vector[train_Nt + test_Nt] alpha;         // Time-varying intercept
    vector[J] beta[train_Nt + test_Nt];         // Time-varying weights for each time step
    real<lower=0> sigma;      // Observation noise standard deviation
    real<lower=0, upper=1> tau;      // Random walk noise standard deviation (alpha)
    real<lower=0, upper=1> eta;      // Random walk noise standard deviation (beta)
  }
  
  transformed parameters {
    vector[J] mu;
    mu = rep_vector(1.0/J, J);
  }
  model {
    // Priors
    sigma ~ cauchy(0, 2);
    tau ~ normal(0, 0.5);
    eta ~ normal(0, 0.3);
    
    alpha[1] ~ normal(0, tau);
    beta[1] ~ normal(mu, eta);
    
    for (t in 2:(train_Nt + test_Nt)) {
      alpha[t] ~ normal(alpha[t-1], tau);  // gaussian prior for intercepts
      beta[t] ~ normal(beta[t-1], eta);  // gaussian prior for weights
    }
  
    // Likelihood
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = train_f[n, t, j];
        }
        train_y[n, t] ~ normal(alpha[t] + dot_product(beta[t], model_predictions), sigma);
      }
    }
  }
  
  generated quantities {
    matrix[N, train_Nt] log_lik;  // Log-likelihood for PSIS-LOO
    matrix[N, test_Nt] test_y_pred; // Predicted values for test data
    real SSE;
    real RMSE; 

    SSE = 0;
    for (n in 1:N) {
      for (t in 1:train_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = train_f[n, t, j];
        }
        log_lik[n, t] = normal_lpdf(train_y[n, t] | alpha[t] + dot_product(beta[t], model_predictions), sigma);
      }
      
      for (t in 1:test_Nt) {
        vector[J] model_predictions;
        for (j in 1:J) {
          model_predictions[j] = test_f[n, t, j];
        }
        test_y_pred[n, t] = normal_rng(alpha[train_Nt + t] + dot_product(beta[train_Nt + t], model_predictions), sigma);
        SSE += square(test_y[n, t] - test_y_pred[n, t]);
      }
    }
    RMSE = sqrt(SSE / (N * test_Nt)); 
  }
  "
  
  # Compile and fit BPS model
  fit <- stan(model_code = stan_code, data = data, iter = iter, chains = chains)
  
  # Extract log-likelihood for PSIS-LOO cross-validation
  log_lik <- extract_log_lik(fit, parameter_name = "log_lik", merge_chains = FALSE)
  loo_result <- loo(log_lik, moment_match = TRUE)
  print(loo_result)
  
  # Extract RMSE
  test_rmse <- mean(extract(fit, "RMSE")$RMSE)
  cat("\n", "Average RMSE across all samples:", test_rmse, "\n")
  
  # Plot posterior distributions of stacking weights
  post_alpha_beta <- bayesplot::mcmc_areas(as.array(fit), pars = c("alpha[1]", "beta[1,1]", "beta[2,1]", "beta[3,1]", "beta[4,1]")) +
    labs(title = "Posterior Distributions of model intercept and weights")
  
  post_plot <- list(post_alpha_beta = post_alpha_beta)

  return (list(fit = fit, log_lik = log_lik, loo_result = loo_result, test_rmse = test_rmse, post_plot = post_plot))
}
