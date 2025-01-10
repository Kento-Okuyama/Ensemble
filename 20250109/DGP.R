DGP <- function(N = 5, Nt = 30, seed = 123, train_ratio = 0.6, val_ratio = 0.2) {
  # ===========================
  #    Initialize Parameters
  # ===========================
  
  set.seed(seed)  # Set seed for reproducibility
  
  # Model parameters
  balance <- 0.5             # Weighting factor for target_mean calculation
  mu <- c(1, -1, 0)           # Means for each regime
  ARma_coef <- c(0.6, -0.2)    # AR coefficients for ARMA(2,3) in regime 1
  arMA_coef <- c(0.5, -0.3, 0.2) # MA coefficients for ARMA(2,3) in regime 1
  ma_coef <- c(0.3, 0.2)
  sigma_m <- 1e-1               # Measurement noise standard deviation
  sigma_s <- c(1e-1, 1e-1, 5e-1)      # Standard deviations for each regime
  
  # Dynamic transition probabilities
  alpha <- matrix(c(3, 1, 0.5, 
                    1, 3, 0.5, 
                    0.3, 0.2, 0.6), nrow = 3, byrow = TRUE)  
  beta <- matrix(c(1, -1, 0, 
                   1, -1, 0, 
                   0, 0, 0), nrow = 3, byrow = TRUE)  
  
  normalize <- function(values) {
    exp(values) / sum(exp(values))  # normalization
  }
  
  calculate_transition_matrix <- function(eta, alpha, beta) {
    transition_matrix <- matrix(NA, nrow = 3, ncol = 3)
    for (i in 1:3) {
      p1 <- alpha[i, 1] + beta[i, 1] * eta  
      p2 <- alpha[i, 2] + beta[i, 2] * eta  
      p3 <- alpha[i, 3] + beta[i, 3] * eta  
      transition_matrix[i, ] <- normalize(c(p1, p2, p3))
    }  
    return(transition_matrix)
  }
  
  # ===========================
  #    Initialize Variables
  # ===========================
  
  eta <- matrix(NA, nrow = N, ncol = Nt)       # Latent state matrix
  y <- matrix(NA, nrow = N, ncol = Nt)         # Observed data matrix
  regime <- matrix(NA, nrow = N, ncol = Nt)    # Regime labels matrix
  eps_s <- matrix(NA, nrow = N, ncol = Nt)     # Structural residuals for each regime
  
  # ===========================
  #    Simulate Time Series for Each Person
  # ===========================
  
  for (p in 1:N) {
    current_regime <- sample(1:2, 1, prob = c(1/2, 1/2))
    regime[p, 1] <- current_regime
    eps_s[p, 1] <- rnorm(1, 0, sigma_s[current_regime])
    eta[p, 1] <- mu[current_regime] + eps_s[p, 1] 
    y[p, 1] <- eta[p, 1] + rnorm(1, 0, sigma_m)
    
    for (t in 2:Nt) {
      transition_matrix <- calculate_transition_matrix(eta[p, t - 1], alpha, beta)
      current_transition_probs <- transition_matrix[current_regime, ]
      current_regime <- sample(1:3, 1, prob = current_transition_probs)
      regime[p, t] <- current_regime
      target_mean <- balance * mu[current_regime] + (1 - balance) * eta[p, t - 1]
      
      eps_s[p, t] <- rnorm(1, 0, sigma_s[current_regime])
      if (current_regime == 1) {
        eta[p, t] <- (1 - sum(ARma_coef)) * target_mean + 
          ARma_coef[1] * eta[p, t - 1] + 
          ARma_coef[2] * ifelse(t > 2, eta[p, t - 2], 0) + 
          eps_s[p, t]  + 
          arMA_coef[1] * eps_s[p, t - 1] + 
          arMA_coef[2] * ifelse(t > 2, eps_s[p, t - 2], 0) + 
          arMA_coef[3] * ifelse(t > 3, eps_s[p, t - 3], 0)
      } else if (current_regime == 2) {
        eta[p, t] <- target_mean + ma_coef[1] * eps_s[p, t - 1] + 
          ma_coef[2] * ifelse(t > 2, eps_s[p, t - 2], 0) + 
          eps_s[p, t]
      } else {
        eta[p, t] <- target_mean + eps_s[p, t]
      }
      
      y[p, t] <- eta[p, t] + rnorm(1, 0, sigma_m)
    }
  }
  
  # ===========================
  #    Split Data
  # ===========================
  
  train_length <- floor(Nt * train_ratio)
  val_length <- floor(Nt * val_ratio)
  test_length <- Nt - train_length - val_length
  
  train_eta <- eta[, 1:train_length]
  train_y <- y[, 1:train_length]
  train_regime <- regime[, 1:train_length]
  
  val_eta <- eta[, (train_length + 1):(train_length + val_length)]
  val_y <- y[, (train_length + 1):(train_length + val_length)]
  val_regime <- regime[, (train_length + 1):(train_length + val_length)]
  
  test_eta <- eta[, (train_length + val_length + 1):Nt]
  test_y <- y[, (train_length + val_length + 1):Nt]
  test_regime <- regime[, (train_length + val_length + 1):Nt]
  
  # ===========================
  #    Prepare Output
  # ===========================
  
  stan_data <- list(
    N = N,
    train_Nt = train_length,
    val_Nt = val_length,
    test_Nt = test_length,
    train_eta = train_eta,
    val_eta = val_eta,
    test_eta = test_eta,
    train_y = train_y,
    val_y = val_y,
    test_y = test_y,
    train_regime = train_regime,
    val_regime = val_regime,
    test_regime = test_regime
  )
  
  return(stan_data)
}
