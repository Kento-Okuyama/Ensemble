DGP <- function(Nt = 30, seed = 123, train_ratio = 0.6, val_ratio = 0.2) {
  # ===========================
  #    Initialize Parameters
  # ===========================
  
  set.seed(seed)  # Set seed for reproducibility
  
  # Model parameters
  balance <- 0.5             # Weighting factor for target_mean calculation
  mu <- c(0, 0, 0)           # Means for each regime
  ar_coef <- 0.7             # Positive AR coefficient for regime 1
  ma_coef <- 0.5             # MA coefficient for regime 2
  lambda <- c(1, 0.8, 1.2)   # Factor loading for eta -> y
  sigma_m <- 3               # Measurement noise standard deviation
  sigma_s <- c(1, 2, 3)      # Standard deviations for each regime
  
  # Dynamic transition probabilities
  alpha <- matrix(c(5, 1, 0.5, 
                    1, 5, 0.5, 
                    0.3, 0.2, 0.6), nrow = 3, byrow = TRUE)  # Base probabilities
  beta <- 2 * matrix(c(0.5, -0.5, 0, 
                   0.5, -0.5, 0, 
                   0.5, -0.5, 0), nrow = 3, byrow = TRUE)  # Dynamic adjustments
  
  # Normalize function ensures probabilities sum to 1
  normalize <- function(values) {
    exp(values) / sum(exp(values))  # normalization
  }
  
  # Function to calculate dynamic transition matrix based on eta
  calculate_transition_matrix <- function(eta, alpha, beta) {
    transition_matrix <- matrix(NA, nrow = 3, ncol = 3)
    for (i in 1:3) {
      p1 <- alpha[i, 1] + beta[i, 1] * eta  # Transition probability to state 1
      p2 <- alpha[i, 2] + beta[i, 2] * eta  # Transition probability to state 2
      p3 <- alpha[i, 3] + beta[i, 3] * eta  # Transition probability to state 3
      transition_matrix[i, ] <- normalize(c(p1, p2, p3))
    }
    return(transition_matrix)
  }
  
  # ===========================
  #    Initialize Variables
  # ===========================
  
  eta <- rep(NA, Nt)          # Latent state vector
  y <- rep(NA, Nt)            # Observed data vector
  regime <- rep(NA, Nt)       # Regime labels
  
  # Initialize first time step
  current_regime <- sample(1:2, 1, prob = c(1/2, 1/2))
  regime[1] <- current_regime
  eta[1] <- rnorm(1, mu[current_regime], sigma_s[current_regime])
  y[1] <- lambda[current_regime] * eta[1] + rnorm(1, 0, sigma_m)
  
  # ===========================
  #    Simulate Time Series
  # ===========================
  
  for (t in 2:Nt) {
    # Transition probabilities
    transition_matrix <- calculate_transition_matrix(eta[t - 1], alpha, beta)
    current_transition_probs <- transition_matrix[current_regime, ]
    
    # Sample next regime
    current_regime <- sample(1:3, 1, prob = current_transition_probs)
    regime[t] <- current_regime
    
    # Calculate target mean
    target_mean <- balance * mu[current_regime] + (1 - balance) * eta[t - 1]
    
    # Generate eta based on current regime
    if (current_regime == 1) {
      eta[t] <- (1 - ar_coef) * target_mean + ar_coef * eta[t - 1] + rnorm(1, 0, sigma_s[current_regime])
    } else if (current_regime == 2) {
      eta[t] <- target_mean + ma_coef * (eta[t - 1] - target_mean) + rnorm(1, 0, sigma_s[current_regime])
    } else {
      eta[t] <- rnorm(1, target_mean, sigma_s[current_regime])
    }
    
    # Generate observed data
    y[t] <- lambda[current_regime] * eta[t] + rnorm(1, 0, sigma_m)
  }
  
  # ===========================
  #    Split Data
  # ===========================
  
  train_length <- floor(Nt * train_ratio)
  val_length <- floor(Nt * val_ratio)
  test_length <- Nt - train_length - val_length
  
  train_eta <- eta[1:train_length]
  train_y <- y[1:train_length]
  train_regime <- regime[1:train_length]
  
  val_eta <- eta[(train_length + 1):(train_length + val_length)]
  val_y <- y[(train_length + 1):(train_length + val_length)]
  val_regime <- regime[(train_length + 1):(train_length + val_length)]
  
  test_eta <- eta[(train_length + val_length + 1):Nt]
  test_y <- y[(train_length + val_length + 1):Nt]
  test_regime <- regime[(train_length + val_length + 1):Nt]
  
  # ===========================
  #    Prepare Output
  # ===========================
  
  stan_data <- list(
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
