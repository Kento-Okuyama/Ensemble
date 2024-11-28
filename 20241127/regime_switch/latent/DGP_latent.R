DGP_latent <- function(N = 5, Nt = 30, seed = 123, train_ratio = 0.8) {
  # Set the seed for reproducibility
  set.seed(seed)
  
  # Parameters for data simulation
  balance <- 0.5           # Weighting factor for target_mean calculation
  mu <- c(2, -2, 1, 0)     # Means for each regime
  ar_pos <- 0.7            # Positive AR coefficient for regime 1
  ar_neg <- -0.7           # Negative AR coefficient for regime 2
  ma_coef <- 0.5           # MA coefficient for regime 3
  lambda <- 1              # Factor loading for eta -> y
  sigma_m <- 1             # Measurement noise standard deviation
  sigma_s <- c(0.5, 0.7, 0.6, 0.4) # Standard deviations for each regime
  
  # Define transition matrix for regime switching
  # Each row sums to 1, indicating valid probabilities
  transition_matrix <- matrix(c(
    0.8, 0.05, 0.1, 0.05,
    0.1, 0.8, 0.05, 0.05,
    0.1, 0.05, 0.8, 0.05,
    0.1, 0.05, 0.1, 0.75
  ), nrow = 4, byrow = TRUE)
  
  # Initialize matrices to store simulated data
  eta <- matrix(NA, nrow = N, ncol = Nt)      # Latent states
  y <- matrix(NA, nrow = N, ncol = Nt)        # Observed data
  regime <- matrix(NA, nrow = N, ncol = Nt)   # Regime labels
  
  # Simulate data with regime transitions
  for (n in 1:N) {
    # Initialize the first time step
    current_regime <- 1  # Start with regime 1
    regime[n, 1] <- current_regime # Record the regime
    eta[n, 1] <- rnorm(1, mu[current_regime], sigma_s[current_regime]) # Initial latent state
    y[n, 1] <- lambda * eta[n, 1] + rnorm(1, 0, sigma_m) # Initial observed data
    
    # Simulate the remaining time steps
    for (t in 2:Nt) {
      # Sample next regime based on the transition probabilities
      current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
      regime[n, t] <- current_regime # Record the current regime
      
      # Calculate target mean for eta based on the balance parameter
      target_mean <- balance * mu[current_regime] + (1 - balance) * eta[n, t - 1]
      
      # Generate eta based on the current regime
      if (current_regime == 1) {
        # Regime 1: AR(1) with positive coefficient
        eta[n, t] <- (1 - ar_pos) * target_mean + ar_pos * eta[n, t - 1] +
          rnorm(1, 0, sigma_s[current_regime])
      } else if (current_regime == 2) {
        # Regime 2: AR(1) with negative coefficient
        eta[n, t] <- (1 - ar_neg) * target_mean + ar_neg * eta[n, t - 1] +
          rnorm(1, 0, sigma_s[current_regime])
      } else if (current_regime == 3) {
        # Regime 3: MA(1) process
        eta[n, t] <- target_mean + ma_coef * (eta[n, t - 1] - target_mean) +
          rnorm(1, 0, sigma_s[current_regime])
      } else {
        # Regime 4: Pure white noise
        eta[n, t] <- rnorm(1, target_mean, sigma_s[current_regime])
      }
      
      # Generate observed data y from eta
      y[n, t] <- lambda * eta[n, t] + rnorm(1, 0, sigma_m)
    }
  }
  
  # Split data into training and testing sets
  train_length <- floor(Nt * train_ratio) # Number of training time points
  test_length <- Nt - train_length        # Number of testing time points
  
  # Training and testing data
  train_eta <- eta[, 1:train_length]       # Training latent factor
  train_y <- y[, 1:train_length]           # Training observed data
  train_regime <- regime[, 1:train_length] # Training regime data
  test_eta <- eta[, (train_length + 1):Nt] # Testing latent factor
  test_y <- y[, (train_length + 1):Nt]     # Testing observed data
  test_regime <- regime[, (train_length + 1):Nt] # Testing regime data
  
  # Prepare data list for Stan
  stan_data <- list(
    N = N,                     # Number of time series
    train_Nt = train_length,   # Length of training time series
    test_Nt = test_length,     # Length of testing time series
    train_eta = train_eta,     # Training latent factor
    test_eta = test_eta,       # Testing latent factor
    train_y = train_y,         # Training observed data
    test_y = test_y,           # Testing observed data
    train_regime = train_regime, # Training regimes
    test_regime = test_regime   # Testing regimes
  )
  
  return(stan_data)
}
