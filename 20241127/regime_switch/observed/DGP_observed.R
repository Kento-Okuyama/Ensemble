DGP_obs <- function(N = 5, Nt = 30, seed = 123, train_ratio = 0.8) {
  # Set the seed for reproducibility
  set.seed(seed)

  # Parameters for data simulation
  lambda <- 0.5  # Smoothing parameter for target_mean
  mu <- c(2, -2, 1, 0)  # Regime means
  ar_pos <- 0.7  # Positive AR coefficient
  ar_neg <- -0.7  # Negative AR coefficient
  ma_coef <- 0.5  # MA coefficient
  sigma <- c(0.5, 0.7, 0.6, 0.4)  # Standard deviations for each regime

  # Define transition matrix for regime switching
  transition_matrix <- matrix(c(
    0.8, 0.05, 0.1, 0.05,
    0.1, 0.8, 0.05, 0.05,
    0.1, 0.05, 0.8, 0.05,
    0.1, 0.05, 0.1, 0.75
  ), nrow = 4, byrow = TRUE)

  # Initialize matrices to store simulated data
  y <- matrix(NA, nrow = N, ncol = Nt)
  regime <- matrix(NA, nrow = N, ncol = Nt)

  # Simulate data with regime transitions
  for (n in 1:N) {
    current_regime <- 1
    y[n, 1] <- rnorm(1, mean = mu[current_regime], sd = sigma[current_regime])  # Initial value
    regime[n, 1] <- current_regime

    for (t in 2:Nt) {
      current_regime <- sample(1:4, 1, prob = transition_matrix[current_regime, ])
      regime[n, t] <- current_regime

      target_mean <- lambda * mu[current_regime] + (1 - lambda) * y[n, t - 1]

      if (current_regime == 1) {
        y[n, t] <- (1 - ar_pos) * target_mean + ar_pos * y[n, t - 1] + rnorm(1, mean = 0, sd = sigma[current_regime])
      } else if (current_regime == 2) {
        y[n, t] <- (1 - ar_neg) * target_mean + ar_neg * y[n, t - 1] + rnorm(1, mean = 0, sd = sigma[current_regime])
      } else if (current_regime == 3) {
        y[n, t] <- target_mean + ma_coef * (y[n, t - 1] - target_mean) + rnorm(1, mean = 0, sd = sigma[current_regime])
      } else {
        y[n, t] <- rnorm(1, mean = target_mean, sd = sigma[current_regime])
      }
    }
  }

  # Split data into training and testing sets
  train_length <- floor(Nt * train_ratio)
  test_length <- Nt - train_length

  train_y <- y[, 1:train_length]
  train_regime <- regime[, 1:train_length]
  test_y <- y[, (train_length + 1):Nt]
  test_regime <- regime[, (train_length + 1):Nt]

  # Prepare data list for Stan
  stan_data <- list(
    N = N,
    train_Nt = train_length,
    test_Nt = test_length,
    train_y = train_y,
    test_y = test_y,
    train_regime = train_regime,
    test_regime = test_regime
  )

  return(stan_data)
}
