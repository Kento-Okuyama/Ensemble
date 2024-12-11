library_load <- function(N, Nt) {
  # install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
  # Load necessary libraries
  library(rstan)
  rstan_options(auto_write = TRUE)
  options(mc.cores = parallel::detectCores())
  library(bayesplot)
  library(loo)
  library(ggplot2)
  library(dplyr)
  library(cmdstanr)
}