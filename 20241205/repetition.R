# ===========================
#  Clear Workspace and Setup
# ===========================
rm(list = ls())  # Clear all objects from the workspace

# Set the working directory
setwd('C:/Users/kento/OneDrive - UT Cloud (1)/PhD/Ensemble/20241205')

# ===========================
#   Load External Scripts
# ===========================
# Load necessary R scripts for custom functions and models
source('library.R')       # Contains library imports and setups
source('DGP.R')           # Function for data generation process
source('fit_apriori.R')   # Model fitting function: Apriori
source('fit_BMA.R')       # Model fitting function: Bayesian Model Averaging
source('fit_BPS.R')       # Model fitting function: Bayesian Predictive Stacking
source('fit_BPS2.R')      # Model fitting function: Alternative Bayesian Predictive Stacking

# Load additional libraries using a custom function
library_load()

repetition <- function(k=1){
  seed <- 123 + k # Set the random seed for reproducibility
  
  # Parameters for data simulation
  Nt <- 100  # Length of each time series
  
  # ===========================
  #    Simulate Data
  # ===========================
  # Generate data using the custom DGP function
  df <- DGP(Nt = Nt, seed = seed, train_ratio = 0.6, val_ratio = 0.2)
  
  # Summarize the generated data's regimes
  table(c(df$train_regime, df$val_regime, df$test_regime))
  
  # Visualize the regimes across train, validation, and test sets
  plot(c(df$train_regime, df$val_regime, df$test_regime), type = "l", 
       main = "Regime Visualization", xlab = "Time", ylab = "Regime")
}