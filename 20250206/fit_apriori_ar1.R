# ============  AR(1) KFAS  ============ #
fit_ar_kfas <- function(data) {
  # data$train_y : N x train_Nt
  
  # 1) Reshape data for KFAS: time=rows, series=cols
  Y <- t(data$train_y)  # (train_Nt) x N
  
  N <- data$N
  train_Nt <- data$train_Nt
  val_Nt   <- data$val_Nt
  test_Nt  <- data$test_Nt
  horizon  <- val_Nt + test_Nt
  
  # 2) Build AR(1) model with common phi
  model_ar <- SSModel(
    Y ~ -1 + SSMcustom(
      Z = diag(1, nrow = N),
      T = diag(NA, nrow = N),
      R = diag(1, nrow = N),
      Q = diag(NA, nrow = N),
      state_names = paste0("state_", 1:N)
    ),
    H = diag(NA, nrow = N)
  )
  
  # 3) updatefn
  updatefn_ar <- function(pars, model) {
    # pars[1] => phi, mapped to (-1,1) via tanh
    # pars[2] => sigma_q = exp(pars[2])
    # pars[3] => sigma_h = exp(pars[3])
    phi  <- tanh(pars[1])
    sigQ <- exp(pars[2])
    sigH <- exp(pars[3])
    
    model$T[,,1] <- diag(phi, nrow = N)
    model$Q[,,1] <- diag(sigQ^2, nrow = N)
    model$H[,,1] <- diag(sigH^2, nrow = N)
    return(model)
  }
  
  # 4) fitSSM
  fit_ar <- fitSSM(
    model   = model_ar,
    inits   = c(0.0, log(0.1), log(0.1)),
    updatefn = updatefn_ar,
    method  = "BFGS"
  )
  
  # 5) Filter & Smoother (stateを指定)
  kfs_ar <- KFS(
    fit_ar$model,
    filtering = c("state","mean"),
    smoothing = c("state","mean")
  )
  
  # 6) Forecast val+test steps
  pred_ar_list <- predict(fit_ar$model, n.ahead = horizon)  # list of ts if N>1
  
  # Convert to numeric matrix: (horizon) x N
  pred_ar_mat <- do.call(cbind, lapply(pred_ar_list, as.numeric))
  colnames(pred_ar_mat) <- paste0("y", 1:N)
  
  return(list(
    fit_ar   = fit_ar,
    kfs_ar   = kfs_ar,       # kfs_ar$alphahat がちゃんと入る
    pred_ar  = pred_ar_mat   # dimension: horizon x N
  ))
}

