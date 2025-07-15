# 必要なライブラリをインストールしてロード
library(torch)

# ----------------------------------------------------------------
# Part 1: カルマンフィルター (この部分は変更なし)
# ----------------------------------------------------------------
kalman_filter_torch <- function(Y1, precomputed_eta2, b0, B1, b2, Lambda1, Q, R, eta1_i0_0, P_i0_0) {
  dims <- Y1$shape
  N <- dims[1]; Nt <- dims[2]; O1 <- dims[3]; L1 <- B1$shape[1]; L2 <- b2$shape[2]
  
  total_log_likelihood <- torch_tensor(0, dtype = torch_float(), device = Y1$device)
  I_mat <- torch_eye(L1, device = Y1$device)
  
  for (i in 1:N) {
    eta_prev <- eta1_i0_0$clone()
    P_prev <- P_i0_0$clone()
    y1_i <- Y1[i, , ]
    eta2_i <- precomputed_eta2[i, ]$reshape(c(L2, 1))
    
    for (t in 1:Nt) {
      y1_it <- y1_i[t, ]$reshape(c(O1, 1))
      
      eta_pred <- b0 + torch_matmul(B1, eta_prev) + torch_matmul(b2, eta2_i)
      P_pred <- torch_matmul(torch_matmul(B1, P_prev), B1$transpose(1, 2)) + Q
      
      v_it <- y1_it - torch_matmul(Lambda1, eta_pred)
      F_it <- torch_matmul(torch_matmul(Lambda1, P_pred), Lambda1$transpose(1, 2)) + R
      F_it_inv <- torch_pinverse(F_it)
      K_it <- torch_matmul(torch_matmul(P_pred, Lambda1$transpose(1, 2)), F_it_inv)
      
      eta_updated <- eta_pred + torch_matmul(K_it, v_it)
      term1 <- I_mat - torch_matmul(K_it, Lambda1)
      P_updated <- torch_matmul(torch_matmul(term1, P_pred), term1$transpose(1, 2)) + 
        torch_matmul(torch_matmul(K_it, R), K_it$transpose(1, 2))
      
      log_det_F_it <- torch_logdet(F_it)
      exponent_term <- -0.5 * torch_matmul(v_it$transpose(1, 2), torch_matmul(F_it_inv, v_it))
      log_likelihood_it <- -0.5 * O1 * log(2 * pi) - 0.5 * log_det_F_it + exponent_term
      
      total_log_likelihood <- total_log_likelihood + log_likelihood_it
      
      eta_prev <- eta_updated
      P_prev <- P_updated
    }
  }
  return(total_log_likelihood)
}

# ----------------------------------------------------------------
# Part 2: データ生成
# ----------------------------------------------------------------
cat("--- 1. Generating Simulation Data ---\n")

# [変更] 次元数を更新
N <- 50; Nt <- 25; O1 <- 9; L1 <- 3; L2 <- 2

# [変更] 真のパラメータを新しい次元数 (L1=3, O1=9) に合わせて更新
b0_true <- torch_tensor(c(0.1, -0.1, 0.05))$reshape(c(L1, 1))

B1_true_matrix <- matrix(c(0.7,  0.1, 0.0,
                           -0.1, 0.6, 0.1,
                           0.0, 0.1, 0.5), nrow = L1, ncol = L1, byrow = TRUE)
B1_true <- torch_tensor(B1_true_matrix, dtype = torch_float())

b2_true <- torch_tensor(matrix(c(0.5, 0.2, 0.0, 0.4, 0.1, 0.1), L1, L2), dtype = torch_float())

# [変更] Lambda1を9x3のブロック構造に
Lambda1_true_matrix <- matrix(
  c(1.0, 0.0, 0.0,
    0.8, 0.0, 0.0,
    0.6, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.2, 0.0,
    0.0, 0.8, 0.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 0.9,
    0.0, 0.0, 0.7),
  nrow = O1, ncol = L1, byrow = TRUE
)
Lambda1_true <- torch_tensor(Lambda1_true_matrix, dtype = torch_float())

Q_true <- torch_eye(L1) * 0.5 # 自動で3x3に
R_true <- torch_eye(O1) * 1.0 # 自動で9x9に

# データ生成プロセス (ロジックは変更なし、次元は自動で反映)
q_dist <- distr_multivariate_normal(torch_zeros(L1), Q_true)
r_dist <- distr_multivariate_normal(torch_zeros(O1), R_true)
eta2_true <- torch_randn(N, L2)
Y1_generated <- torch_zeros(N, Nt, O1)

for (i in 1:N) {
  eta1_prev <- torch_zeros(L1, 1)
  eta2_i <- eta2_true[i, ]$reshape(c(L2, 1))
  for (t in 1:Nt) {
    eta1_mean_t <- b0_true + torch_matmul(B1_true, eta1_prev) + torch_matmul(b2_true, eta2_i)
    noise_q <- q_dist$sample()$reshape(c(L1, 1))
    eta1_t <- eta1_mean_t + noise_q
    y1_mean_t <- torch_matmul(Lambda1_true, eta1_t)
    noise_r <- r_dist$sample()$reshape(c(O1, 1))
    y1_t <- y1_mean_t + noise_r
    Y1_generated[i, t, ] <- y1_t$squeeze()
    eta1_prev <- eta1_t
  }
}
cat("Simulation data generated.\n\n")

## ----------------------------------------------------------------
## Part 3 & 4: Adamオプティマイザーによるパラメータ推定と検証
## ----------------------------------------------------------------
num_runs <- 5
learning_rate <- 0.01
num_epochs <- 1000
l2_decay <- 0.01 

best_loss <- Inf
best_run <- -1
best_params <- list()

for (run in 1:num_runs) {
  cat(sprintf("\n--- Starting Run: %d / %d ---\n", run, num_runs))
  
  # --- パラメータ初期化 ---
  b0 <- torch_randn(L1, 1, requires_grad = TRUE)
  B1_init_matrix <- matrix(runif(L1*L1, -0.3, 1.0), L1, L1)
  diag(B1_init_matrix) <- runif(L1, 0.5, 1.0)
  B1 <- torch_tensor(B1_init_matrix, dtype = torch_float(), requires_grad = TRUE)
  b2 <- torch_randn(L1, L2, requires_grad = TRUE)
  
  # [変更] Lambda1の自由パラメータを6つに
  lambda1_free_params <- torch_randn(6, requires_grad = TRUE)
  
  Q <- Q_true$clone()
  R <- R_true$clone()
  eta1_i0_0 <- torch_zeros(L1, 1)
  P_i0_0 <- torch_eye(L1) * 1e3
  
  optimizer <- optim_adamw(list(b0, B1, b2, lambda1_free_params), lr = learning_rate, weight_decay = l2_decay)
  
  cat(paste("Starting optimization. Epochs:", num_epochs, "LR:", learning_rate, "\n"))
  
  # --- トレーニングループ ---
  for (epoch in 1:num_epochs) {
    # [変更] Lambda1の再構築ロジックを更新
    Lambda1 <- torch_zeros(O1, L1, device = Y1_generated$device)
    
    # 固定パラメータ (各ブロックの最初の負荷量を1に固定)
    Lambda1[1, 1] <- 1
    Lambda1[4, 2] <- 1
    Lambda1[7, 3] <- 1
    
    # 自由パラメータを割り当て
    Lambda1[2, 1] <- lambda1_free_params[1] # L1=1 -> O1=2
    Lambda1[3, 1] <- lambda1_free_params[2] # L1=1 -> O1=3
    Lambda1[5, 2] <- lambda1_free_params[3] # L1=2 -> O1=5
    Lambda1[6, 2] <- lambda1_free_params[4] # L1=2 -> O1=6
    Lambda1[8, 3] <- lambda1_free_params[5] # L1=3 -> O1=8
    Lambda1[9, 3] <- lambda1_free_params[6] # L1=3 -> O1=9
    
    optimizer$zero_grad()
    log_likelihood <- kalman_filter_torch(Y1 = Y1_generated, precomputed_eta2 = eta2_true, b0 = b0, B1 = B1, b2 = b2, Lambda1 = Lambda1, Q = Q, R = R, eta1_i0_0 = eta1_i0_0, P_i0_0 = P_i0_0)
    loss <- -log_likelihood
    loss$backward()
    optimizer$step()
    
    if (epoch %% 200 == 0 || epoch == 1) {
      cat(sprintf("Run: %d, Epoch: %d, Loss: %.4f\n", run, epoch, loss$item()))
    }
  }
  
  cat(sprintf("Optimization finished for run %d. Final Loss: %.4f\n", run, loss$item()))
  
  if (loss$item() < best_loss) {
    cat(sprintf("!!! New best result found in run %d !!!\n", run))
    best_loss <- loss$item()
    best_run <- run
    best_params$b0 <- b0$detach()$clone()
    best_params$B1 <- B1$detach()$clone()
    best_params$b2 <- b2$detach()$clone()
    best_params$lambda1_free_params <- lambda1_free_params$detach()$clone()
  }
}

# --- Part 4: 最良の結果を検証 ---
cat(sprintf("\n--- Verifying Best Result (from Run %d with Loss %.4f) ---\n", best_run, best_loss))

# [変更] Lambda1_estimatedの再構築ロジックを更新
Lambda1_estimated <- torch_zeros(O1, L1)
Lambda1_estimated[1, 1] <- 1
Lambda1_estimated[4, 2] <- 1
Lambda1_estimated[7, 3] <- 1
Lambda1_estimated[2, 1] <- best_params$lambda1_free_params[1]
Lambda1_estimated[3, 1] <- best_params$lambda1_free_params[2]
Lambda1_estimated[5, 2] <- best_params$lambda1_free_params[3]
Lambda1_estimated[6, 2] <- best_params$lambda1_free_params[4]
Lambda1_estimated[8, 3] <- best_params$lambda1_free_params[5]
Lambda1_estimated[9, 3] <- best_params$lambda1_free_params[6]

# 結果表示
cat("True b0:\n"); print(as.matrix(b0_true))
cat("Estimated b0 (Best):\n"); print(as.matrix(best_params$b0))
cat("---\n")

cat("True B1:\n"); print(as.matrix(B1_true))
cat("Estimated B1 (Best):\n"); print(as.matrix(best_params$B1))
cat("---\n")

cat("True b2:\n"); print(as.matrix(b2_true))
cat("Estimated b2 (Best):\n"); print(as.matrix(best_params$b2))
cat("---\n")

cat("True Lambda1:\n"); print(as.matrix(Lambda1_true))
cat("Estimated Lambda1 (Best):\n"); print(as.matrix(Lambda1_estimated))
cat("---\n")