# 必要なライブラリをインストールしてロード
library(torch)

# ----------------------------------------------------------------
# Part 1: カルマンフィルター
# ----------------------------------------------------------------
kalman_filter_torch <- function(Y1, precomputed_eta2, b0, B1, b2, Lambda1, Q, R, eta1_i0_0, P_i0_0) {
  dims <- Y1$shape
  # [修正] L2の次元をb2から取得するように変更
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
cat("--- 1. Generating Simulation Data (User Specified Function Version) ---\n")

N <- 50; Nt <- 25; O1 <- 6; L1 <- 2; L2 <- 2

b0_true <- torch_tensor(c(0.1, -0.1))$reshape(c(L1, 1))
B1_true <- torch_tensor(matrix(c(0.7, 0.1, -0.1, 0.6), L1, L1), dtype = torch_float())
b2_true <- torch_tensor(matrix(c(0.5, 0.0, 0.2, 0.4), L1, L2), dtype = torch_float())
Lambda1_true <- torch_tensor(matrix(c(1, 0.8, 0.6, 0, 0, 0, 0, 0, 0, 1, 1.2, 0.8), O1, L1, byrow = FALSE), dtype = torch_float())
Q_true <- torch_eye(L1) * 0.5
R_true <- torch_eye(O1) * 1.0

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
## Part 3 & 4: Adamオプティマイザーによるパラメータ推定と検証（複数回実行）
## ----------------------------------------------------------------
num_runs <- 5      # 試行回数を5回に増やす
learning_rate <- 0.01
num_epochs <- 1000 # エポック数を1000に増やす
l2_decay <- 0.01 

# --- 最良の結果を保存するための変数を初期化 ---
best_loss <- Inf
best_run <- -1
best_params <- list()

# --- この外側のループで、毎回パラメータが初期化される ---
for (run in 1:num_runs) {
  
  cat(sprintf("\n--- Starting Run: %d / %d ---\n", run, num_runs))
  
  # --- Part 3: パラメータ初期化 ---
  # (B1のInformative Initializationを含む初期化コードはそのまま)
  # ... (b0, B1, b2, lambda1_free_params の初期化) ...
  
  # （元のコードから抜粋・簡略化）
  b0 <- torch_randn(L1, 1, requires_grad = TRUE)
  B1_init_matrix <- matrix(runif(L1*L1, -0.3, 1.0), L1, L1); diag(B1_init_matrix) <- runif(L1, 0.5, 1.0)
  B1 <- torch_tensor(B1_init_matrix, dtype = torch_float(), requires_grad = TRUE)
  b2 <- torch_randn(L1, L2, requires_grad = TRUE)
  lambda1_free_params <- torch_randn(4, requires_grad = TRUE)
  
  Q <- Q_true$clone()
  R <- R_true$clone()
  eta1_i0_0 <- torch_zeros(L1, 1)
  P_i0_0 <- torch_eye(L1) * 1e3
  
  optimizer <- optim_adamw(list(b0, B1, b2, lambda1_free_params), lr = learning_rate, weight_decay = l2_decay)
  
  cat(paste("Starting optimization with AdamW. Epochs:", num_epochs, "LR:", learning_rate, "Weight decay:", l2_decay, "\n"))
  
  # --- トレーニングループ ---
  for (epoch in 1:num_epochs) {
    # ... (Lambda1の再構築、optimizer$zero_grad(), log_likelihood計算、loss$backward(), optimizer$step() はそのまま) ...
    Lambda1 <- torch_zeros(O1, L1, device = Y1_generated$device)
    Lambda1[1, 1] <- 1; Lambda1[4, 2] <- 1
    Lambda1[2, 1] <- lambda1_free_params[1]; Lambda1[3, 1] <- lambda1_free_params[2]
    Lambda1[5, 2] <- lambda1_free_params[3]; Lambda1[6, 2] <- lambda1_free_params[4]
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
  
  # --- 今回の実行結果が過去最高かチェック ---
  if (loss$item() < best_loss) {
    cat(sprintf("!!! New best result found in run %d !!!\n", run))
    best_loss <- loss$item()
    best_run <- run
    
    # detach()とclone()で、勾配情報から切り離した値のコピーを保存
    best_params$b0 <- b0$detach()$clone()
    best_params$B1 <- B1$detach()$clone()
    best_params$b2 <- b2$detach()$clone()
    best_params$lambda1_free_params <- lambda1_free_params$detach()$clone()
  }
}


# --- Part 4: 全実行のうち、最も良かった結果を検証 ---
cat(sprintf("\n--- Verifying Best Result (from Run %d with Loss %.4f) ---\n", best_run, best_loss))

cat("True b0:\n"); print(as.matrix(b0_true))
cat("Estimated b0 (Best):\n"); print(as.matrix(best_params$b0))
cat("---\n")

cat("True B1:\n"); print(as.matrix(B1_true))
cat("Estimated B1 (Best):\n"); print(as.matrix(best_params$B1))
cat("---\n")

cat("True b2:\n"); print(as.matrix(b2_true))
cat("Estimated b2 (Best):\n"); print(as.matrix(best_params$b2))
cat("---\n")

Lambda1_estimated <- torch_zeros(O1, L1)
Lambda1_estimated[1, 1] <- 1
Lambda1_estimated[4, 2] <- 1
Lambda1_estimated[2, 1] <- best_params$lambda1_free_params[1]
Lambda1_estimated[3, 1] <- best_params$lambda1_free_params[2]
Lambda1_estimated[5, 2] <- best_params$lambda1_free_params[3]
Lambda1_estimated[6, 2] <- best_params$lambda1_free_params[4]

cat("True Lambda1:\n"); print(as.matrix(Lambda1_true))
cat("Estimated Lambda1 (Best):\n"); print(as.matrix(Lambda1_estimated))
cat("---\n")