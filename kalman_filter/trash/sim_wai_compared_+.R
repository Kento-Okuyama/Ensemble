#------------------- ここから追加 -------------------
# 現在のR環境（ワークスペース）をすべてクリアする
rm(list = ls())
#------------------- ここまで追加 -------------------

# 出力結果を "sim_wai_compared_result" ファイルに保存開始
sink("sim_wai_compared_result", split = TRUE)

# 必要なライブラリをインストールしてロード
# install.packages("torch")
library(torch)

# ----------------------------------------------------------------
# Part 0: モデルパラメータの定義 (論文ベースの体制スイッチモデル)
# ----------------------------------------------------------------
cat("--- 0. Defining Parameters for the Regime-Switching Model ---\n")

# 全体の設定
N <- 100
Nt <- 20
O <- 12 # 観測変数 (Alliance項目x9 + Symptom項目x3)

# --- State 1: 4因子モデル (治療初期) ---
L1_state1 <- 4 # 潜在変数の数 (Task, Goal, Bond, Symptom)

### 変更点: WAI因子間のクロスラグ効果をゼロに設定 ###
# TASK, GOAL, BOND間の非対角成分を0にする
B1_state1_matrix <- matrix(c(
  0.4,  0.0,  0.0, -0.05, # TASK(t-1) -> GOAL(t), BOND(t) への効果を0に
  0.0, 0.45,  0.0, -0.05, # GOAL(t-1) -> TASK(t), BOND(t) への効果を0に
  0.0,  0.0,  0.5,  -0.1,  # BOND(t-1) -> TASK(t), GOAL(t) への効果を0に
  -0.1, -0.1, -0.1,   0.6
), nrow = L1_state1, ncol = L1_state1, byrow = TRUE)

B1_true_state1 <- torch_tensor(B1_state1_matrix, dtype = torch_float())
Lambda1_state1_matrix <- matrix(0, nrow = O, ncol = L1_state1)
Lambda1_state1_matrix[1:3, 1] <- c(1.0, 0.8, 0.7)
Lambda1_state1_matrix[4:6, 2] <- c(1.0, 1.2, 0.8)
Lambda1_state1_matrix[7:9, 3] <- c(1.0, 0.9, 0.7)
Lambda1_state1_matrix[10:12, 4] <- c(1.0, 1.1, 0.9)
Lambda1_true_state1 <- torch_tensor(Lambda1_state1_matrix, dtype = torch_float())
b0_true_state1 <- torch_zeros(L1_state1, 1)

# --- State 2: 2因子モデル (治療後期) ---
L1_state2 <- 2
B1_state2_matrix <- matrix(c(
  0.6, -0.25,
  -0.1,  0.5
), nrow = L1_state2, ncol = L1_state2, byrow = TRUE)
B1_true_state2 <- torch_tensor(B1_state2_matrix, dtype = torch_float())
Lambda1_state2_matrix <- matrix(0, nrow = O, ncol = L1_state2)
Lambda1_state2_matrix[1:9, 1] <- c(0.9, 0.8, 0.7, 1.0, 1.1, 0.9, 0.9, 0.8, 0.7)
Lambda1_state2_matrix[10:12, 2] <- c(1.0, 1.1, 0.9)
Lambda1_true_state2 <- torch_tensor(Lambda1_state2_matrix, dtype = torch_float())

# (元のスクリプトにはありませんが、実行に必要なので追加)
Q_state1 <- torch_eye(L1_state1)
Q_state2 <- torch_eye(L1_state2)
R_true <- torch_eye(O)
gamma_intercept <- -2.5
gamma_task <- 0.1
gamma_goal <- 0.1
gamma_bond <- 0.1
gamma_symptom <- -0.2

# ----------------------------------------------------------------
# Part 1: データ生成 (体制スイッチモデル)
# ----------------------------------------------------------------
cat("\n--- 1. Generating Simulation Data from Regime-Switching Model ---\n")

Y_generated <- torch_zeros(N, Nt, O)
individual_states <- matrix(1, nrow = N, ncol = Nt)
switch_points <- rep(NA, N)
q_dist_s1 <- distr_multivariate_normal(torch_zeros(L1_state1), Q_state1)
q_dist_s2 <- distr_multivariate_normal(torch_zeros(L1_state2), Q_state2)
r_dist <- distr_multivariate_normal(torch_zeros(O), R_true)

for (i in 1:N) {
  eta_history <- torch_zeros(L1_state1, 1)
  current_state <- 1
  for (t in 1:Nt) {
    if (current_state == 1 && t > 1) {
      task_prev <- eta_history[1, 1]$item()
      goal_prev <- eta_history[2, 1]$item()
      bond_prev <- eta_history[3, 1]$item()
      symptom_prev <- eta_history[4, 1]$item()
      z <- gamma_intercept + gamma_task*task_prev + gamma_goal*goal_prev + gamma_bond*bond_prev + gamma_symptom*symptom_prev
      switch_prob <- 1 / (1 + exp(-z))
      if (runif(1) < switch_prob) {
        current_state <- 2
        switch_points[i] <- t
        wai_prev <- (task_prev + goal_prev + bond_prev) / 3
        symptom_prev_s2 <- symptom_prev
        eta_history <- torch_tensor(c(wai_prev, symptom_prev_s2))$reshape(c(L1_state2, 1))
      }
    }
    if (current_state == 1) {
      eta_mean_t <- torch_matmul(B1_true_state1, eta_history)
      noise_q <- q_dist_s1$sample()$reshape(c(L1_state1, 1))
      eta_t <- eta_mean_t + noise_q
      y_mean_t <- torch_matmul(Lambda1_true_state1, eta_t)
    } else {
      individual_states[i, t] <- 2
      eta_mean_t <- torch_matmul(B1_true_state2, eta_history)
      noise_q <- q_dist_s2$sample()$reshape(c(L1_state2, 1))
      eta_t <- eta_mean_t + noise_q
      y_mean_t <- torch_matmul(Lambda1_true_state2, eta_t)
    }
    noise_r <- r_dist$sample()$reshape(c(O, 1))
    y_t <- y_mean_t + noise_r
    Y_generated[i, t, ] <- y_t$squeeze()
    eta_history <- eta_t
  }
}
cat("Simulation data with regime-switching process generated.\n")
cat(sprintf("Number of individuals switched to State 2: %d / %d\n", sum(!is.na(switch_points)), N))
cat("Average switch time point (for those who switched):", mean(switch_points, na.rm = TRUE), "\n")

# ----------------------------------------------------------------
# Part 2: 共通の関数定義
# ----------------------------------------------------------------
cat("\n--- 2. Defining Common Functions ---\n")

kalman_filter_torch <- function(Y1, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0) {
  dims <- Y1$shape
  N <- dims[1]; Nt <- dims[2]; O1 <- dims[3]; L1 <- B1$shape[1]
  
  total_log_likelihood <- torch_tensor(0, dtype = torch_float(), device = Y1$device)
  I_mat <- torch_eye(L1, device = Y1$device)
  
  for (i in 1:N) {
    eta_prev <- eta1_i0_0$clone()
    P_prev <- P_i0_0$clone()
    y1_i <- Y1[i, , ]
    
    for (t in 1:Nt) {
      y1_it <- y1_i[t, ]$reshape(c(O1, 1))
      
      eta_pred <- b0 + torch_matmul(B1, eta_prev)
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

create_matrix_from_ranges <- function(size, diag_min, diag_max, off_diag_min, off_diag_max) {
  mat <- matrix(runif(size * size, off_diag_min, off_diag_max), nrow = size, ncol = size)
  diag(mat) <- runif(size, diag_min, diag_max)
  return(mat)
}

# --- 推定の共通設定 ---
num_runs <- 5
learning_rate <- 0.001
num_epochs <- 2000
l2_decay <- 0.01

# ----------------------------------------------------------------
# Part 3: 4因子モデルによるパラメータ推定 (簡略化版)
# ----------------------------------------------------------------
cat("\n\n--- 3. Parameter Estimation using a simplified 4-Factor Model ---\n")
L1_4fac <- 4
best_loss_4fac <- Inf
best_params_4fac <- list()
Q_est_4fac <- torch_eye(L1_4fac) * 0.5
R_est_4fac <- torch_eye(O) * 1.0

for (run in 1:num_runs) {
  cat(sprintf("\n--- Starting Run (4-Factor): %d / %d ---\n", run, num_runs))
  
  b0 <- torch_randn(L1_4fac, 1, requires_grad = TRUE)
  
  ### 変更点: B1行列の自由パラメータのみを推定対象とする ###
  b1_free_params <- torch_randn(10, requires_grad = TRUE)
  
  lambda1_free_params <- torch_randn(8, requires_grad = TRUE) # 2*4=8
  
  optimizer <- optim_adamw(list(b0, b1_free_params, lambda1_free_params), lr = learning_rate, weight_decay = l2_decay)
  
  for (epoch in 1:num_epochs) {
    optimizer$zero_grad()
    
    ### 変更点: 自由パラメータからB1行列を構築 ###
    B1 <- torch_zeros(L1_4fac, L1_4fac, device = Y_generated$device)
    # 対角成分
    B1[1,1] <- b1_free_params[1] # T->T
    B1[2,2] <- b1_free_params[2] # G->G
    B1[3,3] <- b1_free_params[3] # B->B
    B1[4,4] <- b1_free_params[4] # S->S
    # WAI -> Symptom
    B1[4,1] <- b1_free_params[5] # T->S
    B1[4,2] <- b1_free_params[6] # G->S
    B1[4,3] <- b1_free_params[7] # B->S
    # Symptom -> WAI
    B1[1,4] <- b1_free_params[8] # S->T
    B1[2,4] <- b1_free_params[9] # S->G
    B1[3,4] <- b1_free_params[10] # S->B
    
    Lambda1 <- torch_zeros(O, L1_4fac, device = Y_generated$device)
    Lambda1[1, 1] <- 1; Lambda1[2, 1] <- lambda1_free_params[1]; Lambda1[3, 1] <- lambda1_free_params[2]
    Lambda1[4, 2] <- 1; Lambda1[5, 2] <- lambda1_free_params[3]; Lambda1[6, 2] <- lambda1_free_params[4]
    Lambda1[7, 3] <- 1; Lambda1[8, 3] <- lambda1_free_params[5]; Lambda1[9, 3] <- lambda1_free_params[6]
    Lambda1[10, 4]<- 1;Lambda1[11, 4]<- lambda1_free_params[7];Lambda1[12, 4]<- lambda1_free_params[8]
    
    log_likelihood <- kalman_filter_torch(
      Y1 = Y_generated, b0 = b0, B1 = B1, Lambda1 = Lambda1,
      Q = Q_est_4fac, R = R_est_4fac,
      eta1_i0_0 = torch_zeros(L1_4fac, 1, device = Y_generated$device),
      P_i0_0 = torch_eye(L1_4fac, device = Y_generated$device) * 1e3
    )
    loss <- -log_likelihood
    
    # 対策3: lossがNaNまたはInfでないかチェック
    if (is.nan(loss$item()) || is.infinite(loss$item())) {
      cat(sprintf("Run: %d, Epoch: %d, Loss is NaN/Inf. Stopping this run.\n", run, epoch))
      break # このrunを中断
    }
    
    loss$backward()
    
    # 対策2: 勾配クリッピングを適用
    params_to_clip <- list(b0, b1_free_params, lambda1_free_params)
    torch::nn_utils_clip_grad_norm_(params_to_clip, max_norm = 1.0)
    
    optimizer$step()
    
    if (epoch %% 500 == 0 || epoch == 1) {
      cat(sprintf("Run: %d, Epoch: %d, Loss: %.4f\n", run, epoch, loss$item()))
    }
  }
  
  if (loss$item() < best_loss_4fac) {
    best_loss_4fac <- loss$item()
    best_params_4fac$b0 <- b0$detach()$clone()
    ### 変更点: b1_free_paramsを保存 ###
    best_params_4fac$b1_free_params <- b1_free_params$detach()$clone()
    best_params_4fac$lambda1_free_params <- lambda1_free_params$detach()$clone()
    cat(sprintf("!!! New best result for 4-Factor Model found in run %d !!!\n", run))
  }
}

# ----------------------------------------------------------------
# Part 4: 2因子モデルによるパラメータ推定
# ----------------------------------------------------------------
cat("\n\n--- 4. Parameter Estimation using a 2-Factor Model ---\n")
L1_2fac <- 2
best_loss_2fac <- Inf
best_params_2fac <- list()
Q_est_2fac <- torch_eye(L1_2fac) * 0.5
R_est_2fac <- torch_eye(O) * 1.0

for (run in 1:num_runs) {
  cat(sprintf("\n--- Starting Run (2-Factor): %d / %d ---\n", run, num_runs))
  
  b0 <- torch_randn(L1_2fac, 1, requires_grad = TRUE)
  B1_init_matrix <- create_matrix_from_ranges(L1_2fac, diag_min = 0.3, diag_max = 0.7, off_diag_min = -0.2, off_diag_max = 0.2)
  B1 <- torch_tensor(B1_init_matrix, dtype = torch_float(), requires_grad = TRUE)
  
  # 自由パラメータ: WAI因子(9-1=8) + Symptom因子(3-1=2) = 10
  lambda1_free_params <- torch_randn(10, requires_grad = TRUE)
  
  optimizer <- optim_adamw(list(b0, B1, lambda1_free_params), lr = learning_rate, weight_decay = l2_decay)
  
  for (epoch in 1:num_epochs) {
    optimizer$zero_grad()
    
    Lambda1 <- torch_zeros(O, L1_2fac, device = Y_generated$device)
    # Factor 1: Integrated WAI (9 items)
    Lambda1[1, 1] <- 1
    Lambda1[2:9, 1] <- lambda1_free_params[1:8]
    # Factor 2: Symptom (3 items)
    Lambda1[10, 2] <- 1
    Lambda1[11:12, 2] <- lambda1_free_params[9:10]
    
    log_likelihood <- kalman_filter_torch(
      Y1 = Y_generated, b0 = b0, B1 = B1, Lambda1 = Lambda1,
      Q = Q_est_2fac, R = R_est_2fac,
      eta1_i0_0 = torch_zeros(L1_2fac, 1, device = Y_generated$device),
      P_i0_0 = torch_eye(L1_2fac, device = Y_generated$device) * 1e3
    )
    loss <- -log_likelihood
    
    # 対策3: lossがNaNまたはInfでないかチェック
    if (is.nan(loss$item()) || is.infinite(loss$item())) {
      cat(sprintf("Run: %d, Epoch: %d, Loss is NaN/Inf. Stopping this run.\n", run, epoch))
      break # このrunを中断
    }
    
    loss$backward()
    
    # 対策2: 勾配クリッピングを適用
    params_to_clip <- list(b0, B1, lambda1_free_params) 
    torch::nn_utils_clip_grad_norm_(params_to_clip, max_norm = 1.0)
    
    optimizer$step()
    
    if (epoch %% 500 == 0 || epoch == 1) {
      cat(sprintf("Run: %d, Epoch: %d, Loss: %.4f\n", run, epoch, loss$item()))
    }
  }
  
  if (loss$item() < best_loss_2fac) {
    best_loss_2fac <- loss$item()
    best_params_2fac$b0 <- b0$detach()$clone()
    best_params_2fac$B1 <- B1$detach()$clone()
    best_params_2fac$lambda1_free_params <- lambda1_free_params$detach()$clone()
    cat(sprintf("!!! New best result for 2-Factor Model found in run %d !!!\n", run))
  }
}

# ----------------------------------------------------------------
# Part 5: 結果の表示とモデル比較
# ----------------------------------------------------------------
cat("\n\n--- 5. Final Results and Model Comparison ---\n\n")

# --- 4因子モデルの最良結果 ---
cat("--- Best 4-Factor Model Estimation Results ---\n")
cat(sprintf("Best Negative Log-Likelihood (Loss): %.4f\n", best_loss_4fac))

### 変更点: 保存した自由パラメータからB1行列を再構築 ###
B1_est_4fac <- torch_zeros(L1_4fac, L1_4fac)
b1_params <- best_params_4fac$b1_free_params
B1_est_4fac[1,1] <- b1_params[1]; B1_est_4fac[2,2] <- b1_params[2]; B1_est_4fac[3,3] <- b1_params[3]; B1_est_4fac[4,4] <- b1_params[4]
B1_est_4fac[4,1] <- b1_params[5]; B1_est_4fac[4,2] <- b1_params[6]; B1_est_4fac[4,3] <- b1_params[7]
B1_est_4fac[1,4] <- b1_params[8]; B1_est_4fac[2,4] <- b1_params[9]; B1_est_4fac[3,4] <- b1_params[10]

Lambda1_est_4fac <- torch_zeros(O, L1_4fac)
lambda_params <- best_params_4fac$lambda1_free_params
Lambda1_est_4fac[1, 1] <- 1; Lambda1_est_4fac[2, 1] <- lambda_params[1]; Lambda1_est_4fac[3, 1] <- lambda_params[2]
Lambda1_est_4fac[4, 2] <- 1; Lambda1_est_4fac[5, 2] <- lambda_params[3]; Lambda1_est_4fac[6, 2] <- lambda_params[4]
Lambda1_est_4fac[7, 3] <- 1; Lambda1_est_4fac[8, 3] <- lambda_params[5]; Lambda1_est_4fac[9, 3] <- lambda_params[6]
Lambda1_est_4fac[10, 4]<- 1;Lambda1_est_4fac[11, 4]<- lambda_params[7];Lambda1_est_4fac[12, 4]<- lambda_params[8]

cat("True B1 (State 1):\n"); print(as.matrix(B1_true_state1))
cat("Estimated B1 (4-Factor):\n"); print(as.matrix(B1_est_4fac))
cat("---\n")
cat("True Lambda1 (State 1):\n"); print(as.matrix(Lambda1_true_state1))
cat("Estimated Lambda1 (4-Factor):\n"); print(as.matrix(Lambda1_est_4fac))
cat("---\n")

# --- 2因子モデルの最良結果 ---
cat("\n--- Best 2-Factor Model Estimation Results ---\n")
cat(sprintf("Best Negative Log-Likelihood (Loss): %.4f\n", best_loss_2fac))

Lambda1_est_2fac <- torch_zeros(O, L1_2fac)
Lambda1_est_2fac[1, 1] <- 1
Lambda1_est_2fac[2:9, 1] <- best_params_2fac$lambda1_free_params[1:8]
Lambda1_est_2fac[10, 2] <- 1
Lambda1_est_2fac[11:12, 2] <- best_params_2fac$lambda1_free_params[9:10]

cat("True B1 (State 2):\n"); print(as.matrix(B1_true_state2))
cat("Estimated B1 (2-Factor):\n"); print(as.matrix(best_params_2fac$B1))
cat("---\n")
cat("True Lambda1 (State 2):\n"); print(as.matrix(Lambda1_true_state2))
cat("Estimated Lambda1 (2-Factor):\n"); print(as.matrix(Lambda1_est_2fac))
cat("---\n")

# --- モデル適合度の比較 ---
cat("\n--- Model Fit Comparison ---\n")
cat(sprintf("4-Factor Model Best Loss: %.4f\n", best_loss_4fac))
cat(sprintf("2-Factor Model Best Loss: %.4f\n", best_loss_2fac))

if (best_loss_4fac < best_loss_2fac) {
  cat("The 4-Factor model provides a better fit to the data based on log-likelihood.\n")
} else if (best_loss_2fac < best_loss_4fac) {
  cat("The 2-Factor model provides a better fit to the data based on log-likelihood.\n")
} else {
  cat("The models have comparable fit based on log-likelihood.\n")
}


# ファイルへの出力を終了し、コンソールに復帰
sink()