
####################################
# 潜在変数モデルの定義とデータ生成 #
####################################

# 全オブジェクトを削除
rm(list = ls())

# 必要なライブラリを読み込む
library(forecast)
library(rstan)
library(loo)
library(ggplot2)

# データ生成のためのパラメータ定義
n_people <- 10
n_burnins <- 25
n_timepoints <- 50

# 潜在変数モデルのデータ生成
set.seed(123)
F1 <- matrix(0, nrow = n_people, ncol = n_timepoints + n_burnins)
F2 <- matrix(0, nrow = n_people, ncol = n_timepoints + n_burnins)
Y1 <- array(0, dim = c(n_people, 3, n_timepoints))
Y2 <- array(0, dim = c(n_people, 3, n_timepoints))

for (i in 1:n_people) {
  F1[i, ] <- arima.sim(list(ar = c(0.5, -0.3)), n = n_burnins + n_timepoints)
  F2[i, ] <- arima.sim(list(ar = c(0.4, -0.2)), n = n_burnins + n_timepoints)
  
  for (j in 1:3) {
    Y1[i, j, ] <- F1[i, (n_burnins + 1):(n_burnins + n_timepoints)] + rnorm(n_timepoints, 0, 0.1)
    Y2[i, j, ] <- F2[i, (n_burnins + 1):(n_burnins + n_timepoints)] + rnorm(n_timepoints, 0, 0.1)
  }
}

# Stan用のデータ準備
data_list <- list(
  N_people = n_people,
  N_timepoints = n_timepoints,
  Y1 = Y1,
  Y2 = Y2
)

# Stanのオプションを設定
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


####################
# Stanモデルの定義 #
####################

# 潜在変数モデルのStanコード
# 潜在変数モデルのStanコード
stan_code <- "
data {
  int<lower=1> N_people;
  int<lower=1> N_timepoints;
  matrix[3, N_timepoints] Y1[N_people];
  matrix[3, N_timepoints] Y2[N_people];
}
parameters {
  matrix[N_timepoints, 2] F[N_people]; // 潜在変数 F1 と F2
  cholesky_factor_corr[2] L;           // 共分散行列のCholesky分解
  real<lower=0> sigma[2, 3];           // 観測ノイズの標準偏差
}
model {
  L ~ lkj_corr_cholesky(2.0); // 共分散行列の事前分布
  for (j in 1:2) {
    for (k in 1:3) {
      sigma[j, k] ~ normal(0, 1);
    }
  }
  for (i in 1:N_people) {
    for (t in 1:N_timepoints) {
      F[i, t] ~ multi_normal_cholesky(rep_vector(0, 2), diag_matrix(rep_vector(1, 2)) * L);
      for (k in 1:3) {
        Y1[i, k, t] ~ normal(F[i, t, 1], sigma[1, k]);
        Y2[i, k, t] ~ normal(F[i, t, 2], sigma[2, k]);
      }
    }
  }
}
generated quantities {
  vector[N_timepoints] log_lik[N_people];
  matrix[3, N_timepoints] y_hat1[N_people];
  matrix[3, N_timepoints] y_hat2[N_people];
  for (i in 1:N_people) {
    for (t in 1:N_timepoints) {
      log_lik[i, t] = 0;
      for (k in 1:3) {
        y_hat1[i, k, t] = normal_rng(F[i, t, 1], sigma[1, k]);
        y_hat2[i, k, t] = normal_rng(F[i, t, 2], sigma[2, k]);
        log_lik[i, t] += normal_lpdf(Y1[i, k, t] | F[i, t, 1], sigma[1, k]) +
                         normal_lpdf(Y2[i, k, t] | F[i, t, 2], sigma[2, k]);
      }
    }
  }
}
"

# Stanモデルのコンパイル
stan_model <- stan_model(model_code = stan_code)

################
# モデルの適合 #
################

# モデルの適合
fit <- sampling(stan_model, data = data_list, iter = 2000, chains = 4)

# サンプルの抽出
samples <- extract(fit)

################
# 結果の可視化 #
################

# 事後予測分布を取得
y_hat <- apply(samples$y_hat, c(2, 3), mean)

# MSEの計算
calculate_mse <- function(y_hat, y_true) {
  return(mean((y_hat - y_true)^2))
}

mse <- array(0, dim = c(n_people, n_timepoints))
for (i in 1:n_people) {
  for (t in 1:n_timepoints) {
    mse[i, t] <- calculate_mse(y_hat[i, t, ], data_list$Y1[i, t, ])
  }
}

# MSEのプロット
mse_df <- data.frame(
  Time = rep(1:n_timepoints, n_people),
  MSE = as.vector(mse),
  Person = rep(1:n_people, each = n_timepoints)
)

ggplot(mse_df, aes(x = Time, y = MSE, color = factor(Person))) +
  geom_line() +
  scale_y_log10() +
  labs(title = "MSE over Time for Each Person", x = "Time", y = "Log(MSE)") +
  theme_minimal()

