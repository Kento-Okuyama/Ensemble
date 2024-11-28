# 必要なライブラリ
# install.packages("rstan")
# install.packages("loo")
library(rstan)
library(loo)

# Stanモデルのサンプル: 簡単な状態遷移モデル
state_switching_model1 <- "
data {
  int<lower=0> N;         // Number of samples
  int<lower=0> T;         // Number of time points
  matrix[N, T] y;         // Observed data
  vector[N] y2;           // Individual-specific covariates
  int S0[N];              // Initial state
}
parameters {
  real c[2];              // Intercepts for each state
  real phi[2];            // Autoregressive coefficients for each state
  real<lower=0> sigma[2]; // Standard deviation for each state
  matrix[N, T] alpha;     // State transition parameters
}
model {
  // Priors
  c ~ normal(0, 1);
  phi ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  
  // Likelihood
  for (i in 1:N) {
    for (t in 2:T) {
      // Transition probability from previous time step
      real p1 = inv_logit(alpha[i, t-1]);
      
      // Observations based on the state
      if (S0[i] == 0) {
        y[i, t] ~ normal(c[1] + phi[1] * y[i, t-1], sigma[1]);
      } else {
        y[i, t] ~ normal(c[2] + phi[2] * y[i, t-1], sigma[2]);
      }
    }
  }
}
generated quantities {
  // Generated states for posterior analysis
  int S_it[N, T];
  for (i in 1:N) {
    for (t in 2:T) {
      real p1 = inv_logit(alpha[i, t-1]);
      S_it[i, t] = bernoulli_rng(p1);
    }
  }
}
"


# Stanモデルのサンプル: 簡単な状態遷移モデル
state_switching_model2 <- "
data {
  int<lower=0> N;         // Number of samples
  int<lower=0> T;         // Number of time points
  matrix[N, T] y;         // Observed data
  vector[N] y2;           // Individual-specific covariates
  int S0[N];              // Initial state
}
parameters {
  real c[2];              // Intercepts for each state
  real phi[2];            // Autoregressive coefficients for each state
  real<lower=0> sigma[2]; // Standard deviation for each state
  matrix[N, T] alpha;     // State transition parameters
}
model {
  // Priors
  c ~ normal(0, 1);
  phi ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  
  // Likelihood
  for (i in 1:N) {
    for (t in 2:T) {
      // Transition probability from previous time step
      real p1 = inv_logit(alpha[i, t-1]);
      
      // Observations based on the state
      if (S0[i] == 0) {
        y[i, t] ~ normal(c[1] + phi[1] * y[i, t-1], sigma[1]);
      } else {
        y[i, t] ~ normal(c[2] + phi[2] * y[i, t-1], sigma[2]);
      }
    }
  }
}
generated quantities {
  // Generated states for posterior analysis
  int S_it[N, T];
  for (i in 1:N) {
    for (t in 2:T) {
      real p1 = inv_logit(alpha[i, t-1]);
      S_it[i, t] = bernoulli_rng(p1);
    }
  }
}
"

# Stanモデルのコンパイル
stan_model1 <- stan_model(model_code = state_switching_model1)
stan_model2 <- stan_model(model_code = state_switching_model2)

# データの準備
stan_data <- list(
  N = N,          # サンプル数
  T = Nt,         # 時間の数
  y = y_1it,      # 観測データ
  y2 = y_2i,      # 個別の共変量
  S0 = S_i0       # 初期状態
)

# モデル1のMCMCサンプリング
fit1 <- sampling(stan_model1, data = stan_data, iter = 1000, chains = 4)
# モデル2のMCMCサンプリング
fit2 <- sampling(stan_model2, data = stan_data, iter = 1000, chains = 4)

# LOO-CV (Leave-One-Out Cross Validation)による対数尤度の推定
log_lik1 <- extract_log_lik(fit1)
log_lik2 <- extract_log_lik(fit2)

# looパッケージを使ったLOOの計算
loo1 <- loo(log_lik1)
loo2 <- loo(log_lik2)

# 重みの計算（BHSの部分）
weights <- stacking_weights(list(loo1, loo2))
print(weights)  # モデルの重みを表示

# 結果をプロット
plot(final_pred, type = "l", main = "Final Predictions with BHS", ylab = "Predicted Values")
