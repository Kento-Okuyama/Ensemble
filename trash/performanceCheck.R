# rstanパッケージのロード
library(rstan)
# 事前に定義したStanモデルコード（例: 'ar1_model.stan'）を使用
model_code <- "
data {
  int<lower=1> N; // データ点数
  vector[N] y;    // 観測データ
}
parameters {
  real phi;       // 自己回帰係数
  real<lower=0> sigma; // ノイズの標準偏差
}
model {
  phi ~ normal(0, 1); // phiの事前分布
  sigma ~ cauchy(0, 2.5); // sigmaの事前分布
  for (n in 2:N) {
    y[n] ~ normal(phi * y[n-1], sigma); // AR(1)モデル
  }
}
"
# データの準備
N <- 100 # 仮のデータポイント数
y <- rnorm(N) # 仮のデータ生成
data_list <- list(N = N, y = y)

# モデルのコンパイル
stan_model <- stan_model(model_code = model_code)

# 異なるチェーン数でMCMCサンプリングを実行し、計算時間と収束性を評価
chain_nums <- c(2, 4, 6, 8) # 試すチェーンの数
results <- list()

for (chains in chain_nums) {
  start_time <- Sys.time()
  fit <- sampling(stan_model, data = data_list, chains = chains, iter = 4000, warmup = 2000)
  end_time <- Sys.time()
  
  # 収束診断の指標を取得
  summary_fit <- summary(fit)
  rhat <- summary_fit$summary[,"Rhat"]
  ess <- summary_fit$summary[,"n_eff"]
  
  # 結果の保存
  results[[as.character(chains)]] <- list(
    chains = chains,
    duration = difftime(end_time, start_time, units = "secs"),
    rhat = rhat,
    ess = ess
  )
}

# 結果の表示
for (chains in names(results)) {
  cat("チェーン数:", results[[chains]]$chains, "\n",
      "所要時間:", results[[chains]]$duration, "\n",
      "Rhat平均:", mean(unlist(results[[chains]]$rhat)), "\n",
      "ESS最小値:", min(unlist(results[[chains]]$ess)), "\n\n")
}
