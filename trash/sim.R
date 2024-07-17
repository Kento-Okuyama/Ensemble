library(forecast)

# シミュレーション回数
n_sim <- 200

# 選択されたモデルを格納するベクトル
selected_models_ar1 <- vector("character", n_sim)
selected_models_arma22 <- vector("character", n_sim)

set.seed(123) # 再現性のためのシード設定

# AR(1)モデルのシミュレーション
for (i in 1:n_sim) {
  # AR(1)モデルでデータ生成
  sim_data_ar1 <- arima.sim(n = 100, model = list(ar = 0.9))
  
  # auto.arimaでモデル選択
  fit_ar1 <- auto.arima(sim_data_ar1)
  selected_models_ar1[i] <- paste(fit_ar1$arma, collapse = ",")
}

# ARMA(2,2)モデルのシミュレーション
for (i in 1:n_sim) {
  # ARMA(2,2)モデルでデータ生成
  sim_data_arma22 <- arima.sim(n = 100, model = list(ar = c(0.5, -0.25), ma = c(0.5, 0.5)))
  
  # auto.arimaでモデル選択
  fit_arma22 <- auto.arima(sim_data_arma22)
  selected_models_arma22[i] <- paste(fit_arma22$arma, collapse = ",")
}

# 結果のサマリー
summary_ar1 <- table(selected_models_ar1)
summary_arma22 <- table(selected_models_arma22)

print("AR(1)モデルのシミュレーション結果")
print(summary_ar1)

print("ARMA(2,2)モデルのシミュレーション結果")
print(summary_arma22)
