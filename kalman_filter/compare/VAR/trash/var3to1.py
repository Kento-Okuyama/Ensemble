import math
import os
import sys
import time
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm, trange
import warnings

# ----------------------------------------------------------------
# Part 0: パラメータとデータ読み込み
# ----------------------------------------------------------------
print("--- 0. Defining Parameters & Loading Data ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ★★★ DGP.pyとモードを合わせる ★★★
# 'IMPLEMENT' または 'IMPLEMENT 2.0' を選択
DGP_MODE = 'IMPLEMENT 2.0' 
print(f"Analyzing data from '{DGP_MODE}' mode.")

# --- ファイルパス ---
DGP_DIR = os.path.join('..', 'DGP')
DATA_FILE = os.path.join(DGP_DIR, f'simulation_data_{DGP_MODE}.pt')

# 出力ファイルは現在のディレクトリ(var3to1)に保存
MODEL_3FAC_FILE = f'fitted_3fac_model_{DGP_MODE}.pt'
MODEL_1FAC_FILE = f'fitted_1fac_model_{DGP_MODE}.pt'


# --- DGPパラメータの再定義（初期値設定のため） ---
if DGP_MODE == 'IMPLEMENT':
    N_dgp, Nt_dgp = 57, 15
    B1_true_state1 = torch.tensor([[0.04, 0.01, -0.11], [-0.01, 0.07, 0.13], [0.02, 0.11, 0.16]], device=device)
    B1_true_state2 = torch.tensor([[0.50]], device=device)
elif DGP_MODE == 'IMPLEMENT 2.0':
    N_dgp, Nt_dgp = 80, 19
    B1_true_state1 = torch.tensor([[0.17, -0.06, 0.00], [0.14, 0.21, -0.10], [-0.29, -0.22, 0.11]], device=device)
    B1_true_state2 = torch.tensor([[0.24]], device=device)

# 共通パラメータ
O = 9
L1_state1, L1_state2 = 3, 1
Q1_VAR, Q2_VAR, R_VAR = 0.5, 0.1, 0.5
lambda1_true_values_state1 = torch.tensor([1.2,0.8,1.1,0.9,1.3,0.7], device=device)
lambda1_true_values_state2 = torch.tensor([1.2,0.8,1.1,0.9,1.3,0.7,0.6,1.0], device=device)

# --- データ読み込み ---
if not os.path.exists(DATA_FILE):
    print(f"Error: Data file '{DATA_FILE}' not found. Please run DGP.py first in the '../DGP' directory.")
    sys.exit()

print(f"Loading data from '{DATA_FILE}'...")
saved_data = torch.load(DATA_FILE, weights_only=False)
Y_generated = saved_data['Y_generated'].to(device)
N, Nt, _ = Y_generated.shape
print("Data loaded successfully.")

#比較表示用に完全な真の行列を再構築
Lambda1_true_state1=torch.zeros(O,L1_state1,device=device)
Lambda1_true_state1[0,0]=1
Lambda1_true_state1[1,0]=lambda1_true_values_state1[0]
Lambda1_true_state1[2,0]=lambda1_true_values_state1[1]
Lambda1_true_state1[3,1]=1
Lambda1_true_state1[4,1]=lambda1_true_values_state1[2]
Lambda1_true_state1[5,1]=lambda1_true_values_state1[3]
Lambda1_true_state1[6,2]=1
Lambda1_true_state1[7,2]=lambda1_true_values_state1[4]
Lambda1_true_state1[8,2]=lambda1_true_values_state1[5]
Lambda1_true_state2=torch.zeros(O,L1_state2,device=device)
Lambda1_true_state2[0,0]=1.0
Lambda1_true_state2[1:9,0]=lambda1_true_values_state2

# ----------------------------------------------------------------
# Part 1: 共通のKalman Filter関数
# ----------------------------------------------------------------
print("\n--- 1. Defining Common Kalman Filter Functions ---")

def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape 
    L1 = B1.shape[0]
    device = Y.device
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    total_log_likelihood = 0.0
    for t in range(Nt):
        y_t = Y[:, t, :].unsqueeze(-1)
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        try:
            F_t_jitter = F_t + torch.eye(O1, device=device) * 1e-6
            dist_t = MultivariateNormal(loc=v_t.squeeze(-1), covariance_matrix=F_t_jitter)
            total_log_likelihood += dist_t.log_prob(torch.zeros_like(v_t.squeeze(-1))).sum()
        except torch.linalg.LinAlgError:
            return torch.tensor(float('nan'), device=device)
        
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        
        # ★★★ 修正点: P_updated の計算にJoseph formを使用 ★★★
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))

        eta_prev, P_prev = eta_updated, P_updated
    return total_log_likelihood

def get_per_time_point_log_likelihood(Y, b0, B1, Lambda1, Q, R):
    """各時点での対数尤度を返すKalman Filter関数"""
    N, Nt, O1 = Y.shape 
    L1 = B1.shape[0]
    device = Y.device
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    log_likelihoods_over_time = torch.zeros(N, Nt, device=device)

    for t in range(Nt):
        y_t = Y[:, t, :].unsqueeze(-1)
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        try:
            F_t_jitter = F_t + torch.eye(O1, device=device) * 1e-6
            dist_t = MultivariateNormal(loc=v_t.squeeze(-1), covariance_matrix=F_t_jitter)
            log_likelihoods_over_time[:, t] = dist_t.log_prob(torch.zeros_like(v_t.squeeze(-1)))
        except torch.linalg.LinAlgError:
            log_likelihoods_over_time[:, t] = -torch.inf
        
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)

        # ★★★ 修正点: P_updated の計算にJoseph formを使用 ★★★
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + \
                    torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))

        eta_prev, P_prev = eta_updated, P_updated
        
    return log_likelihoods_over_time

# ----------------------------------------------------------------
# Part 2: 3因子モデルのフィッティング
# ----------------------------------------------------------------
print("\n--- 2. Fitting 3-Factor Model ---")
if os.path.exists(MODEL_3FAC_FILE):
    print(f"Loading pre-trained 3-Factor model from '{MODEL_3FAC_FILE}'...")
    saved_model_3fac = torch.load(MODEL_3FAC_FILE, map_location=device)
    best_params_3fac = saved_model_3fac['params']
    logL_3fac = saved_model_3fac['logL']
else:
    print("Training 3-Factor model...")
    # --- パラメータをDGPの真値に近い値で初期化 ---
    b0_3fac = torch.zeros(L1_state1, 1, device=device, requires_grad=True)
    B1_3fac = (B1_true_state1.clone() + torch.randn_like(B1_true_state1) * 0.1).requires_grad_(True)
    lambda1_free_params_3fac = (lambda1_true_values_state1.clone() + torch.randn_like(lambda1_true_values_state1) * 0.1).requires_grad_(True)
    log_q_diag_3fac = torch.log(torch.ones(L1_state1, device=device) * Q1_VAR).requires_grad_(True)
    log_r_diag_3fac = torch.log(torch.ones(O, device=device) * R_VAR).requires_grad_(True)
    
    params_to_learn_3fac = [b0_3fac, B1_3fac, lambda1_free_params_3fac, log_q_diag_3fac, log_r_diag_3fac]
    optimizer = torch.optim.Adam(params_to_learn_3fac, lr=0.001)

    best_loss = float('inf')
    patience_counter = 0
    pbar = trange(10000, desc="[3-Fac Training]")
    for epoch in pbar:
        optimizer.zero_grad()
        Q_est = torch.diag(torch.exp(log_q_diag_3fac))
        R_est = torch.diag(torch.exp(log_r_diag_3fac))
        Lambda1_est = torch.zeros(O, L1_state1, device=device)
        Lambda1_est[0,0]=1
        Lambda1_est[1,0]=lambda1_free_params_3fac[0]
        Lambda1_est[2,0]=lambda1_free_params_3fac[1]
        Lambda1_est[3,1]=1
        Lambda1_est[4,1]=lambda1_free_params_3fac[2]
        Lambda1_est[5,1]=lambda1_free_params_3fac[3]
        Lambda1_est[6,2]=1
        Lambda1_est[7,2]=lambda1_free_params_3fac[4]
        Lambda1_est[8,2]=lambda1_free_params_3fac[5]
        
        logL = kalman_filter_torch_loss(Y_generated, b0_3fac, B1_3fac, Lambda1_est, Q_est, R_est)
        loss = -logL
        if torch.isnan(loss): break
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params_3fac = {'b0': b0_3fac.detach(), 'B1': B1_3fac.detach(), 'Lambda1': Lambda1_est.detach(), 'Q': Q_est.detach(), 'R': R_est.detach()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix(loss=loss.item(), best_loss=best_loss)
        if patience_counter >= 100:
            print("Early stopping triggered.")
            break
            
    logL_3fac = -best_loss
    torch.save({'params': best_params_3fac, 'logL': logL_3fac}, MODEL_3FAC_FILE)
    print(f"3-Factor model training finished. Best LogL: {logL_3fac:.2f}")

# ----------------------------------------------------------------
# Part 3: 1因子モデルのフィッティング
# ----------------------------------------------------------------
print("\n--- 3. Fitting 1-Factor Model ---")
if os.path.exists(MODEL_1FAC_FILE):
    print(f"Loading pre-trained 1-Factor model from '{MODEL_1FAC_FILE}'...")
    saved_model_1fac = torch.load(MODEL_1FAC_FILE, map_location=device)
    best_params_1fac = saved_model_1fac['params']
    logL_1fac = saved_model_1fac['logL']
else:
    print("Training 1-Factor model...")
    b0_1fac = torch.zeros(L1_state2, 1, device=device, requires_grad=True)
    B1_1fac = (B1_true_state2.clone() + torch.randn_like(B1_true_state2) * 0.1).requires_grad_(True)
    lambda1_free_params_1fac = (lambda1_true_values_state2.clone() + torch.randn_like(lambda1_true_values_state2) * 0.1).requires_grad_(True)
    log_q_diag_1fac = torch.log(torch.ones(L1_state2, device=device) * Q2_VAR).requires_grad_(True)
    log_r_diag_1fac = torch.log(torch.ones(O, device=device) * R_VAR).requires_grad_(True)
    
    params_to_learn_1fac = [b0_1fac, B1_1fac, lambda1_free_params_1fac, log_q_diag_1fac, log_r_diag_1fac]
    optimizer = torch.optim.Adam(params_to_learn_1fac, lr=0.01)
    
    best_loss = float('inf')
    patience_counter = 0
    pbar = trange(10000, desc="[1-Fac Training]")
    for epoch in pbar:
        optimizer.zero_grad()
        Q_est = torch.diag(torch.exp(log_q_diag_1fac))
        R_est = torch.diag(torch.exp(log_r_diag_1fac))
        Lambda1_est = torch.zeros(O, L1_state2, device=device)
        Lambda1_est[0,0] = 1.0
        Lambda1_est[1:9,0] = lambda1_free_params_1fac[0:8]
        
        logL = kalman_filter_torch_loss(Y_generated, b0_1fac, B1_1fac, Lambda1_est, Q_est, R_est)
        loss = -logL
        if torch.isnan(loss): break
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params_1fac = {'b0': b0_1fac.detach(), 'B1': B1_1fac.detach(), 'Lambda1': Lambda1_est.detach(), 'Q': Q_est.detach(), 'R': R_est.detach()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        pbar.set_postfix(loss=loss.item(), best_loss=best_loss)
        if patience_counter >= 100:
            print("Early stopping triggered.")
            break
            
    logL_1fac = -best_loss
    torch.save({'params': best_params_1fac, 'logL': logL_1fac}, MODEL_1FAC_FILE)
    print(f"1-Factor model training finished. Best LogL: {logL_1fac:.2f}")

# ----------------------------------------------------------------
# Part 4: 予測値と尤度、情報量規準の計算・集計
# ----------------------------------------------------------------
print("\n--- 4. Calculating Predictions, Likelihoods, and Information Criteria ---")

def calculate_predictions_and_likelihoods(model_name: str, params: dict, Y_data: torch.Tensor):
    print(f"\n--- Calculating for {model_name} ---")
    b0, B1, Lambda1, Q, R = params['b0'], params['B1'], params['Lambda1'], params['Q'], params['R']
    
    with torch.no_grad():
        y_pred_mean, y_pred_cov = get_kalman_predictive_distribution(Y_data, b0, B1, Lambda1, Q, R)

    predictive_dist = MultivariateNormal(loc=y_pred_mean, covariance_matrix=y_pred_cov + torch.eye(O, device=device)*1e-6)
    log_likelihoods = predictive_dist.log_prob(Y_data)

    agg_pred_mean_t = y_pred_mean.mean(dim=0)
    agg_log_likelihood_t = log_likelihoods.sum(dim=0)

    print(f"📈 Aggregated results for {model_name}:")
    for t in range(Nt):
        print(f"  Time {t+1:02d}: Aggregated Log-Likelihood = {agg_log_likelihood_t[t]:>8.2f}, "
              f"Avg. Predicted Y[0] = {agg_pred_mean_t[t, 0]:.4f}")

    return {
        'pred_mean_agg_t': agg_pred_mean_t.cpu(),
        'log_likelihood_agg_t': agg_log_likelihood_t.cpu()
    }

results_3fac = calculate_predictions_and_likelihoods("3-Factor Model", best_params_3fac, Y_generated)
results_1fac = calculate_predictions_and_likelihoods("1-Factor Model", best_params_1fac, Y_generated)

# --- 情報量規準の計算 ---
# パラメータ数 (k) の計算
k_3fac = best_params_3fac['b0'].numel() + best_params_3fac['B1'].numel() + 6 + 3 + O
k_1fac = best_params_1fac['b0'].numel() + best_params_1fac['B1'].numel() + 8 + 1 + O

# データポイント数 (n)
n_obs = N * Nt

# AIC と BIC の計算
aic_3fac = 2 * k_3fac - 2 * logL_3fac
bic_3fac = k_3fac * np.log(n_obs) - 2 * logL_3fac
aic_1fac = 2 * k_1fac - 2 * logL_1fac
bic_1fac = k_1fac * np.log(n_obs) - 2 * logL_1fac

# --- 最終比較テーブル ---
print("\n\n--- Final Model Comparison ---")
table_width = 80
print("=" * table_width)
print(f"{'Model':<20} | {'Log-Likelihood':<18} | {'AIC':<15} | {'BIC':<15}")
print("-" * table_width)
print(f"{'3-Factor Model':<20} | {logL_3fac:<18.2f} | {aic_3fac:<15.2f} | {bic_3fac:<15.2f}")
print(f"{'(k = ' + str(k_3fac) + ')':<20} | {'':<18} | {'':<15} | {'':<15}")
print(f"{'1-Factor Model':<20} | {logL_1fac:<18.2f} | {aic_1fac:<15.2f} | {bic_1fac:<15.2f}")
print(f"{'(k = ' + str(k_1fac) + ')':<20} | {'':<18} | {'':<15} | {'':<15}")
print("=" * table_width)
print("* k: Number of estimated parameters")
print("* Lower AIC/BIC indicates a better model fit for the given complexity.")

# ----------------------------------------------------------------
# Part 5: 推定されたパラメータの表示
# ----------------------------------------------------------------
print("\n--- 5. Display Estimated Parameters ---")

# --- 3-Factor Model ---
print("\n--- 3-Factor Model Estimated vs. True Parameters ---")
p3 = best_params_3fac
print(f"  b0 (Intercept): Estimated = {np.round(p3['b0'].squeeze().cpu().numpy(), 3)}, True = [0. 0. 0.]")
print("\n-- Estimated B1 (State Transition) --")
for row in p3['B1']: print(f"  [{row[0]:>6.3f} {row[1]:>6.3f} {row[2]:>6.3f}]")
print("-- True B1 --")
for row in B1_true_state1: print(f"  [{row[0]:>6.3f} {row[1]:>6.3f} {row[2]:>6.3f}]")
print(f"\n-- Estimated Q diag (System Noise): --\n  {np.round(torch.diag(p3['Q']).cpu().numpy(), 3)}")
print(f"-- True Q diag --\n  {[Q1_VAR]*L1_state1}")
print(f"\n-- Estimated R diag (Measurement Noise): --\n  {np.round(torch.diag(p3['R']).cpu().numpy(), 3)}")
print(f"-- True R diag --\n  {[R_VAR]*O}")

# --- 1-Factor Model ---
print("\n\n--- 1-Factor Model Estimated vs. True Parameters ---")
p1 = best_params_1fac
print(f"  b0 (Intercept): Estimated = {np.round(p1['b0'].squeeze().cpu().numpy(), 3)}, True = [0.]")
print(f"\n-- Estimated B1 (State Transition): --\n  {np.round(p1['B1'].cpu().numpy(), 3)}")
print(f"-- True B1 --\n  {np.round(B1_true_state2.cpu().numpy(), 3)}")
print(f"\n-- Estimated Q diag (System Noise): --\n  {np.round(torch.diag(p1['Q']).cpu().numpy(), 3)}")
print(f"-- True Q diag --\n  {[Q2_VAR]*L1_state2}")
print(f"\n-- Estimated R diag (Measurement Noise): --\n  {np.round(torch.diag(p1['R']).cpu().numpy(), 3)}")
print(f"-- True R diag --\n  {[R_VAR]*O}")


# ----------------------------------------------------------------
# Part 6: 最終的なモデル比較
# ----------------------------------------------------------------
# --- 情報量規準の計算 ---
k_3fac = p3['b0'].numel() + p3['B1'].numel() + 6 + 3 + O # b0, B1, lambda_free, Q_diag, R_diag
k_1fac = p1['b0'].numel() + p1['B1'].numel() + 8 + 1 + O # b0, B1, lambda_free, Q_diag, R_diag
n_obs = N * Nt
aic_3fac = 2 * k_3fac - 2 * logL_3fac
bic_3fac = k_3fac * np.log(n_obs) - 2 * logL_3fac
aic_1fac = 2 * k_1fac - 2 * logL_1fac
bic_1fac = k_1fac * np.log(n_obs) - 2 * logL_1fac

# --- 最終比較テーブル ---
print("\n\n--- 6. Final Model Comparison ---")
table_width = 80
print("=" * table_width)
print(f"{'Model':<20} | {'Log-Likelihood':<18} | {'AIC':<15} | {'BIC':<15}")
print("-" * table_width)
print(f"{'3-Factor Model':<20} | {logL_3fac:<18.2f} | {aic_3fac:<15.2f} | {bic_3fac:<15.2f}")
print(f"{'(k = ' + str(k_3fac) + ')':<20} | {'':<18} | {'':<15} | {'':<15}")
print(f"{'1-Factor Model':<20} | {logL_1fac:<18.2f} | {aic_1fac:<15.2f} | {bic_1fac:<15.2f}")
print(f"{'(k = ' + str(k_1fac) + ')':<20} | {'':<18} | {'':<15} | {'':<15}")
print("=" * table_width)
print("* k: Number of estimated parameters")
print("* Lower AIC/BIC indicates a better model fit for the given complexity.")

# ----------------------------------------------------------------
# Part 7: 時点ごとのアンサンブル重みの計算と可視化
# ----------------------------------------------------------------
print("\n--- 7. Calculating and Visualizing Time-Varying Ensemble Weights ---")

# --- 7a. 各時点・各個人の対数尤度を再計算 ---
with torch.no_grad():
    # 3因子モデル
    # ★★★ 修正点: 予測平均(y_pred_mean_3fac)も受け取る ★★★
    y_pred_mean_3fac, cov_3fac = get_kalman_predictive_distribution(Y_generated, **best_params_3fac)
    dist_3fac = MultivariateNormal(loc=y_pred_mean_3fac, covariance_matrix=cov_3fac + torch.eye(O, device=device)*1e-6)
    log_lik_3fac_full = dist_3fac.log_prob(Y_generated) # Shape: (N, Nt)

    # 1因子モデル
    # ★★★ 修正点: 予測平均(y_pred_mean_1fac)も受け取る ★★★
    y_pred_mean_1fac, cov_1fac = get_kalman_predictive_distribution(Y_generated, **best_params_1fac)
    dist_1fac = MultivariateNormal(loc=y_pred_mean_1fac, covariance_matrix=cov_1fac + torch.eye(O, device=device)*1e-6)
    log_lik_1fac_full = dist_1fac.log_prob(Y_generated) # Shape: (N, Nt)

# 対数尤度から尤度に変換し、個人間で平均
avg_lik_3fac = torch.exp(log_lik_3fac_full).mean(dim=0).cpu().numpy()
avg_lik_1fac = torch.exp(log_lik_1fac_full).mean(dim=0).cpu().numpy()

# --- 7b. オンラインベイズ更新で重みを計算 ---
weights = np.zeros((Nt, 2))
weights[0, :] = [0.5, 0.5] # 初期重み

for t in range(1, Nt):
    # 1. 前の時点の重みと、現在の尤度を掛け合わせる
    unnormalized_w0 = weights[t-1, 0] * avg_lik_3fac[t]
    unnormalized_w1 = weights[t-1, 1] * avg_lik_1fac[t]
    
    # 2. 合計が1になるように正規化 (ゼロ除算を回避)
    total_w = unnormalized_w0 + unnormalized_w1
    if total_w > 1e-9:
        weights[t, 0] = unnormalized_w0 / total_w
        weights[t, 1] = unnormalized_w1 / total_w
    else: # 尤度がほぼゼロになった場合、重みを維持
        weights[t, :] = weights[t-1, :]


# --- 7c. 重みの推移をプロット ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

sessions = np.arange(1, Nt + 1)
ax.stackplot(sessions, weights[:, 0], weights[:, 1], 
             labels=['Weight: 3-Factor Model', 'Weight: 1-Factor Model'],
             colors=['#3498db', '#e74c3c'], alpha=0.7)

ax.set_title(f"Time-Varying Ensemble Weights (Mode: {DGP_MODE})", fontsize=16)
ax.set_xlabel("Sessions", fontsize=12)
ax.set_ylabel("Ensemble Weight", fontsize=12)
ax.set_ylim(0, 1)
ax.set_xlim(1, Nt)
ax.legend(loc='upper left')
plt.tight_layout()

plot_filename_weights = f'ensemble_weights_{DGP_MODE}.png'
plt.savefig(plot_filename_weights)
plt.show()

print(f"\n✅ Ensemble weights plot saved as '{plot_filename_weights}'")