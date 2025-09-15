import math
import os
import sys
import time
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm, trange
import warnings
import matplotlib.pyplot as plt

# ----------------------------------------------------------------
# Part 0: パラメータとデータ読み込み
# ----------------------------------------------------------------
print("--- 0. Defining Parameters & Loading Data ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ★★★ DGP.pyとモードを合わせる ★★★
DGP_MODE = 'IMPLEMENT' 
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
Lambda1_true_state1[0,0]=1; Lambda1_true_state1[1,0]=lambda1_true_values_state1[0]; Lambda1_true_state1[2,0]=lambda1_true_values_state1[1]
Lambda1_true_state1[3,1]=1; Lambda1_true_state1[4,1]=lambda1_true_values_state1[2]; Lambda1_true_state1[5,1]=lambda1_true_values_state1[3]
Lambda1_true_state1[6,2]=1; Lambda1_true_state1[7,2]=lambda1_true_values_state1[4]; Lambda1_true_state1[8,2]=lambda1_true_values_state1[5]
Lambda1_true_state2=torch.zeros(O,L1_state2,device=device)
Lambda1_true_state2[0,0]=1.0; Lambda1_true_state2[1:9,0]=lambda1_true_values_state2

# ----------------------------------------------------------------
# Part 1: 共通のKalman Filter関数
# ----------------------------------------------------------------
print("\n--- 1. Defining Common Kalman Filter Functions ---")

def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape 
    L1 = B1.shape[0]
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
        except torch.linalg.LinAlgError: return torch.tensor(float('nan'), device=device)
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))
        eta_prev, P_prev = eta_updated, P_updated
    return total_log_likelihood

def get_per_time_point_log_likelihood(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape 
    L1 = B1.shape[0]
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
        except torch.linalg.LinAlgError: log_likelihoods_over_time[:, t] = -torch.inf
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_mat = torch.eye(L1, device=device).expand(N, -1, -1)
        I_minus_KL = I_mat - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_minus_KL, P_pred), I_minus_KL.transpose(1, 2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(1, 2))
        eta_prev, P_prev = eta_updated, P_updated
    return log_likelihoods_over_time

# DGPパラメータを辞書にまとめる
dgp_params = {
    'L1_state1': L1_state1, 'L1_state2': L1_state2,
    'B1_true_state1': B1_true_state1, 'B1_true_state2': B1_true_state2,
    'lambda1_true_values_state1': lambda1_true_values_state1,
    'lambda1_true_values_state2': lambda1_true_values_state2,
    'Q1_VAR': Q1_VAR, 'Q2_VAR': Q2_VAR, 'R_VAR': R_VAR
}

# ----------------------------------------------------------------
# Part 6: 全データでのモデルフィッティング (BPSモデル用)
# ----------------------------------------------------------------
print("\n\n--- 6. Fitting Models on Full Dataset (for BPS analysis) ---")

def fit_model_full_data(model_type, Y_data, dgp_params, cache_file):
    if os.path.exists(cache_file):
        print(f"Full data model for {model_type} already exists. Skipping. ('{cache_file}')")
        return torch.load(cache_file, map_location=device, weights_only=False)
    print(f"Fitting {model_type} model on the entire dataset...")
    N, Nt, O = Y_data.shape
    if model_type == '3-Factor':
        L1 = dgp_params['L1_state1']
        b0 = torch.zeros(L1, 1, device=device, requires_grad=True)
        B1 = (dgp_params['B1_true_state1'].clone() + torch.randn_like(dgp_params['B1_true_state1']) * 0.1).requires_grad_(True)
        lambda_free = (dgp_params['lambda1_true_values_state1'].clone() + torch.randn_like(dgp_params['lambda1_true_values_state1']) * 0.1).requires_grad_(True)
        log_q_diag = torch.log(torch.ones(L1, device=device) * dgp_params['Q1_VAR']).requires_grad_(True)
        log_r_diag = torch.log(torch.ones(O, device=device) * dgp_params['R_VAR']).requires_grad_(True)
        params_to_learn = [b0, B1, lambda_free, log_q_diag, log_r_diag]
    else: # 1-Factor
        L1 = dgp_params['L1_state2']
        b0 = torch.zeros(L1, 1, device=device, requires_grad=True)
        B1 = (dgp_params['B1_true_state2'].clone() + torch.randn_like(dgp_params['B1_true_state2']) * 0.1).requires_grad_(True)
        lambda_free = (dgp_params['lambda1_true_values_state2'].clone() + torch.randn_like(dgp_params['lambda1_true_values_state2']) * 0.1).requires_grad_(True)
        log_q_diag = torch.log(torch.ones(L1, device=device) * dgp_params['Q2_VAR']).requires_grad_(True)
        log_r_diag = torch.log(torch.ones(O, device=device) * dgp_params['R_VAR']).requires_grad_(True)
        params_to_learn = [b0, B1, lambda_free, log_q_diag, log_r_diag]
    optimizer = torch.optim.Adam(params_to_learn, lr=0.005)
    best_loss = float('inf')
    patience_counter = 0
    pbar = trange(10000, desc=f"Fitting Full ({model_type})")
    for epoch in pbar:
        optimizer.zero_grad()
        Q_est = torch.diag(torch.exp(log_q_diag)); R_est = torch.diag(torch.exp(log_r_diag))
        if model_type == '3-Factor':
            Lambda1_est = torch.zeros(O, L1, device=device)
            Lambda1_est[0,0]=1; Lambda1_est[1,0]=lambda_free[0]; Lambda1_est[2,0]=lambda_free[1]
            Lambda1_est[3,1]=1; Lambda1_est[4,1]=lambda_free[2]; Lambda1_est[5,1]=lambda_free[3]
            Lambda1_est[6,2]=1; Lambda1_est[7,2]=lambda_free[4]; Lambda1_est[8,2]=lambda_free[5]
        else:
            Lambda1_est = torch.zeros(O, L1, device=device)
            Lambda1_est[0,0] = 1.0; Lambda1_est[1:9,0] = lambda_free[0:8]
        logL = kalman_filter_torch_loss(Y_data, b0, B1, Lambda1_est, Q_est, R_est)
        loss = -logL
        if torch.isnan(loss): break
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {'b0': b0.detach(), 'B1': B1.detach(), 'Lambda1': Lambda1_est.detach(), 'Q': Q_est.detach(), 'R': R_est.detach()}
            patience_counter = 0
        else: patience_counter += 1
        if patience_counter >= 100: print("Early stopping triggered."); break
    results = {'params': best_params, 'loss': -best_loss}
    torch.save(results, cache_file)
    print(f"Full data model for {model_type} saved to '{cache_file}'.")
    return results

results_3fac_full = fit_model_full_data('3-Factor', Y_generated, dgp_params, MODEL_3FAC_FILE)
results_1fac_full = fit_model_full_data('1-Factor', Y_generated, dgp_params, MODEL_1FAC_FILE)
print("\nFull data model fitting complete.")

# ----------------------------------------------------------------
# Part 7: 全データで学習したモデルの評価
# ----------------------------------------------------------------
print("\n\n--- 7. Results from Models Fitted on Full Dataset ---")

# --- 7a. Displaying Aggregated Entire Set Log-Likelihoods ---
print("\n--- 7a. Displaying Aggregated Entire Set Log-Likelihoods ---")
with torch.no_grad():
    log_lik_3fac_full_data = get_per_time_point_log_likelihood(Y_generated, **results_3fac_full['params'])
    log_lik_1fac_full_data = get_per_time_point_log_likelihood(Y_generated, **results_1fac_full['params'])
agg_log_lik_3fac_full_t = log_lik_3fac_full_data.sum(dim=0)
agg_log_lik_1fac_full_t = log_lik_1fac_full_data.sum(dim=0)
print(f"\n📈 Aggregated Full-Data Log-Likelihoods for 3-Factor Model:")
for t in range(Nt): print(f"  Time {t+1:02d}: {agg_log_lik_3fac_full_t[t]:>8.2f}")
print(f"\n📈 Aggregated Full-Data Log-Likelihoods for 1-Factor Model:")
for t in range(Nt): print(f"  Time {t+1:02d}: {agg_log_lik_1fac_full_t[t]:>8.2f}")

# --- 7b. Display Estimated Parameters from Full Dataset ---
print("\n\n--- 7b. Display Estimated Parameters from Full Dataset ---")
p3_full = results_3fac_full['params']
p1_full = results_1fac_full['params']
print("\n--- 3-Factor Model Estimated vs. True Parameters ---")
print(f"  b0 (Intercept): Estimated = {np.round(p3_full['b0'].squeeze().cpu().numpy(), 3)}, True = [0. 0. 0.]")
print("\n-- Estimated B1 (State Transition) --"); [print(f"  [{row[0]:>6.3f} {row[1]:>6.3f} {row[2]:>6.3f}]") for row in p3_full['B1']]
print("-- True B1 --"); [print(f"  [{row[0]:>6.3f} {row[1]:>6.3f} {row[2]:>6.3f}]") for row in B1_true_state1]
print(f"\n-- Estimated Q diag (System Noise): --\n  {np.round(torch.diag(p3_full['Q']).cpu().numpy(), 3)}")
print(f"-- True Q diag --\n  {[Q1_VAR]*L1_state1}")
print(f"\n-- Estimated R diag (Measurement Noise): --\n  {np.round(torch.diag(p3_full['R']).cpu().numpy(), 3)}")
print(f"-- True R diag --\n  {[R_VAR]*O}")
print("\n\n--- 1-Factor Model Estimated vs. True Parameters ---")
print(f"  b0 (Intercept): Estimated = {np.round(p1_full['b0'].squeeze().cpu().numpy(), 3)}, True = [0.]")
print(f"\n-- Estimated B1 (State Transition): --\n  {np.round(p1_full['B1'].cpu().numpy(), 3)}")
print(f"-- True B1 --\n  {np.round(B1_true_state2.cpu().numpy(), 3)}")
print(f"\n-- Estimated Q diag (System Noise): --\n  {np.round(torch.diag(p1_full['Q']).cpu().numpy(), 3)}")
print(f"-- True Q diag --\n  {[Q2_VAR]*L1_state2}")
print(f"\n-- Estimated R diag (Measurement Noise): --\n  {np.round(torch.diag(p1_full['R']).cpu().numpy(), 3)}")
print(f"-- True R diag --\n  {[R_VAR]*O}")

# --- 7c. Final Model Comparison (Full Dataset) ---
# ★★★ 修正点: パラメータ数の計算をここに移動 ★★★
k_3fac = p3_full['b0'].numel() + p3_full['B1'].numel() + 6 + 3 + O
k_1fac = p1_full['b0'].numel() + p1_full['B1'].numel() + 8 + 1 + O

logL_full_3fac = results_3fac_full['loss']
logL_full_1fac = results_1fac_full['loss']
n_obs_full = N * Nt
aic_full_3fac = 2 * k_3fac - 2 * logL_full_3fac
bic_full_3fac = k_3fac * np.log(n_obs_full) - 2 * logL_full_3fac
aic_full_1fac = 2 * k_1fac - 2 * logL_full_1fac
bic_full_1fac = k_1fac * np.log(n_obs_full) - 2 * logL_full_1fac

print("\n\n--- 7c. Final Model Comparison (based on Full Dataset) ---")
table_width = 80; print("=" * table_width)
print(f"{'Model':<20} | {'Log-Likelihood':<20} | {'AIC':<15} | {'BIC':<15}")
print("-" * table_width)
print(f"{'3-Factor Model':<20} | {logL_full_3fac:<20.2f} | {aic_full_3fac:<15.2f} | {bic_full_3fac:<15.2f}")
print(f"{'(k = ' + str(k_3fac) + ')':<20} | {'':<20} | {'':<15} | {'':<15}")
print(f"{'1-Factor Model':<20} | {logL_full_1fac:<20.2f} | {aic_full_1fac:<15.2f} | {bic_full_1fac:<15.2f}")
print(f"{'(k = ' + str(k_1fac) + ')':<20} | {'':<20} | {'':<15} | {'':<15}")
print("=" * table_width)