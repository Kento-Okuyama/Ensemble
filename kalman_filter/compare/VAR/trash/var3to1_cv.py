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

# ★★★ 修正箇所 ★★★
def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape 
    L1 = B1.shape[0]
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    total_log_likelihood = 0.0
    for t in range(Nt):
        y_t = Y[:, t, :].unsqueeze(-1)
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        # B1.T と Lambda1.T を使って正しく転置
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        try:
            F_t_jitter = F_t + torch.eye(O1, device=device) * 1e-6
            dist_t = MultivariateNormal(loc=v_t.squeeze(-1), covariance_matrix=F_t_jitter)
            total_log_likelihood += dist_t.log_prob(torch.zeros_like(v_t.squeeze(-1))).sum()
        except torch.linalg.LinAlgError:
            # print(f"Warning: LinAlgError at t={t}. Returning NaN.")
            return torch.tensor(float('nan'), device=device)
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        P_updated = P_pred - torch.bmm(torch.bmm(K_t, F_t_jitter), K_t.transpose(1, 2))
        eta_prev, P_prev = eta_updated, P_updated
    return total_log_likelihood

# ★★★ 修正箇所 ★★★
def get_kalman_predictive_distribution(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape
    L1 = B1.shape[0]
    y_pred_mean_series = torch.zeros(N, Nt, O1, device=device)
    y_pred_cov_series = torch.zeros(N, Nt, O1, O1, device=device)
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3 

    for t in range(Nt):
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        # B1.T と Lambda1.T を使って正しく転置
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.T.expand(N, -1, -1)) + Q
        y_pred_mean = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        y_pred_cov = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.T.expand(N, -1, -1)) + R
        y_pred_mean_series[:, t, :] = y_pred_mean.squeeze(-1)
        y_pred_cov_series[:, t, :, :] = y_pred_cov
        
        y_t = Y[:, t, :].unsqueeze(-1)
        v_t = y_t - y_pred_mean
        F_t_jitter = y_pred_cov + torch.eye(O1, device=device) * 1e-6
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.T.expand(N, -1, -1)), torch.linalg.pinv(F_t_jitter))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        P_updated = P_pred - torch.bmm(torch.bmm(K_t, F_t_jitter), K_t.transpose(1, 2))
        eta_prev, P_prev = eta_updated, P_updated
        
    return y_pred_mean_series, y_pred_cov_series

# ----------------------------------------------------------------
# Part 2: LOOCVによるモデルフィッティングと評価
# ----------------------------------------------------------------

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
        P_updated = P_pred - torch.bmm(torch.bmm(K_t, F_t_jitter), K_t.transpose(1, 2))
        eta_prev, P_prev = eta_updated, P_updated
        
    return log_likelihoods_over_time

def run_loocv_for_model(model_type, Y_data, dgp_params, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached LOOCV results for {model_type} model from '{cache_file}'...")
        return torch.load(cache_file, map_location=device)

    print(f"\n--- Starting LOOCV for {model_type} Model ---")
    print("!!! WARNING: This is computationally expensive and will take a long time. !!!")
    
    N, Nt, O = Y_data.shape
    device = Y_data.device
    all_test_log_liks_full = torch.zeros(N, Nt, device=device)
    
    # Store parameters from each fold to average later
    param_sums = {}

    for i in trange(N, desc=f"LOOCV Folds ({model_type})"):
        Y_train = torch.cat([Y_data[:i], Y_data[i+1:]], dim=0)
        Y_test = Y_data[i:i+1]

        # Initialize parameters
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
        
        optimizer = torch.optim.Adam(params_to_learn, lr=0.01)
        best_loss = float('inf')
        patience_counter = 0

        # Training loop for one fold
        for epoch in range(5000):
            optimizer.zero_grad()
            Q_est = torch.diag(torch.exp(log_q_diag))
            R_est = torch.diag(torch.exp(log_r_diag))
            
            if model_type == '3-Factor':
                Lambda1_est = torch.zeros(O, L1, device=device)
                Lambda1_est[0,0]=1; Lambda1_est[1,0]=lambda_free[0]; Lambda1_est[2,0]=lambda_free[1]
                Lambda1_est[3,1]=1; Lambda1_est[4,1]=lambda_free[2]; Lambda1_est[5,1]=lambda_free[3]
                Lambda1_est[6,2]=1; Lambda1_est[7,2]=lambda_free[4]; Lambda1_est[8,2]=lambda_free[5]
            else:
                Lambda1_est = torch.zeros(O, L1, device=device)
                Lambda1_est[0,0] = 1.0; Lambda1_est[1:9,0] = lambda_free[0:8]
            
            logL = kalman_filter_torch_loss(Y_train, b0, B1, Lambda1_est, Q_est, R_est)
            loss = -logL
            if torch.isnan(loss): break
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params_fold = {'b0': b0.detach(), 'B1': B1.detach(), 'Lambda1': Lambda1_est.detach(), 'Q': Q_est.detach(), 'R': R_est.detach()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 50:
                break
        
        # Test on the held-out individual
        with torch.no_grad():
            test_logL_t = get_per_time_point_log_likelihood(Y_test, **best_params_fold)
            all_test_log_liks_full[i, :] = test_logL_t.squeeze(0)

        # Accumulate parameters for averaging
        for key in best_params_fold:
            if i == 0:
                param_sums[key] = best_params_fold[key]
            else:
                param_sums[key] += best_params_fold[key]

    # Average the parameters across all folds
    avg_params = {key: param_sums[key] / N for key in param_sums}
    
    # Calculate final total log-likelihood
    total_logL = all_test_log_liks_full.sum().item()
    
    results = {
        "log_lik_full": all_test_log_liks_full,
        "total_logL": total_logL,
        "avg_params": avg_params
    }
    torch.save(results, cache_file)
    print(f"LOOCV for {model_type} finished. Results saved to '{cache_file}'.")
    return results

# DGPパラメータを辞書にまとめる
dgp_params = {
    'L1_state1': L1_state1, 'L1_state2': L1_state2,
    'B1_true_state1': B1_true_state1, 'B1_true_state2': B1_true_state2,
    'lambda1_true_values_state1': lambda1_true_values_state1,
    'lambda1_true_values_state2': lambda1_true_values_state2,
    'Q1_VAR': Q1_VAR, 'Q2_VAR': Q2_VAR, 'R_VAR': R_VAR
}

# LOOCVを実行
loocv_results_3fac = run_loocv_for_model('3-Factor', Y_generated, dgp_params, f'loocv_3fac_{DGP_MODE}.pt')
loocv_results_1fac = run_loocv_for_model('1-Factor', Y_generated, dgp_params, f'loocv_1fac_{DGP_MODE}.pt')

# 結果を変数に展開
log_lik_3fac_full = loocv_results_3fac['log_lik_full']
logL_3fac = loocv_results_3fac['total_logL']
avg_params_3fac = loocv_results_3fac['avg_params']

log_lik_1fac_full = loocv_results_1fac['log_lik_full']
logL_1fac = loocv_results_1fac['total_logL']
avg_params_1fac = loocv_results_1fac['avg_params']

# ----------------------------------------------------------------
# Part 3: 予測値と尤度の表示（LOOCVベース）
# ----------------------------------------------------------------
print("\n--- 3. Displaying Aggregated LOOCV-based Log-Likelihoods ---")

agg_log_lik_3fac_t = log_lik_3fac_full.sum(dim=0)
agg_log_lik_1fac_t = log_lik_1fac_full.sum(dim=0)

print(f"\n📈 Aggregated Test Log-Likelihoods for 3-Factor Model:")
for t in range(Nt):
    print(f"  Time {t+1:02d}: {agg_log_lik_3fac_t[t]:>8.2f}")

print(f"\n📈 Aggregated Test Log-Likelihoods for 1-Factor Model:")
for t in range(Nt):
    print(f"  Time {t+1:02d}: {agg_log_lik_1fac_t[t]:>8.2f}")


# ----------------------------------------------------------------
# Part 4: 推定されたパラメータの表示（LOOCV平均値）
# ----------------------------------------------------------------
print("\n--- 4. Display Averaged Estimated Parameters from LOOCV ---")

# --- 3-Factor Model ---
print("\n--- 3-Factor Model Averaged Estimated vs. True Parameters ---")
p3 = avg_params_3fac
# (Parameter printing logic from your previous script)
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
print("\n\n--- 1-Factor Model Averaged Estimated vs. True Parameters ---")
p1 = avg_params_1fac
# (Parameter printing logic from your previous script)
print(f"  b0 (Intercept): Estimated = {np.round(p1['b0'].squeeze().cpu().numpy(), 3)}, True = [0.]")
print(f"\n-- Estimated B1 (State Transition): --\n  {np.round(p1['B1'].cpu().numpy(), 3)}")
print(f"-- True B1 --\n  {np.round(B1_true_state2.cpu().numpy(), 3)}")
print(f"\n-- Estimated Q diag (System Noise): --\n  {np.round(torch.diag(p1['Q']).cpu().numpy(), 3)}")
print(f"-- True Q diag --\n  {[Q2_VAR]*L1_state2}")
print(f"\n-- Estimated R diag (Measurement Noise): --\n  {np.round(torch.diag(p1['R']).cpu().numpy(), 3)}")
print(f"-- True R diag --\n  {[R_VAR]*O}")


# ----------------------------------------------------------------
# Part 5: 最終的なモデル比較 (LOOICベース)
# ----------------------------------------------------------------
k_3fac = p3['b0'].numel() + p3['B1'].numel() + 6 + 3 + O
k_1fac = p1['b0'].numel() + p1['B1'].numel() + 8 + 1 + O
looic_3fac = -2 * logL_3fac
looic_1fac = -2 * logL_1fac

print("\n\n--- 5. Final Model Comparison (based on LOOCV) ---")
table_width = 80
print("=" * table_width)
print(f"{'Model':<20} | {'Out-of-Sample LogL':<22} | {'LOOIC':<15}")
print("-" * table_width)
print(f"{'3-Factor Model':<20} | {logL_3fac:<22.2f} | {looic_3fac:<15.2f}")
print(f"{'(k = ' + str(k_3fac) + ')':<20} | {'':<22} | {'':<15}")
print(f"{'1-Factor Model':<20} | {logL_1fac:<22.2f} | {looic_1fac:<15.2f}")
print(f"{'(k = ' + str(k_1fac) + ')':<20} | {'':<22} | {'':<15}")
print("=" * table_width)
print("* LogL and LOOIC are based on out-of-sample predictions from LOOCV.")
print("* Lower LOOIC indicates better expected predictive performance on new data.")


# ----------------------------------------------------------------
# Part 6: 時点ごとのアンサンブル重みの計算と可視化 (LOOCVベース)
# ----------------------------------------------------------------
print("\n--- 6. Calculating and Visualizing Time-Varying Ensemble Weights (LOOCV-based) ---")

avg_lik_3fac = torch.exp(log_lik_3fac_full).mean(dim=0).cpu().numpy()
avg_lik_1fac = torch.exp(log_lik_1fac_full).mean(dim=0).cpu().numpy()

weights = np.zeros((Nt, 2))
weights[0, :] = [0.5, 0.5]

for t in range(1, Nt):
    unnormalized_w0 = weights[t-1, 0] * avg_lik_3fac[t]
    unnormalized_w1 = weights[t-1, 1] * avg_lik_1fac[t]
    total_w = unnormalized_w0 + unnormalized_w1
    weights[t, 0] = unnormalized_w0 / total_w
    weights[t, 1] = unnormalized_w1 / total_w

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))
sessions = np.arange(1, Nt + 1)
ax.stackplot(sessions, weights[:, 0], weights[:, 1], 
             labels=['Weight: 3-Factor Model', 'Weight: 1-Factor Model'],
             colors=['#3498db', '#e74c3c'], alpha=0.7)
ax.set_title(f"Time-Varying Ensemble Weights based on LOOCV (Mode: {DGP_MODE})", fontsize=16)
ax.set_xlabel("Sessions", fontsize=12)
ax.set_ylabel("Ensemble Weight", fontsize=12)
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'ensemble_weights_loocv_{DGP_MODE}.png')
plt.show()

print(f"\n✅ LOOCV-based ensemble weights plot saved as 'ensemble_weights_loocv_{DGP_MODE}.png'")