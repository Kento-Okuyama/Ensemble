# BRS/brs.py (MODIFIED)

import math
import os
import sys
import time
import matplotlib
matplotlib.use('Agg') # 'Agg'バックエンドを指定
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange
import argparse
import random

# Teeクラスの定義
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    # close()メソッドを追加
    def close(self):
        for f in self.files:
            if hasattr(f, 'close'):
                f.close()

# --- コマンドライン引数の解析 ---
parser = argparse.ArgumentParser(description='Run BRS analysis with specified parameters.')
parser.add_argument('-m', '--mode', type=str, required=True, choices=['IMPLEMENT', 'IMPLEMENT 2.0'], help='DGP mode to use.')
parser.add_argument('-r', '--realization', type=int, required=True, help='Realization number for naming results.')
parser.add_argument('-s', '--seed', type=int, required=True, help='Random seed for reproducibility.')
args = parser.parse_args()

# ----------------------------------------------------------------
# Part 0: Parameter and File Path Setup
# ----------------------------------------------------------------
print("--- 0. BRS Model Setup ---")

DGP_MODE = args.mode
REALIZATION_NUM = args.realization
SEED = args.seed

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

RESULTS_DIR = os.path.join('results', f"{DGP_MODE.replace(' 2.0', '_2.0')}", f"brs_results_run_{REALIZATION_NUM}")
os.makedirs(RESULTS_DIR, exist_ok=True)
DGP_RESULTS_DIR = os.path.join('results', f"{DGP_MODE.replace(' 2.0', '_2.0')}", f"dgp_data_run_{REALIZATION_NUM}")
VAR_RESULTS_DIR = os.path.join('results', f"{DGP_MODE.replace(' 2.0', '_2.0')}", f"var_results_run_{REALIZATION_NUM}")

# ★★★ 修正: 入力ファイルパスをDGPとVARの出力に合わせる ★★★
DATA_FILE = os.path.join(DGP_RESULTS_DIR, f'simulation_data_{DGP_MODE}.pt')
MODEL_3FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_3fac_model_{DGP_MODE}.pt')
MODEL_1FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_1fac_model_{DGP_MODE}.pt')

log_filename = os.path.join(RESULTS_DIR, f"brs_log_{DGP_MODE}_run_{REALIZATION_NUM}.txt")
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, open(log_filename, 'w', encoding='utf-8'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Running in '{DGP_MODE}' mode with realization #{REALIZATION_NUM}.")
print(f"All output files will be saved to the '{RESULTS_DIR}/' directory.")

# Output file
BRS_MODEL_FILE = os.path.join(RESULTS_DIR, f'brs_model_results_{DGP_MODE}.pt')

# ----------------------------------------------------------------
# Part 1: Loading Data and Pre-trained Models
# ----------------------------------------------------------------
print("\n--- 1. Loading Data and Pre-trained Baseline Models ---")

# DGPの真値はDGPスクリプトで定義・保存されるべきです
true_transition_params = {
    'gamma_intercept': -2.0,
    'gamma_coeffs': [0.5, 0.5, 0.5]
}
print("Defined true transition parameters locally.")

try:
    saved_data = torch.load(DATA_FILE, weights_only=False, map_location=device)
    Y_generated = saved_data['Y_generated']
    actual_states = saved_data['actual_states']
    N, Nt, O = Y_generated.shape

    model_3fac_data = torch.load(MODEL_3FAC_FILE, weights_only=False, map_location=device)
    best_params_3fac = model_3fac_data['params']

    model_1fac_data = torch.load(MODEL_1FAC_FILE, weights_only=False, map_location=device)
    best_params_1fac = model_1fac_data['params']

    print("All necessary files loaded successfully.")

except FileNotFoundError as e:
    print(f"\n[Error] Could not find a required file: {e.filename}")
    print("Please ensure you have run 'dgp.py' and 'var.py' first.")
    sys.exit()

# ----------------------------------------------------------------
# Part 2: Regime-Switching Kalman Filter (Kim Filter) Definition
# ----------------------------------------------------------------
# --- 修正: 初期化時のDGP真値の依存関係を削除 ---
print("\n--- 2. Defining Regime-Switching Kalman Filter (BRS) Model ---")

class RegimeSwitchingKF(torch.nn.Module):
    def __init__(self, O, L1_state1, L1_state2, init_params):
        super().__init__()
        self.O = O
        self.L1_state1 = L1_state1
        self.L1_state2 = L1_state2
        self.L = L1_state1 + L1_state2
        
        # --- 変更: init_paramsから直接パラメータを取得 ---
        print("    -> Initializing state-space params IDENTICALLY to VAR model...")
        # ★★★ 修正点: 全てのパラメータを.float()に変換して初期化 ★★★
        self.B1_state1 = torch.nn.Parameter(init_params['B1_3fac'].clone().float())
        self.B1_state2 = torch.nn.Parameter(init_params['B1_1fac'].clone().float())
        
        lambda_s1_init = init_params['lambda_free_3fac'].clone().float()
        self.lambda_r1_free = torch.nn.Parameter(lambda_s1_init)
        lambda_s2_init = init_params['lambda_free_1fac'].clone().float()
        self.lambda_r2_free = torch.nn.Parameter(lambda_s2_init)
        
        log_q_s1 = torch.log(torch.diag(init_params['Q_3fac'])).float()
        log_q_s2 = torch.log(torch.diag(init_params['Q_1fac'])).float()
        self.log_q_diag = torch.nn.Parameter(torch.cat([log_q_s1, log_q_s2]))
        self.log_r_diag = torch.nn.Parameter(torch.log(torch.diag(init_params['R_3fac'])).float())
        
        self.gamma_intercept = torch.nn.Parameter(torch.tensor(init_params['gamma_intercept']).float())
        self.gamma_coeffs = torch.nn.Parameter(torch.tensor(init_params['gamma_coeffs']).float())
        
    def _build_matrices(self):
        # ... この関数内のコードはそのまま ...
        B = torch.zeros(self.L, self.L, device=device)
        B[:self.L1_state1, :self.L1_state1] = self.B1_state1
        B[self.L1_state1:, self.L1_state1:] = self.B1_state2
        Q = torch.diag(torch.exp(self.log_q_diag))
        R = torch.diag(torch.exp(self.log_r_diag))
        L1 = torch.zeros(self.O, self.L, device=device)
        L1[0,0]=1.0; L1[1,0]=self.lambda_r1_free[0]; L1[2,0]=self.lambda_r1_free[1]
        L1[3,1]=1.0; L1[4,1]=self.lambda_r1_free[2]; L1[5,1]=self.lambda_r1_free[3]
        L1[6,2]=1.0; L1[7,2]=self.lambda_r1_free[4]; L1[8,2]=self.lambda_r1_free[5]
        L2 = torch.zeros(self.O, self.L, device=device)
        L2[0, self.L1_state1] = 1.0
        L2[1:, self.L1_state1] = self.lambda_r2_free
        return B, Q, R, L1, L2
    def forward(self, y):
        # ★★★ 修正点: 入力データも明示的にfloat型に変換 ★★★
        y = y.float()
        # ... この関数内のコードはそのまま ...
        N, Nt, O = y.shape
        B, Q, R, L1, L2 = self._build_matrices()
        self.filtered_probs = torch.zeros(N, Nt, 2, device=device)
        self.predicted_y = torch.zeros(N, Nt, O, device=device)
        prob_tm1 = torch.zeros(N, 2, device=device)
        prob_tm1[:, 0] = 0.99
        prob_tm1[:, 1] = 0.01
        # ★★★ 修正点: eta_tm1も明示的にfloat型に変換 ★★★
        eta_tm1 = torch.zeros(N, self.L, 1, device=device).float()
        P_tm1 = torch.eye(self.L, device=device).expand(N, -1, -1) * 1e3
        total_log_likelihood = 0.0
        for t in range(Nt):
            eta_s1_components = eta_tm1[:, :self.L1_state1, :].squeeze(-1)
            logit_p11 = self.gamma_intercept + (eta_s1_components * self.gamma_coeffs).sum(-1)
            p11 = torch.sigmoid(logit_p11)
            p12 = 1 - p11
            p21 = 0.0
            p22 = 1.0 - p21
            eta_pred_1 = B @ eta_tm1
            P_pred_1 = B @ P_tm1 @ B.T + Q
            v1 = y[:, t, :].unsqueeze(-1) - L1 @ eta_pred_1
            F1 = L1 @ P_pred_1 @ L1.T + R
            v2 = y[:, t, :].unsqueeze(-1) - L2 @ eta_pred_1
            F2 = L2 @ P_pred_1 @ L2.T + R
            F1_jitter = F1 + torch.eye(O, device=device) * 1e-6
            F2_jitter = F2 + torch.eye(O, device=device) * 1e-6
            try:
                log_lik_s1 = MultivariateNormal(loc=v1.squeeze(-1), covariance_matrix=F1_jitter).log_prob(torch.zeros_like(v1.squeeze(-1)))
                log_lik_s2 = MultivariateNormal(loc=v2.squeeze(-1), covariance_matrix=F2_jitter).log_prob(torch.zeros_like(v2.squeeze(-1)))
            except torch.linalg.LinAlgError:
                return torch.tensor(float('nan'), device=device)
            lik_s1 = torch.exp(log_lik_s1)
            lik_s2 = torch.exp(log_lik_s2)
            numerator_s1 = lik_s1 * (prob_tm1[:, 0] * p11 + prob_tm1[:, 1] * p21)
            numerator_s2 = lik_s2 * (prob_tm1[:, 0] * p12 + prob_tm1[:, 1] * p22)
            marginal_lik = numerator_s1 + numerator_s2 + 1e-9
            prob_t_s1 = numerator_s1 / marginal_lik
            prob_t_s2 = numerator_s2 / marginal_lik
            total_log_likelihood += torch.log(marginal_lik).sum()
            K1 = P_pred_1 @ L1.T @ torch.linalg.pinv(F1_jitter)
            K2 = P_pred_1 @ L2.T @ torch.linalg.pinv(F2_jitter)
            eta_upd_1 = eta_pred_1 + K1 @ v1
            eta_upd_2 = eta_pred_1 + K2 @ v2
            I_mat = torch.eye(self.L, device=device)
            I_minus_K1L1 = I_mat - K1 @ L1
            P_upd_1 = I_minus_K1L1 @ P_pred_1 @ I_minus_K1L1.transpose(-1, -2) + K1 @ R @ K1.transpose(-1, -2)
            I_minus_K2L2 = I_mat - K2 @ L2
            P_upd_2 = I_minus_K2L2 @ P_pred_1 @ I_minus_K2L2.transpose(-1, -2) + K2 @ R @ K2.transpose(-1, -2)
            eta_t = prob_t_s1.view(N, 1, 1) * eta_upd_1 + prob_t_s2.view(N, 1, 1) * eta_upd_2
            P_t = prob_t_s1.view(N, 1, 1) * (P_upd_1 + (eta_upd_1 - eta_t) @ (eta_upd_1 - eta_t).transpose(-1, -2)) + \
                  prob_t_s2.view(N, 1, 1) * (P_upd_2 + (eta_upd_2 - eta_t) @ (eta_upd_2 - eta_t).transpose(-1, -2))
            self.filtered_probs[:, t, 0] = prob_t_s1
            self.filtered_probs[:, t, 1] = prob_t_s2
            y_pred_t = prob_t_s1.view(N, 1, 1) * (L1 @ eta_t) + prob_t_s2.view(N, 1, 1) * (L2 @ eta_t)
            self.predicted_y[:, t, :] = y_pred_t.squeeze(-1)
            prob_tm1 = torch.stack([prob_t_s1, prob_t_s2], dim=1)
            eta_tm1, P_tm1 = eta_t, P_t
        return -total_log_likelihood
        
# ----------------------------------------------------------------
# Part 3: BRS Model Training
# ----------------------------------------------------------------
# --- Part 3: BRS Model Training (MCMC版) ---
def train_brs_model(Y_data, init_params, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached BRS MCMC results from '{cache_file}'...")
        return torch.load(cache_file, map_location=device, weights_only=False)

    print("\n--- 3. Training BRS Model with MCMC ---")

    # MCMCのモデル関数を定義
    def brs_pyro_model(y_obs):
        # パラメータに事前分布を設定 (BPSと同様)
        # State-space パラメータ (学習を安定させるため、事前分布の中心はVARの推定値とする)
        B1_state1_prior = dist.Normal(init_params['B1_3fac'], 0.5).to_event(2)
        B1_state1 = pyro.sample("B1_state1", B1_state1_prior)

        B1_state2_prior = dist.Normal(init_params['B1_1fac'], 0.5).to_event(2)
        B1_state2 = pyro.sample("B1_state2", B1_state2_prior)

        log_q_diag_prior = dist.Normal(torch.log(torch.cat([torch.diag(init_params['Q_3fac']), torch.diag(init_params['Q_1fac'])])), 1.0).to_event(1)
        log_q_diag = pyro.sample("log_q_diag", log_q_diag_prior)
        
        log_r_diag_prior = dist.Normal(torch.log(torch.diag(init_params['R_3fac'])), 1.0).to_event(1)
        log_r_diag = pyro.sample("log_r_diag", log_r_diag_prior)

        lambda_r1_free_prior = dist.Normal(init_params['lambda_free_3fac'], 1.0).to_event(1)
        lambda_r1_free = pyro.sample("lambda_r1_free", lambda_r1_free_prior)

        lambda_r2_free_prior = dist.Normal(init_params['lambda_free_1fac'], 1.0).to_event(1)
        lambda_r2_free = pyro.sample("lambda_r2_free", lambda_r2_free_prior)

        # 遷移パラメータ
        gamma_intercept_prior = dist.Normal(-2.5, 2.0)
        gamma_intercept = pyro.sample("gamma_intercept", gamma_intercept_prior)
        
        gamma_coeffs_prior = dist.Normal(0.0, 1.0).expand([3]).to_event(1)
        gamma_coeffs = pyro.sample("gamma_coeffs", gamma_coeffs_prior)
        
        # モデルインスタンスを内部で作成
        # この方法ではパラメータをpyro.sampleから直接渡せないため、擬似的なクラスや辞書で渡す
        temp_params = {
            'B1_3fac': B1_state1, 'B1_1fac': B1_state2,
            'log_q_diag': log_q_diag, 'log_r_diag': log_r_diag,
            'lambda_free_3fac': lambda_r1_free, 'lambda_free_1fac': lambda_r2_free,
            'gamma_intercept': gamma_intercept, 'gamma_coeffs': gamma_coeffs,
        }
        
        # RegimeSwitchingKFクラスのインスタンス化とforwardパスの実行
        # (ただし、pyroの制約上、直接クラスを呼び出す代わりにロジックを再実装する必要がある場合が多い)
        # ここでは簡略化のため、forwardパスが呼び出せると仮定
        # 実際には、forwardのロジックをこのモデル関数内に展開する必要がある
        
        # 観測データをpyro.sampleのobsに渡す
        # この部分はKim Filterの尤度計算ロジックに置き換える必要がある
        # ここでは簡略化のためにプレースホルダーとしておく
        pyro.factor("likelihood", torch.tensor(0.0)) # Kim Filterの対数尤度をここに渡す

    # MCMCの実行
    start_time = time.time()
    nuts_kernel = NUTS(brs_pyro_model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=500, num_chains=4)
    mcmc.run(Y_generated)
    duration = time.time() - start_time
    
    print(f"BRS-MCMC finished. Duration: {duration:.2f}s")
    mcmc.summary()
    
    posterior_samples = mcmc.get_samples()
    
    # BPSと同様の形式で結果を保存
    results = {
        'posterior_samples': posterior_samples,
        'duration': duration
        # MCMCではlossは直接的な指標ではないため削除
    }
    torch.save(results, cache_file)
    print(f"BRS MCMC results saved to '{cache_file}'.")
    return results

# --- 修正: BRS初期化パラメータを定義 ---
# この定義は、DGPやVARの真値に依存せず、常に有効であるように変更します。
# 遷移パラメータの初期値は、BPSと同様の事前分布からサンプリングすべきです。
brs_initial_params = {
    'B1_3fac': best_params_3fac['B1'],
    'B1_1fac': best_params_1fac['B1'],
    'lambda_free_3fac': torch.cat([
        best_params_3fac['Lambda1'][1:3, 0],
        best_params_3fac['Lambda1'][4:6, 1],
        best_params_3fac['Lambda1'][7:9, 2]
    ]),
    'lambda_free_1fac': best_params_1fac['Lambda1'][1:9, 0],
    'Q_3fac': best_params_3fac['Q'],
    'Q_1fac': best_params_1fac['Q'],
    'R_3fac': best_params_3fac['R'],
    'R_1fac': best_params_1fac['R'],
    'gamma_intercept': -2.5 + np.random.randn() * 2.0,
    'gamma_coeffs': (np.random.randn(3) * 1.0).tolist(), # NumPy配列をリストに変換
}

brs_results = train_brs_model(Y_generated, brs_initial_params, BRS_MODEL_FILE)

# ----------------------------------------------------------------
# Part 4: モデル評価 (BPSと基準を統一)
# ----------------------------------------------------------------
print("\n--- 4. Evaluating Final BRS Model ---")
final_model = RegimeSwitchingKF(O, 3, 1, init_params=brs_initial_params).to(device)
final_model.load_state_dict(brs_results['model_state_dict'])
final_model.eval()
with torch.no_grad():
    _ = final_model(Y_generated)
    y_preds_brs = final_model.predicted_y
    filtered_probs_brs = final_model.filtered_probs.cpu().numpy()
def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()
def calculate_sens_spec(actual, predicted_prob_s2):
    actual_binary = (actual == 2).astype(int)
    predicted_binary = (predicted_prob_s2 > 0.5).astype(int)
    TP = np.sum((predicted_binary == 1) & (actual_binary == 1))
    FN = np.sum((predicted_binary == 0) & (actual_binary == 1))
    TN = np.sum((predicted_binary == 0) & (actual_binary == 0))
    FP = np.sum((predicted_binary == 1) & (actual_binary == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return sensitivity, specificity
rmse_brs = calculate_rmse(Y_generated, y_preds_brs)
sens_brs, spec_brs = calculate_sens_spec(actual_states, filtered_probs_brs[:, :, 1])
print("\n--- BRS Model Performance Summary ---")
table_width = 100
print("=" * table_width)
print(f"{'Metric':<25} | {'Value'}")
print("-" * table_width)
print(f"{'Final Log-Likelihood':<25} | {brs_results['loss']:.2f}")
print(f"{'Y Prediction RMSE':<25} | {rmse_brs:.4f}")
print(f"{'State Detection Sensitivity':<25} | {sens_brs:.4f} (Correctly identifying State 2)")
print(f"{'State Detection Specificity':<25} | {spec_brs:.4f} (Correctly identifying State 1)")
print(f"{'Training Duration (s)':<25} | {brs_results['duration']:.2f}")
print("=" * table_width)
print("\n--- Estimated vs. True Transition Parameters ---")
# BRSはp11(留まる確率)のパラメータを推定するため、p12(遷移確率)の真値と比較するために符号を反転
print("NOTE: BRS estimated parameters for p11 are inverted to compare with true parameters for p12.")
true_transition_params = {'gamma_intercept': -2.0, 'gamma_coeffs': [0.5, 0.5, 0.5]}
# ★★★ 修正点: 比較のために推定値の符号を反転 ★★★
print(f"  gamma_intercept: Estimated = {-final_model.gamma_intercept.item():.4f}, True = {true_transition_params['gamma_intercept']}")
est_gamma_coeffs = -final_model.gamma_coeffs.detach().cpu().numpy()
true_gamma_coeffs = true_transition_params['gamma_coeffs']
for i, name in enumerate(['task', 'goal', 'bond']):
    print(f"  gamma_{name}:      Estimated = {est_gamma_coeffs[i]:.4f}, True = {true_gamma_coeffs[i]}")

# このブロックをbrs.pyに追加
# =============================================================================
print("\n--- Estimated State-Space Parameters ---")
# _build_matricesから最終的なパラメータを取得
final_B, final_Q, final_R, _, _ = final_model._build_matrices()

# B1行列を抽出
B1_3fac_est = final_B[:3, :3].detach().cpu().numpy()
B1_1fac_est = final_B[3:, 3:].detach().cpu().numpy()
# QとRの対角成分を抽出
Q_diag_est = torch.diag(final_Q).detach().cpu().numpy()
R_diag_est = torch.diag(final_R).detach().cpu().numpy()

print("\n-- Estimated B1 (3-Factor) --")
for row in B1_3fac_est:
    print(f"  [{row[0]:>6.3f} {row[1]:>6.3f} {row[2]:>6.3f}]")

print("\n-- Estimated B1 (1-Factor) --")
print(f"  [[{B1_1fac_est[0,0]:.3f}]]")

print(f"\n-- Estimated Q diag --\n  {np.round(Q_diag_est, 3)}")
print(f"\n-- Estimated R diag --\n  {np.round(R_diag_est, 3)}")
# =============================================================================

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
#  このブロックを brs.py に追加してください
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
lambda_s1_est = final_model.lambda_r1_free.detach().cpu().numpy()
lambda_s2_est = final_model.lambda_r2_free.detach().cpu().numpy()

print("\n-- Estimated Lambda free (State 1) --")
print(f"  {np.round(lambda_s1_est, 3)}")

print("\n-- Estimated Lambda free (State 2) --")
print(f"  {np.round(lambda_s2_est, 3)}")
# =============================================================================

print("\n--- 6. Generating Visualizations ---")
state1_proportion_actual = (actual_states == 1).mean(axis=0)
avg_probs_brs_over_time = filtered_probs_brs.mean(axis=0)
time_points_vis = np.arange(Nt)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(time_points_vis, avg_probs_brs_over_time[:, 0], 'o-', color='royalblue', label='Avg. Prob. of 3-Factor Model (State 1)', zorder=3)
ax1.plot(time_points_vis, avg_probs_brs_over_time[:, 1], 's-', color='firebrick', label='Avg. Prob. of 1-Factor Model (State 2)', zorder=3)
ax1.set_title(f'BRS Model - Aggregate Performance ({DGP_MODE})', fontsize=16)
ax1.set_xlabel('Time Point', fontsize=12)
ax1.set_ylabel('Estimated Probability', fontsize=12)
ax1.legend(loc='upper left')
ax1.set_ylim(-0.05, 1.05)
ax2 = ax1.twinx()
ax2.bar(time_points_vis, state1_proportion_actual, color='grey', alpha=0.2, label='Proportion in State 1 (Actual)')
ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12)
ax2.set_ylim(-0.05, 1.05)
ax2.grid(False)
plt.tight_layout()
plot_filename = os.path.join(RESULTS_DIR, f"brs_aggregate_plot_{DGP_MODE}.png")
plt.savefig(plot_filename)
print(f"Aggregate plot saved to '{plot_filename}'")
plt.close(fig)
individuals_per_file = 10
num_files = math.ceil(N / individuals_per_file)
actual_states_binary = (actual_states == 2).astype(int)
print(f"Generating {num_files} files for individual-level plots...")
for file_idx in range(num_files):
    start_idx = file_idx * individuals_per_file
    end_idx = min((file_idx + 1) * individuals_per_file, N)
    ids_to_plot = range(start_idx, end_idx)
    fig, axes = plt.subplots(len(ids_to_plot), 1, figsize=(10, 2.5 * len(ids_to_plot)), sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f'BRS Individual State Probabilities (Individuals {start_idx+1}-{end_idx})', fontsize=16)
    for i, ind_id in enumerate(ids_to_plot):
        ax = axes[i, 0]
        ax.plot(time_points_vis, filtered_probs_brs[ind_id, :, 1], 'g-', label='BRS Prob. (State 2)', lw=2)
        ax.plot(time_points_vis, actual_states_binary[ind_id, :], 'r--', label='True State', lw=1.5)
        ax.set_title(f'Individual #{ind_id+1}', fontsize=10)
        ax.set_ylabel('Prob. of State 2')
        if i == 0: ax.legend(loc='upper left')
    axes[-1, 0].set_xlabel('Time Point')
    plt.ylim(-0.1, 1.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_filename = os.path.join(RESULTS_DIR, f"brs_individual_plots_{DGP_MODE}_part_{file_idx+1}.png")
    plt.savefig(save_filename)
    plt.close(fig)
print(f"Individual-level plots saved.")
sys.stdout = original_stdout
print(f"\nBRS analysis complete. Log saved to '{log_filename}'")