# FRS/frs.py (MODIFIED)

import math
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange

# --- Common Utilities ---
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

# ----------------------------------------------------------------
# Part 0: Parameter and File Path Setup
# ----------------------------------------------------------------
print("--- 0. FRS Model Setup ---")
DGP_MODE = 'IMPLEMENT 2.0'

# ★★★ CHANGE: Create results directory and set up logging ★★★
RESULTS_DIR = f"results_{DGP_MODE.replace(' 2.0', '_2.0')}"
os.makedirs(RESULTS_DIR, exist_ok=True)

log_filename = os.path.join(RESULTS_DIR, f"frs_log_{DGP_MODE}.txt")
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, open(log_filename, 'w', encoding='utf-8'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Running in '{DGP_MODE}' mode.")
print(f"All output files will be saved to the '{RESULTS_DIR}/' directory.")

# ★★★ CHANGE: Update file paths to use results subdirectories ★★★
# Input files
DGP_RESULTS_DIR = os.path.join('..', 'DGP', RESULTS_DIR)
VAR_RESULTS_DIR = os.path.join('..', 'VAR', RESULTS_DIR)
DATA_FILE = os.path.join(DGP_RESULTS_DIR, f'simulation_data_{DGP_MODE}.pt')
MODEL_3FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_3fac_model_{DGP_MODE}.pt')
MODEL_1FAC_FILE = os.path.join(VAR_RESULTS_DIR, f'fitted_1fac_model_{DGP_MODE}.pt')

# Output file
FRS_MODEL_FILE = os.path.join(RESULTS_DIR, f'frs_model_results_{DGP_MODE}.pt')

# ----------------------------------------------------------------
# Part 1: Loading Data and Pre-trained Models
# ----------------------------------------------------------------
print("\n--- 1. Loading Data and Pre-trained Baseline Models ---")

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
    print("Please ensure you have run 'dgp.py' and 'var/var_split.py' first.")
    sys.exit()

# ★★★ 追加: VARモデルの初期値と完全に一致させるため、DGPの真のパラメータを再定義 ★★★
if DGP_MODE == 'IMPLEMENT':
    B1_true_state1 = torch.tensor([[0.04, 0.01, -0.11], [-0.01, 0.07, 0.13], [0.02, 0.11, 0.16]], device=device)
    B1_true_state2 = torch.tensor([[0.50]], device=device)
elif DGP_MODE == 'IMPLEMENT 2.0':
    B1_true_state1 = torch.tensor([[0.17, -0.06, 0.00], [0.14, 0.21, -0.10], [-0.29, -0.22, 0.11]], device=device)
    B1_true_state2 = torch.tensor([[0.24]], device=device)

Q1_VAR, Q2_VAR, R_VAR = 0.5, 0.1, 0.5
lambda1_true_values_state1 = torch.tensor([1.2,0.8,1.1,0.9,1.3,0.7], device=device)
lambda1_true_values_state2 = torch.tensor([1.2,0.8,1.1,0.9,1.3,0.7,0.6,1.0], device=device)

# ----------------------------------------------------------------
# Part 2: Regime-Switching Kalman Filter (Kim Filter) Definition
# ----------------------------------------------------------------
print("\n--- 2. Defining Regime-Switching Kalman Filter (FRS) Model ---")

class RegimeSwitchingKF(torch.nn.Module):
    def __init__(self, O, L1_state1, L1_state2, init_params):
        super().__init__()
        self.O = O
        self.L1_state1 = L1_state1
        self.L1_state2 = L1_state2
        self.L = L1_state1 + L1_state2

        print("    -> Initializing state-space params IDENTICALLY to VAR model (True value + noise)...")
        
        # ★★★ 修正点: 全ての初期値をVARモデルと完全に同じ方法で設定 ★★★
        noise_level = 0.1
        
        # B Matrix Initialization (True value + noise)
        self.B1_state1 = torch.nn.Parameter(B1_true_state1.clone() + torch.randn_like(B1_true_state1) * noise_level)
        self.B1_state2 = torch.nn.Parameter(B1_true_state2.clone() + torch.randn_like(B1_true_state2) * noise_level)
        
        # Lambda Initialization (True value + noise)
        lambda_s1_init = lambda1_true_values_state1.clone() + torch.randn_like(lambda1_true_values_state1) * noise_level
        self.lambda_r1_free = torch.nn.Parameter(lambda_s1_init)
        
        lambda_s2_init = lambda1_true_values_state2.clone() + torch.randn_like(lambda1_true_values_state2) * noise_level
        self.lambda_r2_free = torch.nn.Parameter(lambda_s2_init)

        # Q and R Initialization (Log of true variance)
        q_s1 = torch.ones(self.L1_state1, device=device) * Q1_VAR
        q_s2 = torch.ones(self.L1_state2, device=device) * Q2_VAR
        self.log_q_diag = torch.nn.Parameter(torch.log(torch.cat([q_s1, q_s2])))
        
        self.log_r_diag = torch.nn.Parameter(torch.log(torch.ones(self.O, device=device) * R_VAR))

        # Gamma Initialization (remains the same: sampling from BPS priors)
        print("    -> Aligning transition param initial values by SAMPLING from BPS priors.")
        init_intercept = torch.tensor(2.5) + torch.randn(()) * 2.0
        self.gamma_intercept = torch.nn.Parameter(init_intercept)
        init_coeffs = torch.zeros(self.L1_state1) + torch.randn(self.L1_state1) * 1.0
        self.gamma_coeffs = torch.nn.Parameter(init_coeffs)

    def _build_matrices(self):
        # ★★★ FIX 2: Construct B as a block-diagonal matrix ★★★
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
        N, Nt, O = y.shape
        B, Q, R, L1, L2 = self._build_matrices()
        self.filtered_probs = torch.zeros(N, Nt, 2, device=device)
        self.predicted_y = torch.zeros(N, Nt, O, device=device)
        prob_tm1 = torch.zeros(N, 2, device=device)
        prob_tm1[:, 0] = 0.99
        prob_tm1[:, 1] = 0.01
        eta_tm1 = torch.zeros(N, self.L, 1, device=device)
        P_tm1 = torch.eye(self.L, device=device).expand(N, -1, -1) * 1e3
        total_log_likelihood = 0.0

        for t in range(Nt):
            # 1. Calculate transition probabilities (dependent on eta_{t-1})
            eta_s1_components = eta_tm1[:, :self.L1_state1, :].squeeze(-1)
            # ★★★ 変更点: 'gamma1', 'gamma2' を 'gamma_intercept', 'gamma_coeffs' に変更 ★★★
            logit_p11 = self.gamma_intercept + (eta_s1_components * self.gamma_coeffs).sum(-1)
            p11 = torch.sigmoid(logit_p11)
            p12 = 1 - p11
            p21 = 0.0 # Assume no return from Regime 2, matching the DGP
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
            # ★★★ 修正点: 共分散の更新をJoseph Formに変更し、数値安定性を向上 ★★★
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
# Part 3: FRS Model Training
# ----------------------------------------------------------------
def train_frs_model(Y_data, init_params, cache_file):

    if os.path.exists(cache_file):
        print(f"Loading cached FRS model from '{cache_file}'...")
        return torch.load(cache_file, map_location=device, weights_only=False)
    print("\n--- 3. Training FRS Model ---")
    model = RegimeSwitchingKF(O, 3, 1, init_params=init_params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    best_loss = float('inf')
    patience_counter = 0
    patience = 100
    start_time = time.time()
    pbar = trange(10000, desc="Training FRS")
    for epoch in pbar:
        optimizer.zero_grad()
        loss = model(Y_data)
        if torch.isnan(loss):
            print("\nLoss became NaN. Stopping training.")
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break
    duration = time.time() - start_time
    print(f"FRS model training finished. Duration: {duration:.2f}s")
    results = {
        'model_state_dict': best_model_state,
        'loss': -best_loss,
        'duration': duration
    }
    torch.save(results, cache_file)
    print(f"FRS model results saved to '{cache_file}'.")
    return results

# ★★★ 移動先: 乱数シードをここに設定する ★★★
torch.manual_seed(123)
np.random.seed(123)

frs_initial_params = {
    'B1_3fac': best_params_3fac['B1'], 'B1_1fac': best_params_1fac['B1'],
    'lambda_free_3fac': torch.cat([
        best_params_3fac['Lambda1'][1:3, 0], 
        best_params_3fac['Lambda1'][4:6, 1], 
        best_params_3fac['Lambda1'][7:9, 2]
    ]),
    'lambda_free_1fac': best_params_1fac['Lambda1'][1:9, 0],
    'Q_3fac': best_params_3fac['Q'], 'Q_1fac': best_params_1fac['Q'],
    'R_3fac': best_params_3fac['R'], 'R_1fac': best_params_1fac['R'],
    'gamma_intercept': true_transition_params['gamma_intercept'],
    'gamma_coeffs': true_transition_params['gamma_coeffs']
}
frs_results = train_frs_model(Y_generated, frs_initial_params, FRS_MODEL_FILE)


# ----------------------------------------------------------------
# Part 4: ★★★ モデル評価 (BPSと基準を統一) ★★★
# ----------------------------------------------------------------
print("\n--- 4. Evaluating Final FRS Model ---")
final_model = RegimeSwitchingKF(O, 3, 1, init_params=frs_initial_params).to(device)
final_model.load_state_dict(frs_results['model_state_dict'])
final_model.eval()

with torch.no_grad():
    _ = final_model(Y_generated)
    y_preds_frs = final_model.predicted_y
    filtered_probs_frs = final_model.filtered_probs.cpu().numpy()

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

rmse_frs = calculate_rmse(Y_generated, y_preds_frs)
sens_frs, spec_frs = calculate_sens_spec(actual_states, filtered_probs_frs[:, :, 1])

# ----------------------------------------------------------------
# Part 5: 結果の要約
# ----------------------------------------------------------------
print("\n--- FRS Model Performance Summary ---")
table_width = 100
print("=" * table_width)
print(f"{'Metric':<25} | {'Value'}")
print("-" * table_width)
print(f"{'Final Log-Likelihood':<25} | {frs_results['loss']:.2f}")
print(f"{'Y Prediction RMSE':<25} | {rmse_frs:.4f}")
print(f"{'State Detection Sensitivity':<25} | {sens_frs:.4f} (Correctly identifying State 2)")
print(f"{'State Detection Specificity':<25} | {spec_frs:.4f} (Correctly identifying State 1)")
print(f"{'Training Duration (s)':<25} | {frs_results['duration']:.2f}")
print("=" * table_width)

print("\n--- Estimated vs. True Transition Parameters ---")
# ★★★ 修正点: FRSの推定値(p11のlogit)と比較するため、DGPの真の値(p12のlogit)の符号を反転させる ★★★
print(f"  gamma_intercept: Estimated = {final_model.gamma_intercept.item():.4f}, True = {-true_transition_params['gamma_intercept']}")
est_gamma_coeffs = final_model.gamma_coeffs.detach().cpu().numpy()
true_gamma_coeffs = true_transition_params['gamma_coeffs']
for i, name in enumerate(['task', 'goal', 'bond']):
    print(f"  gamma_{name}:      Estimated = {est_gamma_coeffs[i]:.4f}, True = {-true_gamma_coeffs[i]}")
# ----------------------------------------------------------------
# Part 6: Visualization
# ----------------------------------------------------------------
print("\n--- 6. Generating Visualizations ---")

# --- 6a. Aggregate Plot ---
state1_proportion_actual = (actual_states == 1).mean(axis=0)
avg_probs_frs_over_time = filtered_probs_frs.mean(axis=0)
time_points_vis = np.arange(Nt)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(time_points_vis, avg_probs_frs_over_time[:, 0], 'o-', color='royalblue', label='Avg. Prob. of 3-Factor Model (State 1)', zorder=3)
ax1.plot(time_points_vis, avg_probs_frs_over_time[:, 1], 's-', color='firebrick', label='Avg. Prob. of 1-Factor Model (State 2)', zorder=3)
ax1.set_title(f'FRS Model - Aggregate Performance ({DGP_MODE})', fontsize=16)
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

# ★★★ CHANGE: Save plot to the results directory ★★★
plot_filename = os.path.join(RESULTS_DIR, f"frs_aggregate_plot_{DGP_MODE}.png")
plt.savefig(plot_filename)
print(f"Aggregate plot saved to '{plot_filename}'")
plt.close(fig)

# --- 6b. Individual Plots ---
individuals_per_file = 10
num_files = math.ceil(N / individuals_per_file)
actual_states_binary = (actual_states == 2).astype(int)
print(f"Generating {num_files} files for individual-level plots...")
for file_idx in range(num_files):
    start_idx = file_idx * individuals_per_file
    end_idx = min((file_idx + 1) * individuals_per_file, N)
    ids_to_plot = range(start_idx, end_idx)
    fig, axes = plt.subplots(len(ids_to_plot), 1, figsize=(10, 2.5 * len(ids_to_plot)), sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f'FRS Individual State Probabilities (Individuals {start_idx+1}-{end_idx})', fontsize=16)
    for i, ind_id in enumerate(ids_to_plot):
        ax = axes[i, 0]
        ax.plot(time_points_vis, filtered_probs_frs[ind_id, :, 1], 'g-', label='FRS Prob. (State 2)', lw=2)
        ax.plot(time_points_vis, actual_states_binary[ind_id, :], 'r--', label='True State', lw=1.5)
        ax.set_title(f'Individual #{ind_id+1}', fontsize=10)
        ax.set_ylabel('Prob. of State 2')
        if i == 0: ax.legend(loc='upper left')
    axes[-1, 0].set_xlabel('Time Point')
    plt.ylim(-0.1, 1.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # ★★★ CHANGE: Save plot to the results directory ★★★
    save_filename = os.path.join(RESULTS_DIR, f"frs_individual_plots_{DGP_MODE}_part_{file_idx+1}.png")
    plt.savefig(save_filename)
    plt.close(fig)
print(f"Individual-level plots saved.")

sys.stdout = original_stdout
print(f"\nFRS analysis complete. Log saved to '{log_filename}'")