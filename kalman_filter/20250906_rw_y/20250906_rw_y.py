# FGME準備：Early Stopping機能付き 修正版スクリプト（完全版）

import math
import os
import random
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm import trange # tqdmをインポート
# (スクリプト冒頭のインポート文)

import torch.distributions.constraints as constraints
from torch.serialization import safe_globals
import pyro.distributions as dist # この行を追加、または変更

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

# ログファイルの設定
log_filename = f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, open(log_filename, 'w'))

# ----------------------------------------------------------------
# Part 0: パラメータとキャッシュファイル名の定義
# ----------------------------------------------------------------
print("--- 0. Defining Parameters ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

N = 100; Nt = 20; O = 9; J = 2
L1_state1 = 3; L1_state2 = 1

DATA_FILE = 'simulation_data_rw.pt'
MODEL_3FAC_FILE = 'trained_3fac_model.pt'
MODEL_1FAC_FILE = 'trained_1fac_model.pt'
BPS_GUIDE_FILE = 'trained_bps_guide_params.pt'

# ----------------------------------------------------------------
# Part 1: データ生成（ファイルが存在しない場合のみ実行）
# ----------------------------------------------------------------
print("\n--- 1. Generating Simulation Data ---")

data_file = 'simulation_data.pt' # 新しいDGP用のファイル名

if os.path.exists(data_file):
    print(f"Loading pre-computed simulation data from '{data_file}'...")
    saved_data = torch.load(data_file, weights_only=False)
    Y_generated = saved_data['Y_generated'].to(device)
    actual_states = saved_data['actual_states']
    eta_true_history = saved_data['eta_true_history'].to(device)
    print("Data loading complete.")
else:
    print("No pre-computed data file found. Generating new simulation data...")
    # DGP parameters
    b0_true_state1 = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(1)
    B1_true_state1 = torch.diag(torch.tensor([0.7, 0.7, 0.7], device=device))
    lambda1_true_values_state1 = [1.2, 0.8, 1.1, 0.9, 1.3, 0.7]
    Lambda1_true_state1 = torch.zeros(O, L1_state1, device=device)
    Lambda1_true_state1[0,0]=1; Lambda1_true_state1[1,0]=lambda1_true_values_state1[0]; Lambda1_true_state1[2,0]=lambda1_true_values_state1[1]
    Lambda1_true_state1[3,1]=1; Lambda1_true_state1[4,1]=lambda1_true_values_state1[2]; Lambda1_true_state1[5,1]=lambda1_true_values_state1[3]
    Lambda1_true_state1[6,2]=1; Lambda1_true_state1[7,2]=lambda1_true_values_state1[4]; Lambda1_true_state1[8,2]=lambda1_true_values_state1[5]
    
    b0_true_state2 = torch.tensor([0.0], device=device).unsqueeze(1)
    B1_true_state2 = torch.tensor([[0.9]], device=device)
    lambda1_true_values_state2 = [1.0] * 8
    Lambda1_true_state2 = torch.zeros(O, L1_state2, device=device)
    Lambda1_true_state2[0,0]=1.0
    Lambda1_true_state2[1:9,0] = torch.tensor(lambda1_true_values_state2, device=device)

    Q_state1=torch.eye(L1_state1, device=device); Q_state2=torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept=-2.5; gamma_task=0.1; gamma_goal=0.1; gamma_bond=0.1

    Y_generated = torch.zeros(N, Nt, O, device=device)
    actual_states = np.zeros((N, Nt))
    eta_true_history = torch.full((N, Nt, L1_state1), float('nan'), device=device) 
    
    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)
    
    for i in trange(N, desc="Generating data for each person"):
        eta_history_i = torch.randn(L1_state1, 1, device=device) # Renamed to avoid confusion
        current_state = 1; has_switched = False
        for t in range(Nt):
            actual_states[i, t] = current_state
            if current_state == 1 and t > 0:
                z = gamma_intercept + (eta_history_i[0]*gamma_task + eta_history_i[1]*gamma_goal + eta_history_i[2]*gamma_bond)
                if random.random() < (1 / (1 + math.exp(-z))): current_state = 2
            
            if current_state == 1:
                eta_t = (B1_true_state1 @ eta_history_i) + q_dist_s1.sample().reshape(L1_state1, 1)
                y_mean_t = Lambda1_true_state1 @ eta_t
            else: 
                if not has_switched:
                    eta_history_i = torch.tensor([eta_history_i.mean()], device=device).reshape(L1_state2, 1)
                    has_switched = True
                eta_t = (B1_true_state2 @ eta_history_i) + q_dist_s2.sample().reshape(L1_state2, 1)
                y_mean_t = Lambda1_true_state2 @ eta_t
            
            Y_generated[i, t, :] = (y_mean_t + r_dist.sample().reshape(O, 1)).squeeze()
            
            if current_state == 1:
                eta_true_history[i, t, :] = eta_t.squeeze()
            else:
                eta_true_history[i, t, 0] = eta_t.squeeze()
            eta_history_i = eta_t
    
    print("Simulation data generated.")
    print(f"Saving simulation data to '{data_file}'...")
    data_to_save = {
        'Y_generated': Y_generated.cpu(),
        'actual_states': actual_states,
        'eta_true_history': eta_true_history.cpu()
    }
    torch.save(data_to_save, data_file)
    print("Data saving complete.")

# ----------------------------------------------------------------
# Part 2: 共通関数の定義
# ----------------------------------------------------------------
print("\n--- 2. Defining Common Functions ---")
def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
    N, Nt, O1 = Y.shape; L1 = B1.shape[0]
    eta_prev = eta1_i0_0.expand(N, -1, -1); P_prev = P_i0_0.expand(N, -1, -1)
    total_log_likelihood = 0.0
    for t in range(Nt):
        y_t = Y[:, t, :].unsqueeze(-1)
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
        try:
            F_t.diagonal(dim1=-2, dim2=-1).add_(1e-6)
            dist = torch.distributions.MultivariateNormal(loc=v_t.squeeze(-1), covariance_matrix=F_t)
            total_log_likelihood += dist.log_prob(torch.zeros(O1, device=device)).sum()
        except torch.linalg.LinAlgError: return torch.tensor(float('nan'))
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_KL = torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_KL, P_pred), I_KL.transpose(-1, -2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(-1, -2))
        eta_prev = eta_updated; P_prev = P_updated
    return total_log_likelihood

def get_kalman_predictions(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape; L1 = B1.shape[0]
    y_pred_series = torch.zeros_like(Y);
    eta_prev = torch.zeros(N, L1, 1, device=device); P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    for t in range(Nt):
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        y_pred_series[:, t, :] = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred).squeeze(-1)
        v_t = Y[:, t, :].unsqueeze(-1) - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
        F_t.diagonal(dim1=-2, dim2=-1).add_(1e-6)
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        P_updated = torch.bmm((torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))), P_pred)
        eta_prev = eta_updated; P_prev = P_updated
    return y_pred_series

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()

def calculate_sens_spec(beta_weights, true_states):
    predicted_states = (beta_weights[:, 1] > beta_weights[:, 0]).astype(int) + 1
    true_positives = np.sum((predicted_states == 2) & (true_states == 2))
    false_negatives = np.sum((predicted_states == 1) & (true_states == 2))
    true_negatives = np.sum((predicted_states == 1) & (true_states == 1))
    false_positives = np.sum((predicted_states == 2) & (true_states == 1))
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    return sensitivity, specificity

# ----------------------------------------------------------------
# Part 3: ベースラインモデルの推定
# ----------------------------------------------------------------
print("\n--- 3. Estimation of Baseline Models ---")

# --- 3a. 3-Factor Model ---
print("\n--- 3a. 3-Factor Model ---")
if os.path.exists(MODEL_3FAC_FILE):
    print(f"Loading pre-trained 3-Factor model from '{MODEL_3FAC_FILE}'...")
    saved_model_3fac = torch.load(MODEL_3FAC_FILE, map_location=device)
    best_params_3fac = saved_model_3fac['params']
    loss_3fac = saved_model_3fac['loss']
else:
    print("Training 3-Factor model...")
    patience = 100
    patience_counter = 0
    best_loss_3fac = float('inf'); best_params_3fac = {}
    
    # --- ここから置き換え ---
    # DGP with noise for initialization
    b0_true_state1 = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(1)
    b0_3fac = (b0_true_state1 + torch.randn_like(b0_true_state1) * 0.1).requires_grad_(True)

    b1_true_state1 = torch.tensor([0.7, 0.7, 0.7], device=device)
    b1_free_params_3fac = (b1_true_state1 + torch.randn_like(b1_true_state1) * 0.1).requires_grad_(True)

    lambda1_true_state1 = torch.tensor([1.2, 0.8, 1.1, 0.9, 1.3, 0.7], device=device)
    lambda1_free_params_3fac = (lambda1_true_state1 + torch.randn_like(lambda1_true_state1) * 0.1).requires_grad_(True)

    log_q_diag_3fac = torch.log(torch.ones(L1_state1, device=device) * 0.5).requires_grad_(True)
    log_r_diag_3fac = torch.log(torch.ones(O, device=device) * 0.5).requires_grad_(True)

    params_to_learn_3fac = [b0_3fac, b1_free_params_3fac, lambda1_free_params_3fac, log_q_diag_3fac, log_r_diag_3fac]
    optimizer_3fac = torch.optim.AdamW(params_to_learn_3fac, lr=0.001, weight_decay=0.01)
    pbar = trange(20000)
    for epoch in pbar:
        optimizer_3fac.zero_grad()
        Q_est_3fac = torch.diag(torch.exp(log_q_diag_3fac)); R_est_3fac = torch.diag(torch.exp(log_r_diag_3fac))
        B1_3fac = torch.diag(b1_free_params_3fac); Lambda1_3fac = torch.zeros(O, L1_state1, device=device)
        Lambda1_3fac[0,0]=1; Lambda1_3fac[1,0]=lambda1_free_params_3fac[0]; Lambda1_3fac[2,0]=lambda1_free_params_3fac[1]; Lambda1_3fac[3,1]=1; Lambda1_3fac[4,1]=lambda1_free_params_3fac[2]
        Lambda1_3fac[5,1]=lambda1_free_params_3fac[3]; Lambda1_3fac[6,2]=1; Lambda1_3fac[7,2]=lambda1_free_params_3fac[4]; Lambda1_3fac[8,2]=lambda1_free_params_3fac[5]
        loss = -kalman_filter_torch_loss(Y_generated, b0_3fac, B1_3fac, Lambda1_3fac, Q_est_3fac, R_est_3fac, torch.zeros(L1_state1, 1, device=device), torch.eye(L1_state1, device=device) * 1e3)
        if torch.isnan(loss): break
        loss.backward(); optimizer_3fac.step()
        pbar.set_description(f"[3-Fac Epoch {epoch+1}] loss: {loss.item():.4f} (Best: {best_loss_3fac:.4f})")
        
        if loss.item() < best_loss_3fac:
            best_loss_3fac = loss.item()
            best_params_3fac = {'b0': b0_3fac.detach(), 'B1': B1_3fac.detach(), 'Lambda1': Lambda1_3fac.detach(), 'Q': Q_est_3fac.detach(), 'R': R_est_3fac.detach()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n   -> Early stopping triggered at epoch {epoch + 1}.")
            break
            
    loss_3fac = best_loss_3fac
    print(f"Finished training. Saving best model to '{MODEL_3FAC_FILE}'...")
    torch.save({'params': best_params_3fac, 'loss': loss_3fac}, MODEL_3FAC_FILE)

# --- 3b. 1-Factor Model ---
print("\n--- 3b. 1-Factor Model ---")
if os.path.exists(MODEL_1FAC_FILE):
    print(f"Loading pre-trained 1-Factor model from '{MODEL_1FAC_FILE}'...")
    saved_model_1fac = torch.load(MODEL_1FAC_FILE, map_location=device)
    best_params_1fac = saved_model_1fac['params']
    loss_1fac = saved_model_1fac['loss']
else:
    print("Training 1-Factor model...")
    patience = 100
    patience_counter = 0
    best_loss_1fac = float('inf'); best_params_1fac = {}

    # --- ここから置き換え ---
    # DGP with noise for initialization
    b0_true_state2 = torch.tensor([0.0], device=device).unsqueeze(1)
    b0_1fac = (b0_true_state2 + torch.randn_like(b0_true_state2) * 0.1).requires_grad_(True)

    B1_true_state2 = torch.tensor([[0.9]], device=device)
    B1_1fac = (B1_true_state2 + torch.randn_like(B1_true_state2) * 0.1).requires_grad_(True)
    
    lambda1_true_state2 = torch.tensor([1.0] * 8, device=device)
    lambda1_free_params_1fac = (lambda1_true_state2 + torch.randn_like(lambda1_true_state2) * 0.1).requires_grad_(True)

    log_q_diag_1fac = torch.log(torch.ones(L1_state2, device=device) * 0.5).requires_grad_(True)
    log_r_diag_1fac = torch.log(torch.ones(O, device=device) * 0.5).requires_grad_(True)
    # --- ここまで置き換え ---
    
    params_to_learn_1fac = [b0_1fac, B1_1fac, lambda1_free_params_1fac, log_q_diag_1fac, log_r_diag_1fac]
    optimizer_1fac = torch.optim.AdamW(params_to_learn_1fac, lr=0.001, weight_decay=0.01)
    pbar = trange(20000)
    for epoch in pbar:
        optimizer_1fac.zero_grad()
        Q_est_1fac = torch.diag(torch.exp(log_q_diag_1fac)); R_est_1fac = torch.diag(torch.exp(log_r_diag_1fac))
        Lambda1_1fac = torch.zeros(O, L1_state2, device=device); Lambda1_1fac[0,0]=1.0; Lambda1_1fac[1:9,0] = lambda1_free_params_1fac[0:8]
        loss = -kalman_filter_torch_loss(Y_generated, b0_1fac, B1_1fac, Lambda1_1fac, Q_est_1fac, R_est_1fac, torch.zeros(L1_state2, 1, device=device), torch.eye(L1_state2, device=device) * 1e3)
        if torch.isnan(loss): break
        loss.backward(); optimizer_1fac.step()
        pbar.set_description(f"[1-Fac Epoch {epoch+1}] loss: {loss.item():.4f} (Best: {best_loss_1fac:.4f})")
        
        if loss.item() < best_loss_1fac:
            best_loss_1fac = loss.item()
            best_params_1fac = {'b0': b0_1fac.detach(), 'B1': B1_1fac.detach(), 'Lambda1': Lambda1_1fac.detach(), 'Q': Q_est_1fac.detach(), 'R': R_est_1fac.detach()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n   -> Early stopping triggered at epoch {epoch + 1}.")
            break

    loss_1fac = best_loss_1fac
    print(f"Finished training. Saving best model to '{MODEL_1FAC_FILE}'...")
    torch.save({'params': best_params_1fac, 'loss': loss_1fac}, MODEL_1FAC_FILE)

# ----------------------------------------------------------------
# Part 4: BPSモデルの定義と学習
# ----------------------------------------------------------------
print("\n--- 4. Defining and Training BPS Random-Walk Model ---")
print("Pre-calculating inputs for BPS model...")
with torch.no_grad():
    preds_3fac_est = get_kalman_predictions(Y_generated, **best_params_3fac)
    preds_1fac_est = get_kalman_predictions(Y_generated, **best_params_1fac)
    factors_f = torch.stack((preds_3fac_est, preds_1fac_est), dim=3)
print("Pre-calculation complete.")

# (Part 4内)
def bps_random_walk_model(y_obs, factors_f):
    N, Nt, O, J = factors_f.shape
    
    # --- 事前分布（変更なし） ---
    log_v_diag = pyro.sample("log_v_diag", dist.Normal(0.0, 1.0).expand([J]).to_event(1))
    V = torch.diag(torch.exp(log_v_diag))
    log_tau = pyro.sample("log_tau", dist.Normal(0.0, 1.0))
    tau = torch.exp(log_tau)
    log_sigma_diag = pyro.sample("log_sigma_diag", dist.Normal(0.0, 1.0).expand([O]).to_event(1))
    Sigma = torch.diag(torch.exp(log_sigma_diag))
    
    # ▼▼▼【変更点１】▼▼▼
    # logit_w_t の初期値の事前分布を設定
    # t=0で3因子モデルの確率がほぼ1になるよう、logitに大きな差をつける
    logit_w_t0_mean = torch.tensor([5.0, -5.0], device=device) # 例: softmax([5,-5]) -> [0.9999, 0.0001]
    logit_w_t0_std = torch.tensor([0.1, 0.1], device=device)
    logit_w_t0 = pyro.sample("logit_w_t0", dist.Normal(logit_w_t0_mean, logit_w_t0_std).to_event(1))
    
    alpha_t0 = pyro.sample("alpha_t0", dist.Normal(0., 1.).expand([O]).to_event(1))

    # --- 潜在変数の軌道を生成 ---
    logit_w_trajectory = []
    alpha_trajectory = []
    logit_w_t = logit_w_t0
    alpha_t = alpha_t0
    for t in range(Nt):
        logit_w_t = pyro.sample(f"logit_w_{t}", dist.MultivariateNormal(logit_w_t, V))
        alpha_t = pyro.sample(f"alpha_{t}", dist.MultivariateNormal(alpha_t, torch.eye(O, device=device) * tau**2))
        logit_w_trajectory.append(logit_w_t)
        alpha_trajectory.append(alpha_t)
    
    logit_w_trajectory = torch.stack(logit_w_trajectory)
    alpha_trajectory = torch.stack(alpha_trajectory)
    
    # ▼▼▼【変更点２】▼▼▼
    # logitから確率的な重みw_tを計算 (Softmax)
    w_trajectory = torch.nn.functional.softmax(logit_w_trajectory, dim=-1)
    
    # --- 観測モデル ---
    # ▼▼▼【変更点３】▼▼▼
    # 重みw_tを使って予測値を加重平均する
    y_mean = (factors_f @ w_trajectory.view(1, Nt, J, 1)).squeeze(-1) + alpha_trajectory.view(1, Nt, O)
    with pyro.plate("individuals", N, dim=-2):
        pyro.sample("obs", dist.MultivariateNormal(y_mean, Sigma).to_event(1), obs=y_obs)

guide = AutoDiagonalNormal(bps_random_walk_model)
optimizer = Adam({"lr": 0.005})
svi = SVI(bps_random_walk_model, guide, optimizer, loss=Trace_ELBO())

if os.path.exists(BPS_GUIDE_FILE):
    print(f"Loading pre-trained BPS model guide from '{BPS_GUIDE_FILE}'...")
    pyro.clear_param_store()
    # ▼▼▼▼▼【修正点】▼▼▼▼▼
    # 安全に読み込むクラスのリストに_SoftplusPositiveを追加
    with safe_globals([constraints._Real, dist.constraints._SoftplusPositive]):
        pyro.get_param_store().load(BPS_GUIDE_FILE, map_location=device)
    # ▲▲▲▲▲【修正点】▲▲▲▲▲
    loss_bps = -9999
else:
    print("Starting BPS model training with VI...")
    patience = 100
    patience_counter = 0
    best_loss_bps = float('inf')
    pyro.clear_param_store()
    pbar = trange(20000)
    for step in pbar:
        loss = svi.step(Y_generated, factors_f) / N
        if torch.isnan(torch.tensor(loss)):
            print("Loss became NaN. Stopping training.")
            break
        pbar.set_description(f"[BPS SVI step {step+1}] ELBO loss: {loss:.4f} (Best: {best_loss_bps:.4f})")
        
        if loss < best_loss_bps:
            best_loss_bps = loss
            patience_counter = 0
            # 損失が改善するたびに、最も良いモデルを保存する
            pyro.get_param_store().save(BPS_GUIDE_FILE)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n   -> Early stopping triggered at step {step + 1}.")
            break
    
    loss_bps = best_loss_bps
    print(f"Finished training. Loading best model state from '{BPS_GUIDE_FILE}'...")
    pyro.clear_param_store()
    # ▼▼▼▼▼【修正点】▼▼▼▼▼
    # こちらも同様に修正
    with safe_globals([constraints._Real, dist.constraints._SoftplusPositive]):
        pyro.get_param_store().load(BPS_GUIDE_FILE, map_location=device)
    # ▲▲▲▲▲【修正点】▲▲▲▲▲
    # 最後に必ずベストのパラメータをロードする

# ----------------------------------------------------------------
# Part 5: モデルの評価と可視化
# ----------------------------------------------------------------
print("\n--- 5. Model Evaluation and Visualization ---")
# (This part is simplified for brevity, assuming the full evaluation logic is here)
# ▼▼▼▼▼【ここからが修正点】▼▼▼▼▼
# guide.median()を呼び出す前に、一度guideを実行して内部状態を初期化する
print("Initializing the guide...")
with torch.no_grad():
    guide(Y_generated, factors_f)
print("Guide initialized.")
# ▲▲▲▲▲【ここまでが修正点】▲▲▲▲▲

# これでguideが初期化されたので、安全にmedianを呼び出せる
median_params = guide.median() # 引数は不要になりま

# logit_wを抽出し、softmaxで重みに変換する
estimated_logits = torch.stack([median_params[f'logit_w_{t}'] for t in range(Nt)]).cpu()
estimated_weights = torch.nn.functional.softmax(estimated_logits, dim=-1).numpy()

estimated_alphas = torch.stack([median_params[f'alpha_{t}'] for t in range(Nt)]).cpu().numpy()


# Plotting
print("\n--- Visualizing BPS Random-Walk Weights ---")
time_points = np.arange(Nt)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Estimated Beta and Alpha Weights Over Time', fontsize=16)

# プロット作成時の変数名をbetaからweightに変更
ax1.plot(time_points, estimated_weights[:, 0], 'o-', color='royalblue', label='Weight for 3-Factor Model (beta_1)')
ax1.plot(time_points, estimated_weights[:, 1], 's-', color='firebrick', label='Weight for 1-Factor Model (beta_2)')
ax1.set_ylabel('Model Weight (Probability)') # Y軸ラベルも変更
ax1.grid(True)
ax1.legend(loc='upper left')

ax1_twin = ax1.twinx()
state1_proportion = (actual_states == 1).mean(axis=0)
ax1_twin.bar(time_points, state1_proportion, color='grey', alpha=0.3, label='Proportion in State 1 (Actual)')
ax1_twin.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12)
ax1_twin.set_ylim(-0.05, 1.05)
ax1_twin.legend(loc='upper right')

mean_alphas = estimated_alphas.mean(axis=1)
ax2.plot(time_points, mean_alphas, '^-', color='green', label='Mean of Alpha Weights')
ax2.axhline(0, color='grey', linestyle='--')
ax2.set_xlabel('Time Point', fontsize=12)
ax2.set_ylabel('Mean Alpha Weight Value', fontsize=12)
ax2.grid(True)
ax2.legend(loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plot_filename = "final_alpha_beta_weights_plot.png"
plt.savefig(plot_filename)
print(f"Combined alpha and beta plot saved to '{plot_filename}'")

sys.stdout = original_stdout
print(f"\nAnalysis complete. Log saved to '{log_filename}'")