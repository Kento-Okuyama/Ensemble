# FGME準備：Early Stopping機能付き 修正版スクリプト（完全版）
# + 0904スクリプトの個別プロット機能（信頼区間付き）を統合

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
from pyro.infer import SVI, Trace_ELBO, Predictive # Predictive をインポート
from pyro.optim import Adam
from torch.distributions.multivariate_normal import MultivariateNormal
from pyro.infer.autoguide import AutoDiagonalNormal
from tqdm import trange

import torch.distributions.constraints as constraints
from torch.serialization import safe_globals

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

# ▼▼▼【変更点】▼▼▼
# Ntを25に増やし、Train-Test Splitを導入
N = 100; Nt = 25; O = 9; J = 2
L1_state1 = 3; L1_state2 = 1
Nt_train = 20 # 訓練期間
Nt_test = Nt - Nt_train # テスト期間
print(f"Time series split: {Nt_train} for training, {Nt_test} for testing.")
# ▲▲▲【変更ここまで】▲▲▲

# DGP parameters (defined globally to be accessible by all parts)
gamma_intercept=-2.5; gamma_task=0.1; gamma_goal=0.1; gamma_bond=0.1

# Cache file names
DATA_FILE = 'simulation_data_rw_forecast.pt' # ファイル名を変更して再生成を促す
MODEL_3FAC_FILE = 'trained_3fac_model_forecast.pt'
MODEL_1FAC_FILE = 'trained_1fac_model_forecast.pt'
BPS_GUIDE_FILE = 'trained_bps_guide_params_forecast.pt'
RS_KF_MODEL_FILE = 'rs_kf_results_forecast.pt'

# ----------------------------------------------------------------
# Part 1: データ生成（ファイルが存在しない場合のみ実行）
# ----------------------------------------------------------------
print("\n--- 1. Generating Simulation Data ---")

if os.path.exists(DATA_FILE):
    print(f"Loading pre-computed simulation data from '{DATA_FILE}'...")
    saved_data = torch.load(DATA_FILE, weights_only=False)
    Y_generated = saved_data['Y_generated'].to(device)
    actual_states = saved_data['actual_states']
    eta_true_history = saved_data['eta_true_history'].to(device)
    print("Data loading complete.")
else:
    print(f"No pre-computed data file found. Generating new simulation data for Nt={Nt}...")
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

    Y_generated = torch.zeros(N, Nt, O, device=device)
    actual_states = np.zeros((N, Nt))
    eta_true_history = torch.full((N, Nt, L1_state1), float('nan'), device=device)

    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)

    for i in trange(N, desc="Generating data for each person"):
        eta_history_i = torch.randn(L1_state1, 1, device=device)
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
    print(f"Saving simulation data to '{DATA_FILE}'...")
    data_to_save = {
        'Y_generated': Y_generated.cpu(),
        'actual_states': actual_states,
        'eta_true_history': eta_true_history.cpu()
    }
    torch.save(data_to_save, DATA_FILE)
    print("Data saving complete.")

# ▼▼▼【追加】▼▼▼
# 訓練データとテストデータに分割
Y_train = Y_generated[:, :Nt_train, :]
Y_test = Y_generated[:, Nt_train:, :]
actual_states_train = actual_states[:, :Nt_train]
actual_states_test = actual_states[:, Nt_train:]
eta_true_history_train = eta_true_history[:, :Nt_train, :]
eta_true_history_test = eta_true_history[:, Nt_train:, :]
# ▲▲▲【追加ここまで】▲▲▲

# ----------------------------------------------------------------
# Part 2: 共通関数の定義
# ----------------------------------------------------------------
print("\n--- 2. Defining Common Functions ---")
def kalman_filter_torch_loss(Y, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
    N, T, O1 = Y.shape; L1 = B1.shape[0] # Nt -> T
    eta_prev = eta1_i0_0.expand(N, -1, -1); P_prev = P_i0_0.expand(N, -1, -1)
    total_log_likelihood = 0.0
    for t in range(T): # Nt -> T
        y_t = Y[:, t, :].unsqueeze(-1)
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
        try:
            F_t.diagonal(dim1=-2, dim2=-1).add_(1e-6)
            # 修正: locとobsを分離
            dist = torch.distributions.MultivariateNormal(loc=torch.zeros_like(v_t.squeeze(-1)), covariance_matrix=F_t)
            total_log_likelihood += dist.log_prob(v_t.squeeze(-1)).sum()
        except torch.linalg.LinAlgError: return torch.tensor(float('nan'))
        F_inv_t = torch.linalg.pinv(F_t)
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), F_inv_t)
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_KL = torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_KL, P_pred), I_KL.transpose(-1, -2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(-1, -2))
        eta_prev = eta_updated; P_prev = P_updated
    return total_log_likelihood

# ▼▼▼【関数追加】▼▼▼
# 0904スクリプトから `get_kalman_predictions_and_latents` を移植・修正
def get_kalman_predictions_and_latents(Y, b0, B1, Lambda1, Q, R):
    N, T, O1 = Y.shape # Nt -> T
    L1 = B1.shape[0]
    
    y_pred_series = torch.zeros_like(Y)
    eta_series = torch.zeros(N, T, L1, device=device)
    P_series = torch.zeros(N, T, L1, L1, device=device) # Pの履歴を保存
    
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
    
    for t in range(T): # Nt -> T
        y_t = Y[:, t, :].unsqueeze(-1)
        
        eta_pred = b0 + torch.bmm(B1.expand(N, -1, -1), eta_prev)
        y_pred_series[:, t, :] = torch.bmm(Lambda1.expand(N, -1, -1), eta_pred).squeeze(-1)
        
        v_t = y_t - torch.bmm(Lambda1.expand(N, -1, -1), eta_pred)
        P_pred = torch.bmm(torch.bmm(B1.expand(N, -1, -1), P_prev), B1.transpose(0, 1).expand(N, -1, -1)) + Q
        F_t = torch.bmm(torch.bmm(Lambda1.expand(N, -1, -1), P_pred), Lambda1.transpose(0, 1).expand(N, -1, -1)) + R
        
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
        
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        P_updated = torch.bmm((torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))), P_pred)
        
        eta_series[:, t, :] = eta_updated.squeeze(-1)
        P_series[:, t, :, :] = P_updated # 更新されたPを保存
        
        eta_prev = eta_updated
        P_prev = P_updated
        
    return y_pred_series, eta_series, P_series # P_seriesも返す

# get_kalman_predictions は get_kalman_predictions_and_latents で代替できるため、元の関数はコメントアウト
# def get_kalman_predictions(Y, b0, B1, Lambda1, Q, R): ...
# ▲▲▲【関数追加ここまで】▲▲▲

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()

def calculate_sens_spec(actual_states, predicted_states_binary):
    ground_truth_binary = (actual_states == 2).astype(int)
    TP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 1)); FN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 1))
    TN = np.sum((predicted_states_binary == 0) & (ground_truth_binary == 0)); FP = np.sum((predicted_states_binary == 1) & (ground_truth_binary == 0))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0; specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return sensitivity, specificity

# ▼▼▼【関数追加】▼▼▼
# 0904スクリプトから `generate_forecasts` と `forecast_rs` を移植
def generate_forecasts(model_params, Y_full_data, train_steps):
    """学習済みカルマンフィルタモデルを使い、1期先予測を生成する"""
    N, Nt, O1 = Y_full_data.shape
    L1 = model_params['B1'].shape[0]
    test_steps = Nt - train_steps
    
    y_forecasts = torch.zeros(N, test_steps, O1, device=device)
    eta_forecasts = torch.zeros(N, test_steps, L1, device=device)
    P_forecasts = torch.zeros(N, test_steps, L1, L1, device=device)

    # 1. 訓練データ全体でフィルタリングを行い、最終状態を取得
    _, eta_series_train, P_series_train = get_kalman_predictions_and_latents(
        Y_full_data[:, :train_steps, :], model_params['b0'], model_params['B1'], 
        model_params['Lambda1'], model_params['Q'], model_params['R']
    )
    eta_prev = eta_series_train[:, -1, :].unsqueeze(-1)
    P_prev = P_series_train[:, -1, :, :]

    # 2. テスト期間を1ステップずつ予測
    with torch.no_grad():
        for t_idx, t in enumerate(range(train_steps, Nt)):
            # Prediction Step
            eta_pred = model_params['b0'] + torch.bmm(model_params['B1'].expand(N, -1, -1), eta_prev)
            P_pred = torch.bmm(torch.bmm(model_params['B1'].expand(N, -1, -1), P_prev), model_params['B1'].transpose(0, 1).expand(N, -1, -1)) + model_params['Q']
            y_pred_t = torch.bmm(model_params['Lambda1'].expand(N, -1, -1), eta_pred).squeeze(-1)
            
            y_forecasts[:, t_idx, :] = y_pred_t
            eta_forecasts[:, t_idx, :] = eta_pred.squeeze(-1)
            P_forecasts[:, t_idx, :, :] = P_pred

            # Update Step (for next prediction)
            y_t_actual = Y_full_data[:, t, :].unsqueeze(-1)
            v_t = y_t_actual - torch.bmm(model_params['Lambda1'].expand(N, -1, -1), eta_pred)
            F_t = torch.bmm(torch.bmm(model_params['Lambda1'].expand(N, -1, -1), P_pred), model_params['Lambda1'].transpose(0, 1).expand(N, -1, -1)) + model_params['R']
            K_t = torch.bmm(torch.bmm(P_pred, model_params['Lambda1'].transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
            eta_updated = eta_pred + torch.bmm(K_t, v_t)
            P_updated = torch.bmm((torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, model_params['Lambda1'].expand(N, -1, -1))), P_pred)
            eta_prev, P_prev = eta_updated, P_updated

    return y_forecasts, eta_forecasts, P_forecasts

def forecast_rs(model_rs, Y_full_data, train_steps):
    """学習済みRS-KFモデルを使い、1期先予測を生成する"""
    N, Nt, O = Y_full_data.shape
    test_steps = Nt - train_steps
    
    # 1. 訓練期間のデータでフィルターを回し、最終状態を取得
    with torch.no_grad():
        _ = model_rs(Y_full_data[:, :train_steps, :])
        prob_prev = model_rs.filtered_probs[:, -1, :]
        eta_r1_prev = model_rs.eta_filtered_r1_history[:, -1, :, None]
        eta_r2_prev = model_rs.eta_filtered_r2_history[:, -1, :, None]
        P_r1_prev = model_rs.P_filtered_r1_history[:, -1, :, :]
        P_r2_prev = model_rs.P_filtered_r2_history[:, -1, :, :]

    # 2. テスト期間で1期先予測を繰り返す
    y_forecasts_rs = torch.zeros(N, test_steps, O, device=device)
    probs_forecast_rs = torch.zeros(N, test_steps, 2, device=device)
    eta_r1_forecasts_rs = torch.zeros(N, test_steps, model_rs.L, device=device)
    eta_r2_forecasts_rs = torch.zeros(N, test_steps, model_rs.L, device=device)
    P_r1_forecasts_rs = torch.zeros(N, test_steps, model_rs.L, model_rs.L, device=device)
    P_r2_forecasts_rs = torch.zeros(N, test_steps, model_rs.L, model_rs.L, device=device)

    with torch.no_grad():
        (B11, B21, B12, B22), (L1_lambda, L2_lambda), (Q, R) = model_rs._build_matrices()
        
        for t_idx, t in enumerate(range(train_steps, Nt)):
            y_t = Y_full_data[:, t, :].unsqueeze(-1)
            # 1ステップ分のフィルター処理 (model_rs.forwardから抜粋・修正)
            eta_state1_components = eta_r1_prev[:, :model_rs.L-1, :].squeeze(-1)
            logit_p11 = model_rs.gamma1 + (eta_state1_components * model_rs.gamma2_learnable).sum(-1)
            p11 = torch.sigmoid(logit_p11); p22 = torch.full((N,), 0.9999, device=device)
            transition_probs = torch.stack([torch.stack([p11, 1-p11], dim=1), torch.stack([1-p22, p22], dim=1)], dim=1)
            
            eta_pred_11 = B11 @ eta_r1_prev; P_pred_11 = B11 @ P_r1_prev @ B11.mT + Q
            eta_pred_22 = B22 @ eta_r2_prev; P_pred_22 = B22 @ P_r2_prev @ B22.mT + Q
            
            y_pred_for_t = prob_prev[:,0].view(N,1,1)* (L1_lambda @ eta_pred_11) + \
                           prob_prev[:,1].view(N,1,1)* (L2_lambda @ eta_pred_22)
            y_forecasts_rs[:, t_idx, :] = y_pred_for_t.squeeze(-1)
            
            # (Update and Collapsing steps...)
            eta_pred_12 = B21 @ eta_r2_prev; P_pred_12 = B21 @ P_r2_prev @ B21.mT + Q
            eta_pred_21 = B12 @ eta_r1_prev; P_pred_21 = B12 @ P_r1_prev @ B12.mT + Q
            v_11 = y_t - L1_lambda @ eta_pred_11; F_11 = L1_lambda @ P_pred_11 @ L1_lambda.mT + R
            v_12 = y_t - L1_lambda @ eta_pred_12; F_12 = L1_lambda @ P_pred_12 @ L1_lambda.mT + R
            v_21 = y_t - L2_lambda @ eta_pred_21; F_21 = L2_lambda @ P_pred_21 @ L2_lambda.mT + R
            v_22 = y_t - L2_lambda @ eta_pred_22; F_22 = L2_lambda @ P_pred_22 @ L2_lambda.mT + R
            f_jitter = torch.eye(O, device=device) * 1e-5
            F_11 += f_jitter; F_12 += f_jitter; F_21 += f_jitter; F_22 += f_jitter
            log_lik_11 = MultivariateNormal(loc=torch.zeros_like(v_11.squeeze(-1)), covariance_matrix=F_11).log_prob(v_11.squeeze(-1))
            log_lik_12 = MultivariateNormal(loc=torch.zeros_like(v_12.squeeze(-1)), covariance_matrix=F_12).log_prob(v_12.squeeze(-1))
            log_lik_21 = MultivariateNormal(loc=torch.zeros_like(v_21.squeeze(-1)), covariance_matrix=F_21).log_prob(v_21.squeeze(-1))
            log_lik_22 = MultivariateNormal(loc=torch.zeros_like(v_22.squeeze(-1)), covariance_matrix=F_22).log_prob(v_22.squeeze(-1))
            log_prob_prev = torch.log(prob_prev + 1e-9)
            log_prob_t_11 = log_prob_prev[:, 0] + torch.log(transition_probs[:, 0, 0] + 1e-9) + log_lik_11
            log_prob_t_12 = log_prob_prev[:, 1] + torch.log(transition_probs[:, 1, 0] + 1e-9) + log_lik_12
            log_prob_t_21 = log_prob_prev[:, 0] + torch.log(transition_probs[:, 0, 1] + 1e-9) + log_lik_21
            log_prob_t_22 = log_prob_prev[:, 1] + torch.log(transition_probs[:, 1, 1] + 1e-9) + log_lik_22
            log_prob_t = torch.stack([log_prob_t_11, log_prob_t_12, log_prob_t_21, log_prob_t_22], dim=1)
            log_likelihood_t = torch.logsumexp(log_prob_t, dim=1)
            prob_t = torch.exp(log_prob_t - log_likelihood_t.unsqueeze(1))
            prob_t_r1 = prob_t[:, 0] + prob_t[:, 1]; prob_t_r2 = prob_t[:, 2] + prob_t[:, 3]
            W_21 = prob_t[:, 2] / (prob_t_r2 + 1e-9); W_22 = prob_t[:, 3] / (prob_t_r2 + 1e-9)
            K_11 = P_pred_11 @ L1_lambda.mT @ torch.linalg.pinv(F_11); eta_upd_11 = eta_pred_11 + K_11 @ v_11
            K_21 = P_pred_21 @ L2_lambda.mT @ torch.linalg.pinv(F_21); eta_upd_21 = eta_pred_21 + K_21 @ v_21
            K_22 = P_pred_22 @ L2_lambda.mT @ torch.linalg.pinv(F_22); eta_upd_22 = eta_pred_22 + K_22 @ v_22
            I_L = torch.eye(model_rs.L, device=device)
            I_KL_11 = I_L - K_11 @ L1_lambda; P_upd_11 = I_KL_11 @ P_pred_11 @ I_KL_11.mT + K_11 @ R @ K_11.mT
            I_KL_21 = I_L - K_21 @ L2_lambda; P_upd_21 = I_KL_21 @ P_pred_21 @ I_KL_21.mT + K_21 @ R @ K_21.mT
            I_KL_22 = I_L - K_22 @ L2_lambda; P_upd_22 = I_KL_22 @ P_pred_22 @ I_KL_22.mT + K_22 @ R @ K_22.mT
            eta_marg_r1_t = eta_upd_11; P_marg_r1_t = P_upd_11
            eta_marg_r2_t = W_21.view(N, 1, 1) * eta_upd_21 + W_22.view(N, 1, 1) * eta_upd_22
            P_marg_r2_t = W_21.view(N,1,1) * (P_upd_21 + (eta_marg_r2_t-eta_upd_21) @ (eta_marg_r2_t-eta_upd_21).transpose(-1,-2)) + W_22.view(N,1,1) * (P_upd_22 + (eta_marg_r2_t-eta_upd_22) @ (eta_marg_r2_t-eta_upd_22).transpose(-1,-2))
            
            eta_r1_forecasts_rs[:, t_idx, :] = eta_marg_r1_t.squeeze(-1)
            eta_r2_forecasts_rs[:, t_idx, :] = eta_marg_r2_t.squeeze(-1)
            P_r1_forecasts_rs[:, t_idx, :, :] = P_marg_r1_t
            P_r2_forecasts_rs[:, t_idx, :, :] = P_marg_r2_t
            
            eta_r1_prev, P_r1_prev = eta_marg_r1_t, P_marg_r1_t
            eta_r2_prev, P_r2_prev = eta_marg_r2_t, P_marg_r2_t
            prob_prev = torch.stack([prob_t_r1, prob_t_r2], dim=1)
            probs_forecast_rs[:, t_idx, :] = prob_prev
    
    return y_forecasts_rs, probs_forecast_rs, eta_r1_forecasts_rs, eta_r2_forecasts_rs, P_r1_forecasts_rs, P_r2_forecasts_rs
# ▲▲▲【関数追加ここまで】▲▲▲

# ----------------------------------------------------------------
# Part 3: ベースラインモデルの推定
# ----------------------------------------------------------------
print("\n--- 3. Estimation of Baseline Models (on Training Data) ---")

# --- 3a. 3-Factor Model ---
print("\n--- 3a. 3-Factor Model ---")
if os.path.exists(MODEL_3FAC_FILE):
    print(f"Loading pre-trained 3-Factor model from '{MODEL_3FAC_FILE}'...")
    saved_model_3fac = torch.load(MODEL_3FAC_FILE, map_location=device)
    best_params_3fac = saved_model_3fac['params']
    loss_3fac = saved_model_3fac['loss']
else:
    print("Training 3-Factor model...")
    patience = 100; patience_counter = 0; best_loss_3fac = float('inf'); best_params_3fac = {}
    b0_3fac = torch.randn(L1_state1, 1, device=device).requires_grad_(True)
    b1_free_params_3fac = torch.randn(L1_state1, device=device).requires_grad_(True)
    lambda1_free_params_3fac = torch.randn(6, device=device).requires_grad_(True)
    log_q_diag_3fac = torch.zeros(L1_state1, device=device).requires_grad_(True)
    log_r_diag_3fac = torch.zeros(O, device=device).requires_grad_(True)
    params_to_learn_3fac = [b0_3fac, b1_free_params_3fac, lambda1_free_params_3fac, log_q_diag_3fac, log_r_diag_3fac]
    optimizer_3fac = torch.optim.AdamW(params_to_learn_3fac, lr=0.001, weight_decay=0.01)
    pbar = trange(20000)
    for epoch in pbar:
        optimizer_3fac.zero_grad()
        Q_est_3fac = torch.diag(torch.exp(log_q_diag_3fac)); R_est_3fac = torch.diag(torch.exp(log_r_diag_3fac))
        B1_3fac = torch.diag(b1_free_params_3fac); Lambda1_3fac = torch.zeros(O, L1_state1, device=device)
        Lambda1_3fac[0,0]=1; Lambda1_3fac[1,0]=lambda1_free_params_3fac[0]; Lambda1_3fac[2,0]=lambda1_free_params_3fac[1]; Lambda1_3fac[3,1]=1; Lambda1_3fac[4,1]=lambda1_free_params_3fac[2]
        Lambda1_3fac[5,1]=lambda1_free_params_3fac[3]; Lambda1_3fac[6,2]=1; Lambda1_3fac[7,2]=lambda1_free_params_3fac[4]; Lambda1_3fac[8,2]=lambda1_free_params_3fac[5]
        # ▼▼▼【変更点】▼▼▼
        loss = -kalman_filter_torch_loss(Y_train, b0_3fac, B1_3fac, Lambda1_3fac, Q_est_3fac, R_est_3fac, torch.zeros(L1_state1, 1, device=device), torch.eye(L1_state1, device=device) * 1e3)
        # ▲▲▲【変更ここまで】▲▲▲
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
    patience = 100; patience_counter = 0; best_loss_1fac = float('inf'); best_params_1fac = {}
    b0_1fac = torch.randn(L1_state2, 1, device=device).requires_grad_(True)
    B1_1fac = torch.randn(L1_state2, L1_state2, device=device).requires_grad_(True)
    lambda1_free_params_1fac = torch.randn(8, device=device).requires_grad_(True)
    log_q_diag_1fac = torch.zeros(L1_state2, device=device).requires_grad_(True)
    log_r_diag_1fac = torch.zeros(O, device=device).requires_grad_(True)
    params_to_learn_1fac = [b0_1fac, B1_1fac, lambda1_free_params_1fac, log_q_diag_1fac, log_r_diag_1fac]
    optimizer_1fac = torch.optim.AdamW(params_to_learn_1fac, lr=0.001, weight_decay=0.01)
    pbar = trange(20000)
    for epoch in pbar:
        optimizer_1fac.zero_grad()
        Q_est_1fac = torch.diag(torch.exp(log_q_diag_1fac)); R_est_1fac = torch.diag(torch.exp(log_r_diag_1fac))
        Lambda1_1fac = torch.zeros(O, L1_state2, device=device); Lambda1_1fac[0,0]=1.0; Lambda1_1fac[1:9,0] = lambda1_free_params_1fac[0:8]
        # ▼▼▼【変更点】▼▼▼
        loss = -kalman_filter_torch_loss(Y_train, b0_1fac, B1_1fac, Lambda1_1fac, Q_est_1fac, R_est_1fac, torch.zeros(L1_state2, 1, device=device), torch.eye(L1_state2, device=device) * 1e3)
        # ▲▲▲【変更ここまで】▲▲▲
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
print("\n--- 4. Defining and Training BPS Random-Walk Model (on Training data) ---")
print("Pre-calculating inputs for BPS model...")
with torch.no_grad():
    # ▼▼▼【変更点】▼▼▼
    # 訓練データで事前計算
    preds_3fac_est_train, _, _ = get_kalman_predictions_and_latents(Y_train, **best_params_3fac)
    preds_1fac_est_train, _, _ = get_kalman_predictions_and_latents(Y_train, **best_params_1fac)
    factors_f_train = torch.stack((preds_3fac_est_train, preds_1fac_est_train), dim=3)
    # ▲▲▲【変更ここまで】▲▲▲
print("Pre-calculation complete.")

def bps_random_walk_model(y_obs, factors_f):
    N, T, O, J = factors_f.shape # Nt -> T
    log_v_diag_person = pyro.sample("log_v_diag_person", dist.Normal(0.0, 1.0).expand([J]).to_event(1))
    V_person = torch.diag_embed(torch.exp(log_v_diag_person))
    log_tau_person = pyro.sample("log_tau_person", dist.Normal(0.0, 1.0))
    tau_person = torch.exp(log_tau_person)
    log_v_diag_time = pyro.sample("log_v_diag_time", dist.Normal(0.0, 1.0).expand([J]).to_event(1))
    V_time = torch.diag_embed(torch.exp(log_v_diag_time))
    log_tau_time = pyro.sample("log_tau_time", dist.Normal(0.0, 1.0))
    tau_time = torch.exp(log_tau_time)
    log_sigma_diag = pyro.sample("log_sigma_diag", dist.Normal(0.0, 1.0).expand([O]).to_event(1))
    Sigma = torch.diag_embed(torch.exp(log_sigma_diag))
    with pyro.plate("individuals", N):
        initial_loc = torch.tensor([5.0, 0.0], device=device) 
        beta_t0 = pyro.sample("beta_t0", dist.MultivariateNormal(initial_loc, V_person))
        alpha_t0 = pyro.sample("alpha_t0", dist.MultivariateNormal(torch.zeros(O, device=device), torch.eye(O, device=device) * tau_person**2))
        beta_t = beta_t0
        alpha_t = alpha_t0
        for t in range(T): # Nt -> T
            beta_t = pyro.sample(f"beta_{t}", dist.MultivariateNormal(beta_t, V_time))
            alpha_t = pyro.sample(f"alpha_{t}", dist.MultivariateNormal(alpha_t, torch.eye(O, device=device) * tau_time**2))
            factors_t = factors_f[:, t, :, :]
            y_mean = alpha_t + torch.einsum('ioj,ij->io', factors_t, beta_t)
            pyro.sample(f"obs_{t}", dist.MultivariateNormal(y_mean, Sigma), obs=y_obs[:, t, :])

guide = AutoDiagonalNormal(bps_random_walk_model)
optimizer = Adam({"lr": 0.005})
svi = SVI(bps_random_walk_model, guide, optimizer, loss=Trace_ELBO())

if os.path.exists(BPS_GUIDE_FILE):
    print(f"Loading pre-trained BPS model guide from '{BPS_GUIDE_FILE}'...")
    pyro.clear_param_store()
    with safe_globals([constraints._Real, dist.constraints._SoftplusPositive]):
        pyro.get_param_store().load(BPS_GUIDE_FILE, map_location=device)
    loss_bps = -9999
else:
    print("Starting BPS model training with VI...")
    patience = 100; patience_counter = 0; best_loss_bps = float('inf')
    pyro.clear_param_store()
    pbar = trange(20000)
    for step in pbar:
        # ▼▼▼【変更点】▼▼▼
        loss = svi.step(Y_train, factors_f_train) / N
        # ▲▲▲【変更ここまで】▲▲▲
        if torch.isnan(torch.tensor(loss)):
            print("Loss became NaN. Stopping training."); break
        pbar.set_description(f"[BPS SVI step {step+1}] ELBO loss: {loss:.4f} (Best: {best_loss_bps:.4f})")
        if loss < best_loss_bps:
            best_loss_bps = loss
            patience_counter = 0
            pyro.get_param_store().save(BPS_GUIDE_FILE)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"\n   -> Early stopping triggered at step {step + 1}.")
            break
    loss_bps = best_loss_bps
    print(f"Finished training. Loading best model state from '{BPS_GUIDE_FILE}'...")
    pyro.clear_param_store()
    with safe_globals([constraints._Real, dist.constraints._SoftplusPositive]):
        pyro.get_param_store().load(BPS_GUIDE_FILE, map_location=device)

# ----------------------------------------------------------------
# Part 5: RS-KFモデルの定義と学習
# ----------------------------------------------------------------
print("\n\n--- 5. Defining and Training the Regime-Switching Kalman Filter Model (on Training data) ---")

class RegimeSwitchingKF(torch.nn.Module):
    def __init__(self, O, initial_belief='informative', init_params=None):
        super().__init__()
        self.initial_belief = initial_belief
        self.O = O; self.dim_g, self.dim_t, self.dim_b, self.dim_w = 1, 1, 1, 1
        self.L = self.dim_g + self.dim_t + self.dim_b + self.dim_w
        if init_params:
            print("     -> Initializing FRS model with pre-trained baseline parameters.")
            self.B_G = torch.nn.Parameter(init_params['B1_3fac'][0, 0].clone().reshape(self.dim_g, self.dim_g))
            self.B_T = torch.nn.Parameter(init_params['B1_3fac'][1, 1].clone().reshape(self.dim_t, self.dim_t))
            self.B_B = torch.nn.Parameter(init_params['B1_3fac'][2, 2].clone().reshape(self.dim_b, self.dim_b))
            self.B_W = torch.nn.Parameter(init_params['B1_1fac'][0, 0].clone().reshape(self.dim_w, self.dim_w))
            self.lambda_r1_free = torch.nn.Parameter(init_params['lambda1_free_3fac'].clone())
            self.lambda_r2_free = torch.nn.Parameter(init_params['lambda1_free_1fac'].clone())
            q_diag_3fac = torch.diag(init_params['Q_3fac']); q_diag_1fac = torch.diag(init_params['Q_1fac'])
            self.q_diag = torch.nn.Parameter(torch.cat([q_diag_3fac, q_diag_1fac]).clone())
            r_diag_3fac = torch.diag(init_params['R_3fac']); r_diag_1fac = torch.diag(init_params['R_1fac'])
            self.r_diag = torch.nn.Parameter(((r_diag_3fac + r_diag_1fac) / 2.0).clone())
            print("     -> Initializing FRS HMM parameters by SAMPLING AROUND TRUE DGP values.")
            std_dev = 0.1
            self.gamma1 = torch.nn.Parameter(dist.Normal(torch.tensor(gamma_intercept, device=device), std_dev).sample())
            true_gamma2 = torch.tensor([gamma_task, gamma_goal, gamma_bond], device=device)
            self.gamma2_learnable = torch.nn.Parameter(dist.Normal(true_gamma2, std_dev).sample())
        else: # (random initialization, not used in this run but kept for completeness)
            print("     -> Initializing FRS model with random parameters.")
            beta_dist=torch.distributions.beta.Beta(5.0,1.5);self.B_G=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_g,self.dim_g));self.B_T=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_t,self.dim_t));self.B_B=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_b,self.dim_b));self.B_W=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_w,self.dim_w));self.lambda_r1_free=torch.nn.Parameter(torch.randn(6));self.lambda_r2_free=torch.nn.Parameter(torch.randn(O-1));self.q_diag=torch.nn.Parameter(torch.rand(self.L));self.r_diag=torch.nn.Parameter(torch.rand(O));self.gamma1=torch.nn.Parameter(torch.abs(torch.randn(())));self.gamma2_learnable=torch.nn.Parameter(torch.randn(self.L-1))
    def _build_matrices(self):
        L,dg,dt,db,dw=self.L,self.dim_g,self.dim_t,self.dim_b,self.dim_w;B_1_to_1=torch.zeros(L,L,device=device);B_1_to_1[0:dg,0:dg]=self.B_G;B_1_to_1[dg:dg+dt,dg:dg+dt]=self.B_T;B_1_to_1[dg+dt:dg+dt+db,dg+dt:dg+dt+db]=self.B_B;B_2_to_2=torch.zeros(L,L,device=device);B_2_to_2[dg+dt+db:,dg+dt+db:]=self.B_W;B_1_to_2=torch.zeros(L,L,device=device);B_1_to_2[dg+dt+db:,0:dg]=1/3;B_1_to_2[dg+dt+db:,dg:dg+dt]=1/3;B_1_to_2[dg+dt+db:,dg+dt:dg+dt+db]=1/3;B_2_to_1=torch.zeros(L,L,device=device);Lambda_r1=torch.zeros(self.O,L,device=device);Lambda_r1[0,0]=1.0;Lambda_r1[1,0]=self.lambda_r1_free[0];Lambda_r1[2,0]=self.lambda_r1_free[1];Lambda_r1[3,1]=1.0;Lambda_r1[4,1]=self.lambda_r1_free[2];Lambda_r1[5,1]=self.lambda_r1_free[3];Lambda_r1[6,2]=1.0;Lambda_r1[7,2]=self.lambda_r1_free[4];Lambda_r1[8,2]=self.lambda_r1_free[5];Lambda_r2=torch.zeros(self.O,L,device=device);Lambda_r2[0,dg+dt+db]=1.0;Lambda_r2[1:,dg+dt+db]=self.lambda_r2_free;Q=torch.diag(self.q_diag.abs()+1e-4);R=torch.diag(self.r_diag.abs()+1e-4);return(B_1_to_1,B_2_to_1,B_1_to_2,B_2_to_2),(Lambda_r1,Lambda_r2),(Q,R)
    def forward(self, y):
        N,T,O=y.shape;(B11,B21,B12,B22),(L1_lambda,L2_lambda),(Q,R)=self._build_matrices();self.eta_filtered_r1_history=torch.zeros(N,T,self.L,device=device);self.eta_filtered_r2_history=torch.zeros(N,T,self.L,device=device);self.P_filtered_r1_history=torch.zeros(N,T,self.L,self.L,device=device);self.P_filtered_r2_history=torch.zeros(N,T,self.L,self.L,device=device);self.filtered_probs=torch.zeros(N,T,2,device=device);self.predicted_y=torch.zeros(N,T,O,device=device);prob_tm1=torch.zeros(N,2,device=device);eta_marg_r1_tm1=torch.zeros(N,self.L,1,device=device);eta_marg_r2_tm1=torch.zeros(N,self.L,1,device=device);initial_P_diag=torch.tensor([1e3,1e3,1e3,1e-9],device=device);P_0=torch.diag(initial_P_diag).expand(N,-1,-1);P_marg_r1_tm1=P_0;P_marg_r2_tm1=P_0.clone()
        if self.initial_belief=='informative':prob_tm1[:,0]=0.99;prob_tm1[:,1]=0.01
        else:prob_tm1[:,0]=0.5;prob_tm1[:,1]=0.5
        total_log_likelihood=0.0
        for t in range(T):
            y_t=y[:,t,:].unsqueeze(-1);eta_state1_components=eta_marg_r1_tm1[:,:self.L-1,:].squeeze(-1);logit_p11=self.gamma1+(eta_state1_components*self.gamma2_learnable).sum(-1);p11=torch.sigmoid(logit_p11);p22=torch.full((N,),0.9999,device=device);transition_probs=torch.stack([torch.stack([p11,1-p11],dim=1),torch.stack([1-p22,p22],dim=1)],dim=1);eta_pred_11=B11@eta_marg_r1_tm1;P_pred_11=B11@P_marg_r1_tm1@B11.mT+Q;eta_pred_12=B21@eta_marg_r2_tm1;P_pred_12=B21@P_marg_r2_tm1@B21.mT+Q;eta_pred_21=B12@eta_marg_r1_tm1;P_pred_21=B12@P_marg_r1_tm1@B12.mT+Q;eta_pred_22=B22@eta_marg_r2_tm1;P_pred_22=B22@P_marg_r2_tm1@B22.mT+Q;v_11=y_t-L1_lambda@eta_pred_11;F_11=L1_lambda@P_pred_11@L1_lambda.mT+R;v_12=y_t-L1_lambda@eta_pred_12;F_12=L1_lambda@P_pred_12@L1_lambda.mT+R;v_21=y_t-L2_lambda@eta_pred_21;F_21=L2_lambda@P_pred_21@L2_lambda.mT+R;v_22=y_t-L2_lambda@eta_pred_22;F_22=L2_lambda@P_pred_22@L2_lambda.mT+R;f_jitter=torch.eye(O,device=device)*1e-5;F_11+=f_jitter;F_12+=f_jitter;F_21+=f_jitter;F_22+=f_jitter;log_lik_11=MultivariateNormal(loc=torch.zeros_like(v_11.squeeze(-1)),covariance_matrix=F_11).log_prob(v_11.squeeze(-1));log_lik_12=MultivariateNormal(loc=torch.zeros_like(v_12.squeeze(-1)),covariance_matrix=F_12).log_prob(v_12.squeeze(-1));log_lik_21=MultivariateNormal(loc=torch.zeros_like(v_21.squeeze(-1)),covariance_matrix=F_21).log_prob(v_21.squeeze(-1));log_lik_22=MultivariateNormal(loc=torch.zeros_like(v_22.squeeze(-1)),covariance_matrix=F_22).log_prob(v_22.squeeze(-1));log_prob_tm1=torch.log(prob_tm1+1e-9);log_prob_t_11=log_prob_tm1[:,0]+torch.log(transition_probs[:,0,0]+1e-9)+log_lik_11;log_prob_t_12=log_prob_tm1[:,1]+torch.log(transition_probs[:,1,0]+1e-9)+log_lik_12;log_prob_t_21=log_prob_tm1[:,0]+torch.log(transition_probs[:,0,1]+1e-9)+log_lik_21;log_prob_t_22=log_prob_tm1[:,1]+torch.log(transition_probs[:,1,1]+1e-9)+log_lik_22;log_prob_t=torch.stack([log_prob_t_11,log_prob_t_12,log_prob_t_21,log_prob_t_22],dim=1);log_likelihood_t=torch.logsumexp(log_prob_t,dim=1);total_log_likelihood+=log_likelihood_t.sum();prob_t=torch.exp(log_prob_t-log_likelihood_t.unsqueeze(1));prob_t_r1=prob_t[:,0]+prob_t[:,1];prob_t_r2=prob_t[:,2]+prob_t[:,3];W_21=prob_t[:,2]/(prob_t_r2+1e-9);W_22=prob_t[:,3]/(prob_t_r2+1e-9);K_11=P_pred_11@L1_lambda.mT@torch.linalg.pinv(F_11);eta_upd_11=eta_pred_11+K_11@v_11;K_21=P_pred_21@L2_lambda.mT@torch.linalg.pinv(F_21);eta_upd_21=eta_pred_21+K_21@v_21;K_22=P_pred_22@L2_lambda.mT@torch.linalg.pinv(F_22);eta_upd_22=eta_pred_22+K_22@v_22;I_L=torch.eye(self.L,device=device);I_KL_11=I_L-K_11@L1_lambda;P_upd_11=I_KL_11@P_pred_11@I_KL_11.mT+K_11@R@K_11.mT;I_KL_21=I_L-K_21@L2_lambda;P_upd_21=I_KL_21@P_pred_21@I_KL_21.mT+K_21@R@K_21.mT;I_KL_22=I_L-K_22@L2_lambda;P_upd_22=I_KL_22@P_pred_22@I_KL_22.mT+K_22@R@K_22.mT;eta_marg_r1_t=eta_upd_11;P_marg_r1_t=P_upd_11;eta_marg_r2_t=W_21.view(N,1,1)*eta_upd_21+W_22.view(N,1,1)*eta_upd_22;P_marg_r2_t=W_21.view(N,1,1)*(P_upd_21+(eta_marg_r2_t-eta_upd_21)@(eta_marg_r2_t-eta_upd_21).transpose(-1,-2))+W_22.view(N,1,1)*(P_upd_22+(eta_marg_r2_t-eta_upd_22)@(eta_marg_r2_t-eta_upd_22).transpose(-1,-2));eta_marg_r1_tm1,P_marg_r1_tm1=eta_marg_r1_t,P_marg_r1_t;eta_marg_r2_tm1,P_marg_r2_tm1=eta_marg_r2_t,P_marg_r2_t;prob_tm1=torch.stack([prob_t_r1,prob_t_r2],dim=1);self.eta_filtered_r1_history[:,t,:]=eta_marg_r1_t.squeeze(-1);self.eta_filtered_r2_history[:,t,:]=eta_marg_r2_t.squeeze(-1);self.P_filtered_r1_history[:,t,:,:]=P_marg_r1_t;self.P_filtered_r2_history[:,t,:,:]=P_marg_r2_t;self.filtered_probs[:,t,:]=prob_tm1;y_pred_t=prob_tm1[:,0].view(N,1,1)*(L1_lambda@eta_marg_r1_t)+prob_tm1[:,1].view(N,1,1)*(L2_lambda@eta_marg_r2_t);self.predicted_y[:,t,:]=y_pred_t.squeeze(-1)
        return -total_log_likelihood

frs_initial_params = {'B1_3fac':best_params_3fac['B1'],'B1_1fac':best_params_1fac['B1'],'lambda1_free_3fac':torch.cat([best_params_3fac['Lambda1'][1:3,0],best_params_3fac['Lambda1'][4:6,1],best_params_3fac['Lambda1'][7:9,2]]),'lambda1_free_1fac':best_params_1fac['Lambda1'][1:9,0],'Q_3fac':best_params_3fac['Q'],'Q_1fac':best_params_1fac['Q'],'R_3fac':best_params_3fac['R'],'R_1fac':best_params_1fac['R']}

if os.path.exists(RS_KF_MODEL_FILE):
    print(f"Loading pre-computed RS-KF model results from '{RS_KF_MODEL_FILE}'...")
    saved_data = torch.load(RS_KF_MODEL_FILE, weights_only=False)
    best_model_rs_state = saved_data['model_state_dict']
    loss_rs_train = saved_data['loss']
else:
    start_time_rs = time.time(); best_loss_rs = float('inf'); best_model_rs_state = None
    patience = 100; patience_counter = 0; absolute_threshold = 0.1
    print("Starting RS-KF model training...")
    model_rs = RegimeSwitchingKF(O, initial_belief='informative', init_params=frs_initial_params).to(device)
    optimizer_rs = torch.optim.Adam(model_rs.parameters(), lr=1e-3)
    pbar = trange(20000)
    for epoch in pbar:
        optimizer_rs.zero_grad()
        # ▼▼▼【変更点】▼▼▼
        loss = model_rs(Y_train) # Train on training data
        # ▲▲▲【変更ここまで】▲▲▲
        if torch.isnan(loss): print(f"Run failed due to NaN loss."); break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model_rs.parameters(), 1.0); optimizer_rs.step()
        pbar.set_description(f"[RS-KF Epoch {epoch+1}] loss: {loss.item():.4f} (Best: {best_loss_rs:.4f})")
        if best_loss_rs - loss.item() > absolute_threshold:
            best_loss_rs = loss.item(); patience_counter = 0; best_model_rs_state = model_rs.state_dict()
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    -> Early stopping triggered at epoch {epoch + 1}."); break
    loss_rs_train = best_loss_rs
    duration_rs = time.time() - start_time_rs
    print(f"RS-KF model training finished. Duration: {duration_rs:.2f}s")
    print(f"Saving RS-KF model results to '{RS_KF_MODEL_FILE}'...")
    results_to_save = {'model_state_dict': best_model_rs_state, 'loss': loss_rs_train, 'duration': duration_rs}
    torch.save(results_to_save, RS_KF_MODEL_FILE)

# ================================================================
# Part 6 & 7: モデル評価とプロット用データの準備（統合・修正版）
# ================================================================
print("\n--- Part 6 & 7: Model Evaluation and Plotting Data Preparation ---")

# --- 6a. ベースラインモデルの評価 ---
print("\n--- 6a. Evaluating Baseline models on Test Set ---")
with torch.no_grad():
    # 訓練データを用いてテスト期間の予測値を生成
    preds_3fac_forecast, _, _ = generate_forecasts(best_params_3fac, Y_generated, Nt_train)
    preds_1fac_forecast, _, _ = generate_forecasts(best_params_1fac, Y_generated, Nt_train)

# RMSEと、モデルの性質に基づく固定の感度・特異度を計算
rmse_3fac = calculate_rmse(Y_test, preds_3fac_forecast)
rmse_1fac = calculate_rmse(Y_test, preds_1fac_forecast)
sens_3fac, spec_3fac = 0.0, 1.0 # 3因子モデルは常にState 1と予測すると仮定
sens_1fac, spec_1fac = 1.0, 0.0 # 1因子モデルは常にState 2と予測すると仮定

# --- 6b. BPSモデルの評価とデータ準備 ---
print("\n--- 6b. Evaluating BPS Model and Preparing Data ---")
with torch.no_grad():
    # ステップ1: BPSモデルへの入力となる因子（各ベースラインモデルの予測値）を全期間分計算
    preds_3fac_full, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
    preds_1fac_full, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_1fac)
    factors_f_full = torch.stack((preds_3fac_full, preds_1fac_full), dim=3)

    # ステップ2: 訓練期間(t=0..19)について、学習済みガイドから事後サンプルを生成
    guide.requires_grad_(False)
    num_samples = 1000
    predictive_train = Predictive(bps_random_walk_model, guide=guide, num_samples=num_samples,
                                  return_sites=[f"beta_{t}" for t in range(Nt_train)])
    posterior_samples_train = predictive_train(Y_train, factors_f_train)
    beta_samples_train = torch.stack([posterior_samples_train[f'beta_{t}'] for t in range(Nt_train)], dim=2)

    # ステップ3: テスト期間(t=20..24)について、ランダムウォークで将来のbetaをサンプリング（フォワードサンプリング）
    beta_samples_forecast = torch.zeros(num_samples, N, Nt_test, J, device=device)
    beta_prev = beta_samples_train[:, :, -1, :]  # 訓練期間の最後のbetaを取得 (Shape: 1000, 100, 2)

    median_params = guide.median()
    V_time = torch.diag(torch.exp(median_params["log_v_diag_time"])) # 時間遷移の共分散 (Shape: 2, 2)

    for t in range(Nt_test):
        # 【エラー修正箇所】バッチ処理のために形状を一時的に変更
        original_shape = beta_prev.shape
        beta_prev_flat = beta_prev.reshape(-1, J)  # Shape: (1000 * 100, 2)

        # ランダムウォークに従って次のbetaをサンプリング
        beta_next_flat = dist.MultivariateNormal(beta_prev_flat, V_time).sample()

        # 元の形状に戻す
        beta_next = beta_next_flat.reshape(original_shape)

        beta_samples_forecast[:, :, t, :] = beta_next
        beta_prev = beta_next

    # ステップ4: 訓練期間とテスト期間のサンプルを結合し、全期間の事後サンプルを完成
    beta_samples = torch.cat([beta_samples_train, beta_samples_forecast], dim=2)

    # ステップ5: 評価のために、事後サンプルの中央値を取得して点推定値とする
    median_betas_full = torch.median(beta_samples, dim=0).values
    estimated_weights_bps_full = torch.nn.functional.softmax(median_betas_full, dim=-1)
    
    # テストデータでRMSE、感度、特異度を計算
    weights_test = estimated_weights_bps_full[:, Nt_train:, :]
    preds_3fac_test = preds_3fac_full[:, Nt_train:, :]
    preds_1fac_test = preds_1fac_full[:, Nt_train:, :]
    y_pred_mixed_bps_test = weights_test[..., 0].unsqueeze(-1) * preds_3fac_test + \
                            weights_test[..., 1].unsqueeze(-1) * preds_1fac_test
    
    rmse_bps = calculate_rmse(Y_test, y_pred_mixed_bps_test)
    predicted_states_bps = (weights_test[:, :, 1] > 0.5).cpu().numpy().astype(int)
    sens_bps, spec_bps = calculate_sens_spec(actual_states_test, predicted_states_bps)
    print("BPS model evaluation on test set complete.")

    # ステップ6: プロット用に、事後サンプル全体から信頼区間を計算
    prob_samples = torch.nn.functional.softmax(beta_samples, dim=-1)
    prob_samples_s2 = prob_samples[..., 1]
    prob_samples_s2_permuted = prob_samples_s2.permute(1, 2, 0)
    bps_lower_ci = torch.quantile(prob_samples_s2_permuted, 0.025, dim=-1).cpu().numpy()
    bps_upper_ci = torch.quantile(prob_samples_s2_permuted, 0.975, dim=-1).cpu().numpy()
    print("Calculated 95% credible intervals for BPS model.")


# --- 6c. RS-KFモデルの評価とデータ準備 ---
print("\n--- 6c. Evaluating RS-KF Model and Preparing Data ---")
final_model_rs = RegimeSwitchingKF(O, initial_belief='informative', init_params=frs_initial_params).to(device)
if 'best_model_rs_state' in locals() and best_model_rs_state:
    final_model_rs.load_state_dict(best_model_rs_state)
else: # スクリプトを途中から実行した場合など
    saved_data = torch.load(RS_KF_MODEL_FILE, weights_only=False)
    final_model_rs.load_state_dict(saved_data['model_state_dict'])

with torch.no_grad():
    # 評価：テスト期間の予測値を生成してRMSE等を計算
    y_preds_rs_forecast, probs_rs_forecast, _, _, _, _ = forecast_rs(final_model_rs, Y_generated, Nt_train)
    rmse_rs = calculate_rmse(Y_test, y_preds_rs_forecast)
    predicted_states_rs = (probs_rs_forecast[:, :, 1] > 0.5).cpu().numpy().astype(int)
    sens_rs, spec_rs = calculate_sens_spec(actual_states_test, predicted_states_rs)
    print("RS-KF model evaluation on test set complete.")

    # プロット用データ準備：全期間の確率を生成
    _ = final_model_rs(Y_train)
    probs_rs_train = final_model_rs.filtered_probs.cpu().numpy()
    probs_rs_test = probs_rs_forecast.cpu().numpy()
    predicted_probs_rs_full = np.concatenate((probs_rs_train, probs_rs_test), axis=1)
    print("Generated full-period probabilities for RS-KF model.")

# --- 6d. 最終比較テーブル ---
table_width = 110
print("\n" + "="*table_width)
print(f"{'Model':<30} | {'Train Loss':<15} | {'Test RMSE':<15} | {'Test Sensitivity':<15} | {'Test Specificity':<15} |")
print("-"*table_width)
print(f"{'3-Factor Model':<30} | {loss_3fac:<15.4f} | {rmse_3fac:<15.4f} | {sens_3fac:<15.4f} | {spec_3fac:<15.4f} |")
print(f"{'1-Factor Model':<30} | {loss_1fac:<15.4f} | {rmse_1fac:<15.4f} | {sens_1fac:<15.4f} | {spec_1fac:<15.4f} |")
print(f"{'BPS Random Walk':<30} | {loss_bps:<15.4f} | {rmse_bps:<15.4f} | {sens_bps:<15.4f} | {spec_bps:<15.4f} |")
print(f"{'Regime-Switching KF':<30} | {loss_rs_train:<15.4f} | {rmse_rs:<15.4f} | {sens_rs:<15.4f} | {spec_rs:<15.4f} |")
print("Note: Loss for Baseline/RS-KF is NegLogLikelihood. Loss for BPS is ELBO.")
print("      Metrics (RMSE, Sensitivity, Specificity) are evaluated on the test set.")
print("="*table_width)


# --- 7. 潜在変数プロット用のデータ準備 ---
print("\n--- 7. Preparing Latent Variable (eta) Data for Plots ---")
with torch.no_grad():
    # ベースラインモデルの潜在変数
    _, latents_3fac_train, P_3fac_train = get_kalman_predictions_and_latents(Y_train, **best_params_3fac)
    _, latents_3fac_test, P_3fac_test = generate_forecasts(best_params_3fac, Y_generated, Nt_train)
    latents_3fac_full = torch.cat((latents_3fac_train, latents_3fac_test), dim=1).cpu().numpy()
    P_3fac_full = torch.cat((P_3fac_train, P_3fac_test), dim=1).cpu().numpy()

    _, latents_1fac_train, _ = get_kalman_predictions_and_latents(Y_train, **best_params_1fac)
    _, latents_1fac_test, _ = generate_forecasts(best_params_1fac, Y_generated, Nt_train)
    latents_1fac_full = torch.cat((latents_1fac_train, latents_1fac_test), dim=1).cpu().numpy()

    # RS-KFモデルの潜在変数
    _ = final_model_rs(Y_train) # 訓練期間のフィルタリング結果
    eta_r1_train_rs, P_r1_train_rs = final_model_rs.eta_filtered_r1_history, final_model_rs.P_filtered_r1_history
    eta_r2_train_rs, P_r2_train_rs = final_model_rs.eta_filtered_r2_history, final_model_rs.P_filtered_r2_history
    
    # forecast_rsからテスト期間の結果を取得
    _, _, eta_r1_test_rs, eta_r2_test_rs, P_r1_test_rs, P_r2_test_rs = forecast_rs(final_model_rs, Y_generated, Nt_train)
    
    eta_r1_full_rs = torch.cat((eta_r1_train_rs, eta_r1_test_rs), dim=1).cpu().numpy()
    P_r1_full_rs = torch.cat((P_r1_train_rs, P_r1_test_rs), dim=1).cpu().numpy()
    eta_r2_full_rs = torch.cat((eta_r2_train_rs, eta_r2_test_rs), dim=1).cpu().numpy()
    P_r2_full_rs = torch.cat((P_r2_train_rs, P_r2_test_rs), dim=1).cpu().numpy()

eta_true_hist_numpy = eta_true_history.cpu().numpy()
print("All data for plotting is now ready.")
# このブロックはここまでです。次はPart 8の可視化コードが続きます。

# ----------------------------------------------------------------
# Part 8: 可視化
# ----------------------------------------------------------------
print("\n--- 8. Visualization of Results ---")
time_points = np.arange(Nt)

# --- 8a. 個人レベルのレジームスイッチ過程のプロット (新規追加) ---
print("\n--- 8a. Generating individual regime switch plot with C.I. ---")
individual_ids_to_plot = [10, 25, 50, 75]
if N < max(individual_ids_to_plot):
    individual_ids_to_plot = random.sample(range(N), 4)

weight_mean_full = estimated_weights_bps_full.cpu().numpy()
actual_states_binary = (actual_states == 2).astype(int)

fig, axes = plt.subplots(len(individual_ids_to_plot), 2, figsize=(16, 4 * len(individual_ids_to_plot)), sharex=True, sharey=True)
fig.suptitle('Individual Regime Switch Trajectories (Filtered on Train, Forecast Implied on Test)', fontsize=20)

for i, ind_id in enumerate(individual_ids_to_plot):
    # BPS Plot
    ax_bps = axes[i, 0]
    ax_bps.plot(time_points, weight_mean_full[ind_id, :, 1], 'g-', label='BPS Prob. (State 2)', lw=2)
    ax_bps.fill_between(time_points, bps_lower_ci[ind_id, :], bps_upper_ci[ind_id, :], color='green', alpha=0.2, label='95% Credible Interval')
    ax_bps.plot(time_points, actual_states_binary[ind_id, :], 'r--', label='True State', lw=2)
    ax_bps.set_title(f'Individual #{ind_id} - BPS Random Walk Model')
    ax_bps.set_ylabel('Probability of State 2')
    
    # RS-KF Plot
    ax_rs = axes[i, 1]
    ax_rs.plot(time_points, predicted_probs_rs_full[ind_id, :, 1], 'b-', label='RS-KF Prob. (State 2)', lw=2)
    ax_rs.plot(time_points, actual_states_binary[ind_id, :], 'r--', label='True State', lw=2)
    ax_rs.set_title(f'Individual #{ind_id} - RS-KF Model')

    ax_bps.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2)
    ax_rs.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2)
    if i == 0:
        ax_bps.legend(loc='upper left'); ax_rs.legend(loc='upper left')

axes[-1, 0].set_xlabel('Time Point'); axes[-1, 1].set_xlabel('Time Point')
plt.ylim(-0.1, 1.1); plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("individual_switches_plot_with_ci.png")
print("Individual switch plot saved to individual_switches_plot_with_ci.png")
plt.show()


# --- 8b. 潜在変数 eta の比較プロット (新規追加) ---
print("\n--- 8b. Generating latent variable plot with uncertainty ---")
fig, axes = plt.subplots(len(individual_ids_to_plot), 4, figsize=(7 * 4, 4 * len(individual_ids_to_plot)), sharex=True)
fig.suptitle('Latent Variable Trajectories (Filtered on Train, Forecast on Test with 95% CI)', fontsize=20)

for i, ind_id in enumerate(individual_ids_to_plot):
    mask_true_r1 = ~np.isnan(eta_true_hist_numpy[ind_id, :, 1])
    for l in range(L1_state1):
        ax = axes[i, l]
        mean_rs = eta_r1_full_rs[ind_id, :, l]
        std_dev_rs = np.sqrt(P_r1_full_rs[ind_id, :, l, l])
        ax.plot(time_points, mean_rs, 'b-', label='RS-KF', lw=1.5)
        ax.fill_between(time_points, mean_rs - 1.96 * std_dev_rs, mean_rs + 1.96 * std_dev_rs, color='blue', alpha=0.15)
        ax.plot(time_points, latents_3fac_full[ind_id, :, l], 'g-', label='Baseline (3-Fac)', lw=1.5)
        if np.any(mask_true_r1):
            ax.plot(time_points[mask_true_r1], eta_true_hist_numpy[ind_id, :, l][mask_true_r1], 'r--', label='True', lw=2)
        ax.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=1.5)
        ax.set_title(f'Individual #{ind_id} - $\eta_{l+1}$ (3-Factor)')
        if i == 0: ax.legend()
        if l == 0: ax.set_ylabel('Value')

    ax = axes[i, 3]
    mean_rs_w = eta_r2_full_rs[ind_id, :, 3]
    std_dev_rs_w = np.sqrt(P_r2_full_rs[ind_id, :, 3, 3])
    ax.plot(time_points, mean_rs_w, 'b-', label='RS-KF', lw=1.5)
    ax.fill_between(time_points, mean_rs_w - 1.96 * std_dev_rs_w, mean_rs_w + 1.96 * std_dev_rs_w, color='blue', alpha=0.15)
    ax.plot(time_points, latents_1fac_full[ind_id, :, 0], 'g-', label='Baseline (1-Fac)', lw=1.5)
    mask_true_r2 = ~mask_true_r1
    if np.any(mask_true_r2):
        ax.plot(time_points[mask_true_r2], eta_true_hist_numpy[ind_id, :, 0][mask_true_r2], 'r--', label='True', lw=2)
    ax.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=1.5)
    ax.set_title(f'Individual #{ind_id} - $\eta$ (1-Factor)')
    if i == 0: ax.legend()

for l in range(4):
    axes[-1, l].set_xlabel('Time Point')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig("eta_trajectories_forecast_with_ci.png")
print("Latent variable forecast plot saved to eta_trajectories_forecast_with_ci.png")
plt.show()

# --- 8c. 集計レベルでのモデル性能比較プロット (元のスクリプトから維持) ---
print("\n--- 8c. Generating Aggregate Model Comparison Plot ---")
state1_proportion_actual = (actual_states == 1).mean(axis=0)
avg_weights_bps_over_time = estimated_weights_bps_full.mean(dim=0).cpu().numpy()
avg_probs_rs_over_time = predicted_probs_rs_full.mean(axis=0)

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
fig.suptitle('Model Comparison: BPS Random Walk vs. RS-KF (Aggregate Performance)', fontsize=20)
# BPS Model Plot
ax1.plot(time_points, avg_weights_bps_over_time[:, 0], 'o-', color='royalblue', label='Avg. Weight for 3-Factor Model (State 1)', zorder=3)
ax1.plot(time_points, avg_weights_bps_over_time[:, 1], 's-', color='firebrick', label='Avg. Weight for 1-Factor Model (State 2)', zorder=3)
ax1.set_title('BPS Random Walk Model', fontsize=16); ax1.set_xlabel('Time Point', fontsize=12); ax1.set_ylabel('Estimated Model Weight', fontsize=12)
ax1.legend(loc='upper left'); ax1.set_ylim(-0.05, 1.05)
ax1.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2, label='Train-Test Split')
ax2 = ax1.twinx()
ax2.bar(time_points, state1_proportion_actual, color='grey', alpha=0.2, label='Proportion in State 1 (Actual)')
ax2.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12); ax2.set_ylim(-0.05, 1.05); ax2.grid(False)
# RS-KF Model Plot
ax3.plot(time_points, avg_probs_rs_over_time[:, 0], 'o-', color='royalblue', label='Avg. Prob. of Regime 1 (3-Factor)', zorder=3)
ax3.plot(time_points, avg_probs_rs_over_time[:, 1], 's-', color='firebrick', label='Avg. Prob. of Regime 2 (1-Factor)', zorder=3)
ax3.set_title('Regime-Switching KF Model', fontsize=16); ax3.set_xlabel('Time Point', fontsize=12)
ax3.legend(loc='upper left')
ax3.axvline(x=Nt_train - 0.5, color='black', linestyle='--', linewidth=2)
ax4 = ax3.twinx()
ax4.bar(time_points, state1_proportion_actual, color='grey', alpha=0.2, label='Proportion in State 1 (Actual)')
ax4.set_ylabel('Proportion in State 1 (Ground Truth)', fontsize=12); ax4.set_ylim(-0.05, 1.05); ax4.grid(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("model_comparison_plot_forecast.png")
print("Aggregate model comparison plot saved to model_comparison_plot_forecast.png")
plt.show()

# --- Restore stdout ---
sys.stdout = original_stdout
print(f"\nAnalysis complete. Log saved to '{log_filename}'")