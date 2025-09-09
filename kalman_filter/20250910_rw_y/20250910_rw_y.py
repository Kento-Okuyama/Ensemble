# FGME準備：Early Stopping機能付き 修正版スクリプト（完全版）
# + 0904スクリプトの個別プロット機能（信頼区間付き）を統合
# + 各モデルの計算時間を計測・表示する機能を追加
# +【ご依頼による修正】データ生成ロジックの修正 と BPSモデルへの時間重み付け学習の導入

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
from pyro.infer import SVI, Trace_ELBO, Predictive
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
log_filename = f"model_comparison_modified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
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
# ▼▼▼【修正点1-1】潜在変数の合計次元数を定義 ▼▼▼
L_total = L1_state1 + L1_state2 # 3 + 1 = 4

# DGP parameters (defined globally to be accessible by all parts)
gamma_intercept=-2.5; gamma_task=0.1; gamma_goal=0.1; gamma_bond=0.1

# Cache file names
DATA_FILE = 'simulation_data_rw.pt'
MODEL_3FAC_FILE = 'trained_3fac_model.pt'
MODEL_1FAC_FILE = 'trained_1fac_model.pt'
BPS_GUIDE_FILE = 'trained_bps_guide_params_weighted.pt' # 重み付き学習のためキャッシュファイル名を変更
RS_KF_MODEL_FILE = 'rs_kf_results.pt'

# 時間計測用の変数を初期化
duration_3fac, duration_1fac, duration_bps, duration_rs = 0.0, 0.0, 0.0, 0.0

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
    print("No pre-computed data file found. Generating new simulation data...")
    b0_true_state1=torch.tensor([0.0,0.0,0.0],device=device).unsqueeze(1);B1_true_state1=torch.diag(torch.tensor([0.7,0.7,0.7],device=device));lambda1_true_values_state1=[1.2,0.8,1.1,0.9,1.3,0.7];Lambda1_true_state1=torch.zeros(O,L1_state1,device=device);Lambda1_true_state1[0,0]=1;Lambda1_true_state1[1,0]=lambda1_true_values_state1[0];Lambda1_true_state1[2,0]=lambda1_true_values_state1[1];Lambda1_true_state1[3,1]=1;Lambda1_true_state1[4,1]=lambda1_true_values_state1[2];Lambda1_true_state1[5,1]=lambda1_true_values_state1[3];Lambda1_true_state1[6,2]=1;Lambda1_true_state1[7,2]=lambda1_true_values_state1[4];Lambda1_true_state1[8,2]=lambda1_true_values_state1[5];b0_true_state2=torch.tensor([0.0],device=device).unsqueeze(1);B1_true_state2=torch.tensor([[0.9]],device=device);lambda1_true_values_state2=[1.0]*8;Lambda1_true_state2=torch.zeros(O,L1_state2,device=device);Lambda1_true_state2[0,0]=1.0;Lambda1_true_state2[1:9,0]=torch.tensor(lambda1_true_values_state2,device=device);Q_state1=torch.eye(L1_state1,device=device);Q_state2=torch.eye(L1_state2,device=device);R_true=torch.eye(O,device=device);Y_generated=torch.zeros(N,Nt,O,device=device);actual_states=np.zeros((N,Nt));
    # ▼▼▼【修正点1-2】eta_true_historyを正しい4次元で初期化 ▼▼▼
    eta_true_history=torch.full((N,Nt,L_total),float('nan'),device=device)
    q_dist_s1=MultivariateNormal(torch.zeros(L1_state1,device=device),Q_state1);q_dist_s2=MultivariateNormal(torch.zeros(L1_state2,device=device),Q_state2);r_dist=MultivariateNormal(torch.zeros(O,device=device),R_true)
    for i in trange(N,desc="Generating data for each person"):
        eta_history_i=torch.randn(L1_state1,1,device=device);current_state=1;has_switched=False
        for t in range(Nt):
            actual_states[i,t]=current_state
            if current_state==1 and t>0:
                z=gamma_intercept+(eta_history_i[0]*gamma_task+eta_history_i[1]*gamma_goal+eta_history_i[2]*gamma_bond)
                if random.random()<(1/(1+math.exp(-z))):current_state=2
            if current_state==1:eta_t=(B1_true_state1@eta_history_i)+q_dist_s1.sample().reshape(L1_state1,1);y_mean_t=Lambda1_true_state1@eta_t
            else:
                if not has_switched:eta_history_i=torch.tensor([eta_history_i.mean()],device=device).reshape(L1_state2,1);has_switched=True
                eta_t=(B1_true_state2@eta_history_i)+q_dist_s2.sample().reshape(L1_state2,1);y_mean_t=Lambda1_true_state2@eta_t
            Y_generated[i,t,:]=(y_mean_t+r_dist.sample().reshape(O,1)).squeeze()
            
            # ▼▼▼【修正点1-3】状態に応じて、etaの真値を別々の列に保存するロジックに変更 ▼▼▼
            if current_state == 1:
                # 3因子状態の場合、0, 1, 2列目に保存
                eta_true_history[i, t, 0:3] = eta_t.squeeze()
            else:
                # 1因子状態の場合、3列目に保存
                eta_true_history[i, t, 3] = eta_t.squeeze()
            
            eta_history_i=eta_t
    print("Simulation data generated.")
    print(f"Saving simulation data to '{DATA_FILE}'...")
    torch.save({'Y_generated':Y_generated.cpu(),'actual_states':actual_states,'eta_true_history':eta_true_history.cpu()},DATA_FILE)
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
            total_log_likelihood += dist.log_prob(torch.zeros_like(v_t.squeeze(-1))).sum()
        except torch.linalg.LinAlgError: return torch.tensor(float('nan'))
        K_t = torch.bmm(torch.bmm(P_pred, Lambda1.transpose(0, 1).expand(N, -1, -1)), torch.linalg.pinv(F_t))
        eta_updated = eta_pred + torch.bmm(K_t, v_t)
        I_KL = torch.eye(L1, device=device).expand(N, -1, -1) - torch.bmm(K_t, Lambda1.expand(N, -1, -1))
        P_updated = torch.bmm(torch.bmm(I_KL, P_pred), I_KL.transpose(-1, -2)) + torch.bmm(torch.bmm(K_t, R.expand(N, -1, -1)), K_t.transpose(-1, -2))
        eta_prev = eta_updated; P_prev = P_updated
    return total_log_likelihood

def get_kalman_predictions_and_latents(Y, b0, B1, Lambda1, Q, R):
    N, Nt, O1 = Y.shape; L1 = B1.shape[0]
    y_pred_series = torch.zeros_like(Y)
    eta_series = torch.zeros(N, Nt, L1, device=device)
    P_series = torch.zeros(N, Nt, L1, L1, device=device)
    eta_prev = torch.zeros(N, L1, 1, device=device)
    P_prev = torch.eye(L1, device=device).expand(N, -1, -1) * 1e3
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
        eta_series[:, t, :] = eta_updated.squeeze(-1)
        P_series[:, t, :, :] = P_updated
        eta_prev, P_prev = eta_updated, P_updated
    return y_pred_series, eta_series, P_series

def calculate_rmse(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred)**2)).item()

def calculate_sens_spec(actual_states, predicted_states_binary):
    ground_truth_binary=(actual_states==2).astype(int);TP=np.sum((predicted_states_binary==1)&(ground_truth_binary==1));FN=np.sum((predicted_states_binary==0)&(ground_truth_binary==1));TN=np.sum((predicted_states_binary==0)&(ground_truth_binary==0));FP=np.sum((predicted_states_binary==1)&(ground_truth_binary==0));sensitivity=TP/(TP+FN)if(TP+FN)>0 else 0;specificity=TN/(TN+FP)if(TN+FP)>0 else 0;return sensitivity,specificity

# ----------------------------------------------------------------
# Part 3: ベースラインモデルの推定
# ----------------------------------------------------------------
print("\n--- 3. Estimation of Baseline Models ---")

# --- 3a. 3-Factor Model ---
print("\n--- 3a. 3-Factor Model ---")
if os.path.exists(MODEL_3FAC_FILE):
    print(f"Loading pre-trained 3-Factor model from '{MODEL_3FAC_FILE}'...")
    saved_model_3fac = torch.load(MODEL_3FAC_FILE, map_location=device);best_params_3fac = saved_model_3fac['params'];loss_3fac = saved_model_3fac['loss']
else:
    print("Training 3-Factor model...")
    start_time_3fac = time.time()
    patience=100;patience_counter=0;best_loss_3fac=float('inf');best_params_3fac={};b0_3fac=(torch.tensor([0.,0.,0.],device=device).unsqueeze(1)+torch.randn(3,1,device=device)*0.1).requires_grad_(True);b1_free_params_3fac=(torch.tensor([0.7,0.7,0.7],device=device)+torch.randn(3,device=device)*0.1).requires_grad_(True);lambda1_free_params_3fac=(torch.tensor([1.2,0.8,1.1,0.9,1.3,0.7],device=device)+torch.randn(6,device=device)*0.1).requires_grad_(True);log_q_diag_3fac=torch.log(torch.ones(L1_state1,device=device)*0.5).requires_grad_(True);log_r_diag_3fac=torch.log(torch.ones(O,device=device)*0.5).requires_grad_(True);params_to_learn_3fac=[b0_3fac,b1_free_params_3fac,lambda1_free_params_3fac,log_q_diag_3fac,log_r_diag_3fac];optimizer_3fac=torch.optim.AdamW(params_to_learn_3fac,lr=0.001,weight_decay=0.01);pbar = tqdm(range(20000), file=original_stdout)

    for epoch in pbar:
        optimizer_3fac.zero_grad();Q_est_3fac=torch.diag(torch.exp(log_q_diag_3fac));R_est_3fac=torch.diag(torch.exp(log_r_diag_3fac));B1_3fac=torch.diag(b1_free_params_3fac);Lambda1_3fac=torch.zeros(O,L1_state1,device=device);Lambda1_3fac[0,0]=1;Lambda1_3fac[1,0]=lambda1_free_params_3fac[0];Lambda1_3fac[2,0]=lambda1_free_params_3fac[1];Lambda1_3fac[3,1]=1;Lambda1_3fac[4,1]=lambda1_free_params_3fac[2];Lambda1_3fac[5,1]=lambda1_free_params_3fac[3];Lambda1_3fac[6,2]=1;Lambda1_3fac[7,2]=lambda1_free_params_3fac[4];Lambda1_3fac[8,2]=lambda1_free_params_3fac[5];loss=-kalman_filter_torch_loss(Y_generated,b0_3fac,B1_3fac,Lambda1_3fac,Q_est_3fac,R_est_3fac,torch.zeros(L1_state1,1,device=device),torch.eye(L1_state1,device=device)*1e3)
        if torch.isnan(loss):break
        loss.backward();optimizer_3fac.step();pbar.set_description(f"[3-Fac Epoch {epoch+1}] loss: {loss.item():.4f} (Best: {best_loss_3fac:.4f})")
        if loss.item()<best_loss_3fac:best_loss_3fac=loss.item();best_params_3fac={'b0':b0_3fac.detach(),'B1':B1_3fac.detach(),'Lambda1':Lambda1_3fac.detach(),'Q':Q_est_3fac.detach(),'R':R_est_3fac.detach()};patience_counter=0
        else:patience_counter+=1
        if patience_counter>=patience:print(f"\n   -> Early stopping triggered at epoch {epoch+1}.");break
    duration_3fac = time.time() - start_time_3fac
    print(f"3-Factor model training finished. Duration: {duration_3fac:.2f}s")
    loss_3fac=best_loss_3fac;print(f"Finished training. Saving best model to '{MODEL_3FAC_FILE}'...");torch.save({'params':best_params_3fac,'loss':loss_3fac},MODEL_3FAC_FILE)

# --- 3b. 1-Factor Model ---
print("\n--- 3b. 1-Factor Model ---")
if os.path.exists(MODEL_1FAC_FILE):
    print(f"Loading pre-trained 1-Factor model from '{MODEL_1FAC_FILE}'...");saved_model_1fac=torch.load(MODEL_1FAC_FILE,map_location=device);best_params_1fac=saved_model_1fac['params'];loss_1fac=saved_model_1fac['loss']
else:
    print("Training 1-Factor model...")
    start_time_1fac = time.time()
    patience=100;patience_counter=0;best_loss_1fac=float('inf');best_params_1fac={};b0_1fac=(torch.tensor([0.0],device=device).unsqueeze(1)+torch.randn(1,1,device=device)*0.1).requires_grad_(True);B1_1fac=(torch.tensor([[0.9]],device=device)+torch.randn(1,1,device=device)*0.1).requires_grad_(True);lambda1_free_params_1fac=(torch.tensor([1.0]*8,device=device)+torch.randn(8,device=device)*0.1).requires_grad_(True);log_q_diag_1fac=torch.log(torch.ones(L1_state2,device=device)*0.5).requires_grad_(True);log_r_diag_1fac=torch.log(torch.ones(O,device=device)*0.5).requires_grad_(True);params_to_learn_1fac=[b0_1fac,B1_1fac,lambda1_free_params_1fac,log_q_diag_1fac,log_r_diag_1fac];optimizer_1fac=torch.optim.AdamW(params_to_learn_1fac,lr=0.001,weight_decay=0.01);pbar = tqdm(range(20000), file=original_stdout)

    for epoch in pbar:
        optimizer_1fac.zero_grad();Q_est_1fac=torch.diag(torch.exp(log_q_diag_1fac));R_est_1fac=torch.diag(torch.exp(log_r_diag_1fac));Lambda1_1fac=torch.zeros(O,L1_state2,device=device);Lambda1_1fac[0,0]=1.0;Lambda1_1fac[1:9,0]=lambda1_free_params_1fac[0:8];loss=-kalman_filter_torch_loss(Y_generated,b0_1fac,B1_1fac,Lambda1_1fac,Q_est_1fac,R_est_1fac,torch.zeros(L1_state2,1,device=device),torch.eye(L1_state2,device=device)*1e3)
        if torch.isnan(loss):break
        loss.backward();optimizer_1fac.step();pbar.set_description(f"[1-Fac Epoch {epoch+1}] loss: {loss.item():.4f} (Best: {best_loss_1fac:.4f})")
        if loss.item()<best_loss_1fac:best_loss_1fac=loss.item();best_params_1fac={'b0':b0_1fac.detach(),'B1':B1_1fac.detach(),'Lambda1':Lambda1_1fac.detach(),'Q':Q_est_1fac.detach(),'R':R_est_1fac.detach()};patience_counter=0
        else:patience_counter+=1
        if patience_counter>=patience:print(f"\n   -> Early stopping triggered at epoch {epoch+1}.");break
    duration_1fac = time.time() - start_time_1fac
    print(f"1-Factor model training finished. Duration: {duration_1fac:.2f}s")
    loss_1fac=best_loss_1fac;print(f"Finished training. Saving best model to '{MODEL_1FAC_FILE}'...");torch.save({'params':best_params_1fac,'loss':loss_1fac},MODEL_1FAC_FILE)

# ----------------------------------------------------------------
# Part 4: BPSモデルの定義と学習
# ----------------------------------------------------------------
print("\n--- 4. Defining and Training BPS Random-Walk Model ---")
print("Pre-calculating inputs for BPS model...")
with torch.no_grad():
    preds_3fac_est, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
    preds_1fac_est, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_1fac)
    factors_f = torch.stack((preds_3fac_est, preds_1fac_est), dim=3)
print("Pre-calculation complete.")

# ▼▼▼【修正点2-1】時間による重み付けのための重みベクトルを定義 ▼▼▼
T = Nt  # 全期間の長さ
gamma = 0.05        # 過去のデータが持つ最小限の影響力を決めるパラメータ
time_points = torch.arange(T, device=device, dtype=torch.float32)
# 論文 P.1051 の式に基づいて重みを計算
pi_t = 1 + gamma - (1 - time_points / (T - 1))**2
# 正規化
pi_t = (pi_t / pi_t.sum()) * T


# ▼▼▼【修正点2-2】BPSモデルの定義を修正し、time_weightsを引数に取るように変更 ▼▼▼
def bps_random_walk_model(y_obs, factors_f, time_weights):
    N, Nt, O, J = factors_f.shape
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
        for t in range(Nt):
            beta_t = pyro.sample(f"beta_{t}", dist.MultivariateNormal(beta_t, V_time))
            alpha_t = pyro.sample(f"alpha_{t}", dist.MultivariateNormal(alpha_t, torch.eye(O, device=device) * tau_time**2))
            factors_t = factors_f[:, t, :, :]
            y_mean = alpha_t + torch.einsum('ioj,ij->io', factors_t, beta_t)
            # ▼▼▼【修正点2-3】観測データの尤度を計算し、重み付けしてモデルに追加 ▼▼▼
            log_prob_t = dist.MultivariateNormal(y_mean, Sigma).log_prob(y_obs[:, t, :])
            pyro.factor(f"obs_factor_{t}", time_weights[t] * log_prob_t)

guide = AutoDiagonalNormal(bps_random_walk_model);optimizer = Adam({"lr": 0.005});svi = SVI(bps_random_walk_model, guide, optimizer, loss=Trace_ELBO())

if os.path.exists(BPS_GUIDE_FILE):
    print(f"Loading pre-trained BPS model guide from '{BPS_GUIDE_FILE}'...");pyro.clear_param_store()
    with safe_globals([constraints._Real, dist.constraints._SoftplusPositive]):pyro.get_param_store().load(BPS_GUIDE_FILE, map_location=device)
    loss_bps = -9999
else:
    print("Starting BPS model training with VI...")
    start_time_bps = time.time()
    patience=100;patience_counter=0;best_loss_bps=float('inf');pyro.clear_param_store();pbar = tqdm(range(20000), file=original_stdout)

    for step in pbar:
        # ▼▼▼【修正点2-4】SVIのステップ関数に time_weights を渡す ▼▼▼
        loss = svi.step(Y_generated, factors_f, time_weights=pi_t) / N
        if torch.isnan(torch.tensor(loss)):print("Loss became NaN. Stopping training.");break
        pbar.set_description(f"[BPS SVI step {step+1}] ELBO loss: {loss:.4f} (Best: {best_loss_bps:.4f})")
        if loss < best_loss_bps:best_loss_bps=loss;patience_counter=0;pyro.get_param_store().save(BPS_GUIDE_FILE)
        else:patience_counter+=1
        if patience_counter>=patience:print(f"\n   -> Early stopping triggered at step {step+1}.");break
    duration_bps = time.time() - start_time_bps
    print(f"BPS model training finished. Duration: {duration_bps:.2f}s")
    loss_bps=best_loss_bps;print(f"Finished training. Loading best model state from '{BPS_GUIDE_FILE}'...");pyro.clear_param_store()
    with safe_globals([constraints._Real, dist.constraints._SoftplusPositive]):pyro.get_param_store().load(BPS_GUIDE_FILE, map_location=device)

# ----------------------------------------------------------------
# Part 5: RS-KFモデルの定義と学習
# ----------------------------------------------------------------
print("\n\n--- 5. Defining and Training the Regime-Switching Kalman Filter Model ---")

class RegimeSwitchingKF(torch.nn.Module):
    def __init__(self,O,initial_belief='informative',init_params=None):
        super().__init__();self.initial_belief=initial_belief;self.O=O;self.dim_g,self.dim_t,self.dim_b,self.dim_w=1,1,1,1;self.L=self.dim_g+self.dim_t+self.dim_b+self.dim_w
        if init_params:
            print("     -> Initializing FRS model with pre-trained baseline parameters.");self.B_G=torch.nn.Parameter(init_params['B1_3fac'][0,0].clone().reshape(self.dim_g,self.dim_g));self.B_T=torch.nn.Parameter(init_params['B1_3fac'][1,1].clone().reshape(self.dim_t,self.dim_t));self.B_B=torch.nn.Parameter(init_params['B1_3fac'][2,2].clone().reshape(self.dim_b,self.dim_b));self.B_W=torch.nn.Parameter(init_params['B1_1fac'][0,0].clone().reshape(self.dim_w,self.dim_w));self.lambda_r1_free=torch.nn.Parameter(init_params['lambda1_free_3fac'].clone());self.lambda_r2_free=torch.nn.Parameter(init_params['lambda1_free_1fac'].clone());q_diag_3fac=torch.diag(init_params['Q_3fac']);q_diag_1fac=torch.diag(init_params['Q_1fac']);initial_q_diag=torch.cat([q_diag_3fac,q_diag_1fac]);self.q_diag=torch.nn.Parameter(initial_q_diag.clone());r_diag_3fac=torch.diag(init_params['R_3fac']);r_diag_1fac=torch.diag(init_params['R_1fac']);initial_r_diag=(r_diag_3fac+r_diag_1fac)/2.0;self.r_diag=torch.nn.Parameter(initial_r_diag.clone());print("     -> Initializing FRS HMM parameters by SAMPLING AROUND TRUE DGP values.");std_dev=0.1;self.gamma1=torch.nn.Parameter(dist.Normal(torch.tensor(gamma_intercept,device=device),std_dev).sample());true_gamma2=torch.tensor([gamma_task,gamma_goal,gamma_bond],device=device);self.gamma2_learnable=torch.nn.Parameter(dist.Normal(true_gamma2,std_dev).sample())
        else:print("     -> Initializing FRS model with random parameters.");beta_dist=torch.distributions.beta.Beta(5.0,1.5);self.B_G=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_g,self.dim_g));self.B_T=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_t,self.dim_t));self.B_B=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_b,self.dim_b));self.B_W=torch.nn.Parameter(beta_dist.sample().reshape(self.dim_w,self.dim_w));self.lambda_r1_free=torch.nn.Parameter(torch.randn(6));self.lambda_r2_free=torch.nn.Parameter(torch.randn(O-1));self.q_diag=torch.nn.Parameter(torch.rand(self.L));self.r_diag=torch.nn.Parameter(torch.rand(O));self.gamma1=torch.nn.Parameter(torch.abs(torch.randn(())));self.gamma2_learnable=torch.nn.Parameter(torch.randn(self.L-1))
        self.eta_filtered_r1_history=None;self.eta_filtered_r2_history=None;self.P_filtered_r1_history=None;self.P_filtered_r2_history=None;self.filtered_probs=None;self.predicted_y=None
    def _build_matrices(self):
        L,dg,dt,db,dw=self.L,self.dim_g,self.dim_t,self.dim_b,self.dim_w;B_1_to_1=torch.zeros(L,L,device=device);B_1_to_1[0:dg,0:dg]=self.B_G;B_1_to_1[dg:dg+dt,dg:dg+dt]=self.B_T;B_1_to_1[dg+dt:dg+dt+db,dg+dt:dg+dt+db]=self.B_B;B_2_to_2=torch.zeros(L,L,device=device);B_2_to_2[dg+dt+db:,dg+dt+db:]=self.B_W;B_1_to_2=torch.zeros(L,L,device=device);B_1_to_2[dg+dt+db:,0:dg]=1/3;B_1_to_2[dg+dt+db:,dg:dg+dt]=1/3;B_1_to_2[dg+dt+db:,dg+dt:dg+dt+db]=1/3;B_2_to_1=torch.zeros(L,L,device=device);Lambda_r1=torch.zeros(self.O,L,device=device);Lambda_r1[0,0]=1.0;Lambda_r1[1,0]=self.lambda_r1_free[0];Lambda_r1[2,0]=self.lambda_r1_free[1];Lambda_r1[3,1]=1.0;Lambda_r1[4,1]=self.lambda_r1_free[2];Lambda_r1[5,1]=self.lambda_r1_free[3];Lambda_r1[6,2]=1.0;Lambda_r1[7,2]=self.lambda_r1_free[4];Lambda_r1[8,2]=self.lambda_r1_free[5];Lambda_r2=torch.zeros(self.O,L,device=device);Lambda_r2[0,dg+dt+db]=1.0;Lambda_r2[1:,dg+dt+db]=self.lambda_r2_free;Q=torch.diag(self.q_diag.abs()+1e-4);R=torch.diag(self.r_diag.abs()+1e-4);return(B_1_to_1,B_2_to_1,B_1_to_2,B_2_to_2),(Lambda_r1,Lambda_r2),(Q,R)
    def forward(self,y):
        N,Nt,O=y.shape;(B11,B21,B12,B22),(L1_lambda,L2_lambda),(Q,R)=self._build_matrices();self.eta_filtered_r1_history=torch.zeros(N,Nt,self.L,device=device);self.eta_filtered_r2_history=torch.zeros(N,Nt,self.L,device=device);self.P_filtered_r1_history=torch.zeros(N,Nt,self.L,self.L,device=device);self.P_filtered_r2_history=torch.zeros(N,Nt,self.L,self.L,device=device);self.filtered_probs=torch.zeros(N,Nt,2,device=device);self.predicted_y=torch.zeros(N,Nt,O,device=device);prob_tm1=torch.zeros(N,2,device=device);eta_marg_r1_tm1=torch.zeros(N,self.L,1,device=device);eta_marg_r2_tm1=torch.zeros(N,self.L,1,device=device);initial_P_diag=torch.tensor([1e3,1e3,1e3,1e-9],device=device);P_0=torch.diag(initial_P_diag).expand(N,-1,-1);P_marg_r1_tm1=P_0;P_marg_r2_tm1=P_0.clone()
        if self.initial_belief=='informative':prob_tm1[:,0]=0.99;prob_tm1[:,1]=0.01
        else:prob_tm1[:,0]=0.5;prob_tm1[:,1]=0.5
        total_log_likelihood=0.0
        for t in range(Nt):
            y_t=y[:,t,:].unsqueeze(-1);eta_state1_components=eta_marg_r1_tm1[:,:self.L-1,:].squeeze(-1);logit_p11=self.gamma1+(eta_state1_components*self.gamma2_learnable).sum(-1);p11=torch.sigmoid(logit_p11);p22=torch.full((N,),0.9999,device=device);transition_probs=torch.stack([torch.stack([p11,1-p11],dim=1),torch.stack([1-p22,p22],dim=1)],dim=1);eta_pred_11=B11@eta_marg_r1_tm1;P_pred_11=B11@P_marg_r1_tm1@B11.mT+Q;eta_pred_12=B21@eta_marg_r2_tm1;P_pred_12=B21@P_marg_r2_tm1@B21.mT+Q;eta_pred_21=B12@eta_marg_r1_tm1;P_pred_21=B12@P_marg_r1_tm1@B12.mT+Q;eta_pred_22=B22@eta_marg_r2_tm1;P_pred_22=B22@P_marg_r2_tm1@B22.mT+Q;v_11=y_t-L1_lambda@eta_pred_11;F_11=L1_lambda@P_pred_11@L1_lambda.mT+R;v_12=y_t-L1_lambda@eta_pred_12;F_12=L1_lambda@P_pred_12@L1_lambda.mT+R;v_21=y_t-L2_lambda@eta_pred_21;F_21=L2_lambda@P_pred_21@L2_lambda.mT+R;v_22=y_t-L2_lambda@eta_pred_22;F_22=L2_lambda@P_pred_22@L2_lambda.mT+R;f_jitter=torch.eye(O,device=device)*1e-5;F_11+=f_jitter;F_12+=f_jitter;F_21+=f_jitter;F_22+=f_jitter;log_lik_11=MultivariateNormal(loc=torch.zeros_like(v_11.squeeze(-1)),covariance_matrix=F_11).log_prob(v_11.squeeze(-1));log_lik_12=MultivariateNormal(loc=torch.zeros_like(v_12.squeeze(-1)),covariance_matrix=F_12).log_prob(v_12.squeeze(-1));log_lik_21=MultivariateNormal(loc=torch.zeros_like(v_21.squeeze(-1)),covariance_matrix=F_21).log_prob(v_21.squeeze(-1));log_lik_22=MultivariateNormal(loc=torch.zeros_like(v_22.squeeze(-1)),covariance_matrix=F_22).log_prob(v_22.squeeze(-1));log_prob_tm1=torch.log(prob_tm1+1e-9);log_prob_t_11=log_prob_tm1[:,0]+torch.log(transition_probs[:,0,0]+1e-9)+log_lik_11;log_prob_t_12=log_prob_tm1[:,1]+torch.log(transition_probs[:,1,0]+1e-9)+log_lik_12;log_prob_t_21=log_prob_tm1[:,0]+torch.log(transition_probs[:,0,1]+1e-9)+log_lik_21;log_prob_t_22=log_prob_tm1[:,1]+torch.log(transition_probs[:,1,1]+1e-9)+log_lik_22;log_prob_t=torch.stack([log_prob_t_11,log_prob_t_12,log_prob_t_21,log_prob_t_22],dim=1);log_likelihood_t=torch.logsumexp(log_prob_t,dim=1);total_log_likelihood+=log_likelihood_t.sum();prob_t=torch.exp(log_prob_t-log_likelihood_t.unsqueeze(1));prob_t_r1=prob_t[:,0]+prob_t[:,1];prob_t_r2=prob_t[:,2]+prob_t[:,3];W_21=prob_t[:,2]/(prob_t_r2+1e-9);W_22=prob_t[:,3]/(prob_t_r2+1e-9);K_11=P_pred_11@L1_lambda.mT@torch.linalg.pinv(F_11);eta_upd_11=eta_pred_11+K_11@v_11;K_21=P_pred_21@L2_lambda.mT@torch.linalg.pinv(F_21);eta_upd_21=eta_pred_21+K_21@v_21;K_22=P_pred_22@L2_lambda.mT@torch.linalg.pinv(F_22);eta_upd_22=eta_pred_22+K_22@v_22;I_L=torch.eye(self.L,device=device);I_KL_11=I_L-K_11@L1_lambda;P_upd_11=I_KL_11@P_pred_11@I_KL_11.mT+K_11@R@K_11.mT;I_KL_21=I_L-K_21@L2_lambda;P_upd_21=I_KL_21@P_pred_21@I_KL_21.mT+K_21@R@K_21.mT;I_KL_22=I_L-K_22@L2_lambda;P_upd_22=I_KL_22@P_pred_22@I_KL_22.mT+K_22@R@K_22.mT;eta_marg_r1_t=eta_upd_11;P_marg_r1_t=P_upd_11;eta_marg_r2_t=W_21.view(N,1,1)*eta_upd_21+W_22.view(N,1,1)*eta_upd_22;P_marg_r2_t=W_21.view(N,1,1)*(P_upd_21+(eta_marg_r2_t-eta_upd_21)@(eta_marg_r2_t-eta_upd_21).transpose(-1,-2))+W_22.view(N,1,1)*(P_upd_22+(eta_marg_r2_t-eta_upd_22)@(eta_marg_r2_t-eta_upd_22).transpose(-1,-2));eta_marg_r1_tm1,P_marg_r1_tm1=eta_marg_r1_t,P_marg_r1_t;eta_marg_r2_tm1,P_marg_r2_tm1=eta_marg_r2_t,P_marg_r2_t;prob_tm1=torch.stack([prob_t_r1,prob_t_r2],dim=1);self.eta_filtered_r1_history[:,t,:]=eta_marg_r1_t.squeeze(-1);self.eta_filtered_r2_history[:,t,:]=eta_marg_r2_t.squeeze(-1);self.P_filtered_r1_history[:,t,:,:]=P_marg_r1_t;self.P_filtered_r2_history[:,t,:,:]=P_marg_r2_t;self.filtered_probs[:,t,:]=prob_tm1;y_pred_t=prob_tm1[:,0].view(N,1,1)*(L1_lambda@eta_marg_r1_t)+prob_tm1[:,1].view(N,1,1)*(L2_lambda@eta_marg_r2_t);self.predicted_y[:,t,:]=y_pred_t.squeeze(-1)
        return-total_log_likelihood

frs_initial_params={'B1_3fac':best_params_3fac['B1'],'B1_1fac':best_params_1fac['B1'],'lambda1_free_3fac':torch.cat([best_params_3fac['Lambda1'][1:3,0],best_params_3fac['Lambda1'][4:6,1],best_params_3fac['Lambda1'][7:9,2]]),'lambda1_free_1fac':best_params_1fac['Lambda1'][1:9,0],'Q_3fac':best_params_3fac['Q'],'Q_1fac':best_params_1fac['Q'],'R_3fac':best_params_3fac['R'],'R_1fac':best_params_1fac['R']}

if os.path.exists(RS_KF_MODEL_FILE):
    print(f"Loading pre-computed RS-KF model results from '{RS_KF_MODEL_FILE}'...");saved_data=torch.load(RS_KF_MODEL_FILE,weights_only=False);best_model_rs_state=saved_data['model_state_dict'];loss_rs=saved_data['loss']
    duration_rs = saved_data.get('duration', 0.0)
else:
    start_time_rs=time.time()
    best_loss_rs=float('inf');best_model_rs_state=None;patience=100;patience_counter=0;absolute_threshold=0.1;print("Starting RS-KF model training...");model_rs=RegimeSwitchingKF(O,initial_belief='informative',init_params=frs_initial_params).to(device);optimizer_rs=torch.optim.Adam(model_rs.parameters(),lr=1e-3);pbar = tqdm(range(20000), file=original_stdout)

    for epoch in pbar:
        optimizer_rs.zero_grad();loss=model_rs(Y_generated)
        if torch.isnan(loss):print(f"Run failed due to NaN loss.");break
        loss.backward();torch.nn.utils.clip_grad_norm_(model_rs.parameters(),1.0);optimizer_rs.step();pbar.set_description(f"[RS-KF Epoch {epoch+1}] loss: {loss.item():.4f} (Best: {best_loss_rs:.4f})")
        if best_loss_rs-loss.item()>absolute_threshold:best_loss_rs=loss.item();patience_counter=0;best_model_rs_state=model_rs.state_dict()
        else:patience_counter+=1
        if patience_counter>=patience:print(f"    -> Early stopping triggered at epoch {epoch+1}.");break
    loss_rs=best_loss_rs
    duration_rs=time.time()-start_time_rs
    print(f"RS-KF model training finished. Duration: {duration_rs:.2f}s");print(f"Saving RS-KF model results to '{RS_KF_MODEL_FILE}'...");results_to_save={'model_state_dict':best_model_rs_state,'loss':loss_rs,'duration':duration_rs};torch.save(results_to_save,RS_KF_MODEL_FILE)

# ----------------------------------------------------------------
# Part 6: モデルの評価と最終比較
# ----------------------------------------------------------------
print("\n--- 6. Model Evaluation and Final Comparison ---")

# --- 6a. BPSモデルの評価 ---
print("Evaluating BPS model...");
with torch.no_grad():
    # ▼▼▼【修正点2-5】Predictiveの実行時に time_weights を渡す（ただし、予測自体には影響しない）▼▼▼
    guide(Y_generated, factors_f, time_weights=pi_t);median_params = guide.median();estimated_logits = torch.stack([median_params[f'beta_{t}'] for t in range(Nt)]).permute(1, 0, 2);estimated_weights_bps = torch.nn.functional.softmax(estimated_logits, dim=-1).cpu().numpy();preds_3fac_numpy = preds_3fac_est.cpu().numpy();preds_1fac_numpy = preds_1fac_est.cpu().numpy();y_pred_mixed_bps = estimated_weights_bps[..., 0, np.newaxis] * preds_3fac_numpy + estimated_weights_bps[..., 1, np.newaxis] * preds_1fac_numpy;rmse_bps = calculate_rmse(Y_generated.cpu(), torch.from_numpy(y_pred_mixed_bps));predicted_states_bps = (estimated_weights_bps[:, :, 1] > 0.5).astype(int);sens_bps, spec_bps = calculate_sens_spec(actual_states, predicted_states_bps)

# --- 6b. RS-KFモデルの評価 ---
print("Evaluating RS-KF model...")
final_model_rs = RegimeSwitchingKF(O, initial_belief='informative', init_params=frs_initial_params).to(device)
if 'best_model_rs_state' in locals() and best_model_rs_state:final_model_rs.load_state_dict(best_model_rs_state)
else:saved_data=torch.load(RS_KF_MODEL_FILE,weights_only=False);final_model_rs.load_state_dict(saved_data['model_state_dict'])
with torch.no_grad():
    _ = final_model_rs(Y_generated);y_preds_rs = final_model_rs.predicted_y;predicted_probs_rs = final_model_rs.filtered_probs.cpu().numpy()
rmse_rs = calculate_rmse(Y_generated, y_preds_rs);predicted_states_rs = (predicted_probs_rs[:, :, 1] > 0.5).astype(int);sens_rs, spec_rs = calculate_sens_spec(actual_states, predicted_states_rs)

# --- 6c. ベースラインモデルの評価 ---
print("Evaluating Baseline models...")
with torch.no_grad():
    preds_3fac_full, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
    preds_1fac_full, _, _ = get_kalman_predictions_and_latents(Y_generated, **best_params_1fac)
rmse_3fac = calculate_rmse(Y_generated, preds_3fac_full);rmse_1fac = calculate_rmse(Y_generated, preds_1fac_full);sens_3fac, spec_3fac = 0.0, 1.0;sens_1fac, spec_1fac = 1.0, 0.0

# --- 6d. 最終比較テーブル ---
table_width = 128
print("\n"+"="*table_width)
print(f"{'Model':<30} | {'Loss':<15} | {'RMSE':<15} | {'Sensitivity':<15} | {'Specificity':<15} | {'Duration (s)':<15} |")
print("-"*table_width)
print(f"{'3-Factor Model':<30} | {loss_3fac:<15.4f} | {rmse_3fac:<15.4f} | {sens_3fac:<15.4f} | {spec_3fac:<15.4f} | {duration_3fac:<15.2f} |")
print(f"{'1-Factor Model':<30} | {loss_1fac:<15.4f} | {rmse_1fac:<15.4f} | {sens_1fac:<15.4f} | {spec_1fac:<15.4f} | {duration_1fac:<15.2f} |")
print(f"{'BPS Random Walk':<30} | {loss_bps:<15.4f} | {rmse_bps:<15.4f} | {sens_bps:<15.4f} | {spec_bps:<15.4f} | {duration_bps:<15.2f} |")
print(f"{'Regime-Switching KF':<30} | {loss_rs:<15.4f} | {rmse_rs:<15.4f} | {sens_rs:<15.4f} | {spec_rs:<15.4f} | {duration_rs:<15.2f} |")
print("Note: Loss for Baseline/RS-KF is NegLogLikelihood. Loss for BPS is ELBO.")
print("      RMSE, Sensitivity, and Specificity are directly comparable metrics.")
print("      Duration is 0.00 if model was loaded from cache.")
print("="*table_width)

# ----------------------------------------------------------------
# Part 7: 可視化
# ----------------------------------------------------------------
print("\n--- 7. Visualization of Results ---")
time_points_vis=np.arange(Nt);state1_proportion_actual=(actual_states==1).mean(axis=0);avg_weights_bps_over_time=estimated_weights_bps.transpose(1,0,2).mean(axis=1)

print("\n--- Generating Aggregate Model Comparison Plot ---")
avg_probs_rs=predicted_probs_rs.mean(axis=0);plt.style.use('seaborn-v0_8-whitegrid');fig,(ax1,ax3)=plt.subplots(1,2,figsize=(20,8),sharey=True);fig.suptitle('Model Comparison: BPS Random Walk vs. Regime-Switching KF (Aggregate Performance)',fontsize=20);ax1.plot(time_points_vis,avg_weights_bps_over_time[:,0],'o-',color='royalblue',label='Avg. Weight for 3-Factor Model (State 1)',zorder=3);ax1.plot(time_points_vis,avg_weights_bps_over_time[:,1],'s-',color='firebrick',label='Avg. Weight for 1-Factor Model (State 2)',zorder=3);ax1.set_title('BPS Random Walk Model',fontsize=16);ax1.set_xlabel('Time Point',fontsize=12);ax1.set_ylabel('Estimated Model Weight',fontsize=12);ax1.legend(loc='upper left');ax1.set_ylim(-0.05,1.05);ax2=ax1.twinx();ax2.bar(time_points_vis,state1_proportion_actual,color='grey',alpha=0.2,label='Proportion in State 1 (Actual)');ax2.set_ylabel('Proportion in State 1 (Ground Truth)',fontsize=12);ax2.set_ylim(-0.05,1.05);ax2.grid(False);ax3.plot(time_points_vis,avg_probs_rs[:,0],'o-',color='royalblue',label='Avg. Prob. of Regime 1 (3-Factor)',zorder=3);ax3.plot(time_points_vis,avg_probs_rs[:,1],'s-',color='firebrick',label='Avg. Prob. of Regime 2 (1-Factor)',zorder=3);ax3.set_title('Regime-Switching KF Model',fontsize=16);ax3.set_xlabel('Time Point',fontsize=12);ax3.legend(loc='upper left');ax4=ax3.twinx();ax4.bar(time_points_vis,state1_proportion_actual,color='grey',alpha=0.2,label='Proportion in State 1 (Actual)');ax4.set_ylabel('Proportion in State 1 (Ground Truth)',fontsize=12);ax4.set_ylim(-0.05,1.05);ax4.grid(False);plt.tight_layout(rect=[0,0.03,1,0.95]);plt.savefig("model_comparison_plot.png");print("Aggregate model comparison plot saved to model_comparison_plot.png");plt.close(fig)

# ----------------------------------------------------------------
# Part 8: 個人レベルのプロット用データ準備
# ----------------------------------------------------------------
print("\n\n--- 8. Preparing data for individual-level plots ---")

# --- 8a. BPSモデルの信頼区間を生成 ---
print("\n--- 8a. Generating posterior samples for BPS model uncertainty ---")
guide.requires_grad_(False)
predictive = Predictive(bps_random_walk_model, guide=guide, num_samples=1000,
                        return_sites=[f"beta_{t}" for t in range(Nt)])

# ▼▼▼【修正点2-6】Predictiveの実行時に time_weights を渡す ▼▼▼
posterior_samples = predictive(Y_generated, factors_f, time_weights=pi_t)
print("Generated 1000 samples from the posterior distribution of beta.")

beta_samples = torch.stack([posterior_samples[f'beta_{t}'] for t in range(Nt)], dim=2)
prob_samples = torch.nn.functional.softmax(beta_samples, dim=-1)
prob_samples_s2 = prob_samples[..., 1]
prob_samples_s2_permuted = prob_samples_s2.permute(1, 2, 0)
bps_lower_ci = torch.quantile(prob_samples_s2_permuted, 0.025, dim=-1).cpu().numpy()
bps_upper_ci = torch.quantile(prob_samples_s2_permuted, 0.975, dim=-1).cpu().numpy()
print("Calculated 95% credible intervals for BPS state probabilities.")

# --- 8b. 潜在変数(eta)の全期間データを生成 ---
print("\n--- 8b. Generating full-period latent variable trajectories ---")
with torch.no_grad():
    _, latents_3fac_full, P_3fac_full = get_kalman_predictions_and_latents(Y_generated, **best_params_3fac)
    _, latents_1fac_full, P_1fac_full = get_kalman_predictions_and_latents(Y_generated, **best_params_1fac)
    latents_3fac_full = latents_3fac_full.cpu().numpy()
    P_3fac_full = P_3fac_full.cpu().numpy()
    latents_1fac_full = latents_1fac_full.cpu().numpy()

    _ = final_model_rs(Y_generated)
    eta_r1_full_rs = final_model_rs.eta_filtered_r1_history.cpu().numpy()
    P_r1_full_rs = final_model_rs.P_filtered_r1_history.cpu().numpy()
    eta_r2_full_rs = final_model_rs.eta_filtered_r2_history.cpu().numpy()
    P_r2_full_rs = final_model_rs.P_filtered_r2_history.cpu().numpy()

eta_true_hist_numpy = eta_true_history.cpu().numpy()

# ----------------------------------------------------------------
# Part 9: 個人レベルの可視化 (全員分・複数ファイルに分割)
# ----------------------------------------------------------------
print("\n--- 9. Visualization of Individual-level Results for all individuals ---")

all_individual_ids = list(range(N))
individuals_per_file = 25 
num_files = math.ceil(N / individuals_per_file)
actual_states_binary = (actual_states == 2).astype(int)

# --- 9a. 個人レベルのレジームスイッチ過程のプロット ---
print(f"\n--- 9a. Generating {num_files} files for individual regime switch plots with C.I. ---")

for file_idx in range(num_files):
    start_idx = file_idx * individuals_per_file
    end_idx = min((file_idx + 1) * individuals_per_file, N)
    ids_to_plot = all_individual_ids[start_idx:end_idx]
    num_rows = len(ids_to_plot)
    fig, axes = plt.subplots(num_rows, 2, figsize=(16, 2.5 * num_rows), sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f'Individual Regime Switch Trajectories (Filtered) - Individuals {start_idx+1}-{end_idx}', fontsize=20)
    for i, ind_id in enumerate(ids_to_plot):
        ax_bps = axes[i, 0]
        ax_bps.plot(time_points_vis, estimated_weights_bps[ind_id, :, 1], 'g-', label='BPS Prob. (State 2)', lw=1.5)
        ax_bps.fill_between(time_points_vis, bps_lower_ci[ind_id, :], bps_upper_ci[ind_id, :], color='green', alpha=0.2, label='95% Credible Interval')
        ax_bps.plot(time_points_vis, actual_states_binary[ind_id, :], 'r--', label='True State', lw=1.5)
        ax_bps.set_title(f'Individual #{ind_id+1} - BPS Random Walk Model', fontsize=10)
        ax_bps.set_ylabel('Prob. of State 2')
        ax_rs = axes[i, 1]
        ax_rs.plot(time_points_vis, predicted_probs_rs[ind_id, :, 1], 'b-', label='RS-KF Prob. (State 2)', lw=1.5)
        ax_rs.plot(time_points_vis, actual_states_binary[ind_id, :], 'r--', label='True State', lw=1.5)
        ax_rs.set_title(f'Individual #{ind_id+1} - RS-KF Model', fontsize=10)
        if i == 0:
            ax_bps.legend(loc='upper left')
            ax_rs.legend(loc='upper left')
    axes[-1, 0].set_xlabel('Time Point')
    axes[-1, 1].set_xlabel('Time Point')
    plt.ylim(-0.1, 1.1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_filename = f"individual_switches_plot_with_ci_part_{file_idx+1}.png"
    plt.savefig(save_filename)
    print(f"Individual switch plot saved to {save_filename}")
    plt.close(fig)

# --- 9b. 潜在変数 eta の比較プロット ---
print(f"\n--- 9b. Generating {num_files} files for latent variable plots with uncertainty ---")

for file_idx in range(num_files):
    start_idx = file_idx * individuals_per_file
    end_idx = min((file_idx + 1) * individuals_per_file, N)
    ids_to_plot = all_individual_ids[start_idx:end_idx]
    num_rows = len(ids_to_plot)
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 2.5 * num_rows), sharex=True, squeeze=False)
    fig.suptitle(f'Latent Variable Trajectories (Filtered with 95% CI) - Individuals {start_idx+1}-{end_idx}', fontsize=20)
    for i, ind_id in enumerate(ids_to_plot):
        # ▼▼▼【修正点1-4】修正後のeta_true_historyの構造に合わせて、マスクの定義を変更 ▼▼▼
        mask_true_r2 = ~np.isnan(eta_true_hist_numpy[ind_id, :, 3])
        mask_true_r1 = ~mask_true_r2

        for l in range(L1_state1):
            ax = axes[i, l]
            mean_rs = eta_r1_full_rs[ind_id, :, l]
            std_dev_rs = np.sqrt(P_r1_full_rs[ind_id, :, l, l])
            ax.plot(time_points_vis, mean_rs, 'b-', label='RS-KF', lw=1.5)
            ax.fill_between(time_points_vis, mean_rs - 1.96 * std_dev_rs, mean_rs + 1.96 * std_dev_rs, color='blue', alpha=0.15)
            ax.plot(time_points_vis, latents_3fac_full[ind_id, :, l], 'g-', label='Baseline (3-Fac)', lw=1.5)
            if np.any(mask_true_r1):
                ax.plot(time_points_vis[mask_true_r1], eta_true_hist_numpy[ind_id, :, l][mask_true_r1], 'r--', label='True', lw=1.5)
            ax.set_title(f'Ind. #{ind_id+1} - $\eta_{l+1}$ (3-Fac)', fontsize=10)
            if i == 0: ax.legend()
            if l == 0: ax.set_ylabel('Value')
        ax = axes[i, 3]
        mean_rs_w = eta_r2_full_rs[ind_id, :, 3]
        std_dev_rs_w = np.sqrt(P_r2_full_rs[ind_id, :, 3, 3])
        ax.plot(time_points_vis, mean_rs_w, 'b-', label='RS-KF', lw=1.5)
        ax.fill_between(time_points_vis, mean_rs_w - 1.96 * std_dev_rs_w, mean_rs_w + 1.96 * std_dev_rs_w, color='blue', alpha=0.15)
        ax.plot(time_points_vis, latents_1fac_full[ind_id, :, 0], 'g-', label='Baseline (1-Fac)', lw=1.5)
        
        # ▼▼▼【修正点1-5】修正後のeta_true_historyの構造に合わせて、プロットする真値の列を変更 (0 -> 3) ▼▼▼
        if np.any(mask_true_r2):
            ax.plot(time_points_vis[mask_true_r2], eta_true_hist_numpy[ind_id, :, 3][mask_true_r2], 'r--', label='True', lw=1.5)
            
        ax.set_title(f'Ind. #{ind_id+1} - $\eta$ (1-Fac)', fontsize=10)
        if i == 0: ax.legend()
    for col in range(4):
        axes[-1, col].set_xlabel('Time Point')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    save_filename = f"eta_trajectories_filtered_with_ci_part_{file_idx+1}.png"
    plt.savefig(save_filename)
    print(f"Latent variable plot saved to {save_filename}")
    plt.close(fig)

# --- Restore stdout ---
sys.stdout = original_stdout
print(f"\nAnalysis complete. Log saved to '{log_filename}'")