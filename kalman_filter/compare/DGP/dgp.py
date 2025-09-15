import math
import os
import random
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import tqdm, trange

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

# dgp.py (MODIFIED SECTION in Part 0)

# ----------------------------------------------------------------
# Part 0: パラメータとキャッシュファイル名の定義
# ----------------------------------------------------------------
print("--- 0. Defining Parameters ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★★★ DGPモード選択 ★★★
DGP_MODE = 'IMPLEMENT'

# ★★★ 変更点: DGP_MODEに応じてresultsディレクトリ名を動的に作成 ★★★
RESULTS_DIR = f"results_{DGP_MODE.replace(' 2.0', '_2.0')}"
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"All output files will be saved to the '{RESULTS_DIR}/' directory.")

# ログファイルの設定
log_filename = os.path.join(RESULTS_DIR, f"model_comparison_{DGP_MODE}.txt")
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, open(log_filename, 'w', encoding='utf-8'))

# ログファイル設定後に情報を出力
print(f"Using device: {device}")
print(f"Running in '{DGP_MODE}' mode.")


# ----------------------------------------------------------------
# Part 1: DGPパラメータの設定 (論文ベース)
# ----------------------------------------------------------------
print(f"\n--- 1. Configuring DGP Parameters based on Flückiger et al. (2021) ---")

if DGP_MODE == 'IMPLEMENT':
    N, Nt = 57, 15
elif DGP_MODE == 'IMPLEMENT 2.0':
    N, Nt = 80, 19

print(f"Set N = {N} participants and Nt = {Nt} sessions.")

O = 9
L1_state1 = 3
L1_state2 = 1
# ★★★ 変更点: データファイルの保存先をresultsディレクトリに指定 ★★★
DATA_FILE = os.path.join(RESULTS_DIR, f'simulation_data_{DGP_MODE}.pt')

# --- 状態遷移のパラメータ (共通) ---
gamma_intercept = -2
gamma_task = 0.5
gamma_goal = 0.5
gamma_bond = 0.5

# --- ノイズのパラメータ (共通) ---
Q1_VAR, Q2_VAR, R_VAR = 0.5, 0.1, 0.5
Q_state1 = torch.eye(L1_state1, device=device) * Q1_VAR
Q_state2 = torch.eye(L1_state2, device=device) * Q2_VAR
R_true = torch.eye(O, device=device) * R_VAR

# --- 状態ごとの自己回帰パラメータ ---
b0_true_state1 = torch.tensor([0.0, 0.0, 0.0], device=device).unsqueeze(1)
b0_true_state2 = torch.tensor([0.0], device=device).unsqueeze(1)
if DGP_MODE == 'IMPLEMENT':
    B1_true_state1 = torch.tensor([[0.04, 0.01, -0.11], [-0.01, 0.07, 0.13], [0.02, 0.11, 0.16]], device=device)
    B1_true_state2 = torch.tensor([[0.50]], device=device)
elif DGP_MODE == 'IMPLEMENT 2.0':
    B1_true_state1 = torch.tensor([[0.17, -0.06, 0.00], [0.14, 0.21, -0.10], [-0.29, -0.22, 0.11]], device=device)
    B1_true_state2 = torch.tensor([[0.24]], device=device)

# --- 因子負荷量Lambda (共通) ---
lambda1_true_values_state1=[1.2,0.8,1.1,0.9,1.3,0.7]
Lambda1_true_state1=torch.zeros(O,L1_state1,device=device)
Lambda1_true_state1[0,0]=1; Lambda1_true_state1[1,0]=lambda1_true_values_state1[0]; Lambda1_true_state1[2,0]=lambda1_true_values_state1[1]
Lambda1_true_state1[3,1]=1; Lambda1_true_state1[4,1]=lambda1_true_values_state1[2]; Lambda1_true_state1[5,1]=lambda1_true_values_state1[3]
Lambda1_true_state1[6,2]=1; Lambda1_true_state1[7,2]=lambda1_true_values_state1[4]; Lambda1_true_state1[8,2]=lambda1_true_values_state1[5]

lambda1_true_values_state2=[1.2,0.8,1.1,0.9,1.3,0.7,0.6,1.0]
Lambda1_true_state2=torch.zeros(O,L1_state2,device=device)
Lambda1_true_state2[0,0]=1.0
Lambda1_true_state2[1:9,0]=torch.tensor(lambda1_true_values_state2,device=device)

# ----------------------------------------------------------------
# Part 2: データ生成（ファイルが存在しない場合のみ実行）
# ----------------------------------------------------------------
print("\n--- 2. Generating Simulation Data ---")

if os.path.exists(DATA_FILE):
    print(f"Removing existing data file '{DATA_FILE}' to generate new data...")
    os.remove(DATA_FILE)

print("Generating new simulation data...")
Y_generated=torch.zeros(N,Nt,O,device=device)
actual_states=np.zeros((N,Nt))
q_dist_s1=MultivariateNormal(torch.zeros(L1_state1,device=device),Q_state1)
q_dist_s2=MultivariateNormal(torch.zeros(L1_state2,device=device),Q_state2)
r_dist=MultivariateNormal(torch.zeros(O,device=device),R_true)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

for i in trange(N,desc="Generating data for each person"):
    eta_history_i=torch.randn(L1_state1,1,device=device)
    current_state=1
    has_switched=False
    for t in range(Nt):
        # ★★★ 修正点: スイッチ判定を状態記録の前に行う ★★★
        # t>0 の場合にのみ、前期のetaに基づいてスイッチを判定
        if current_state==1 and t>0:
            z = gamma_intercept + (eta_history_i[0] * gamma_task +
                                   eta_history_i[1] * gamma_goal +
                                   eta_history_i[2] * gamma_bond)

            if random.random() < (1 / (1 + math.exp(-z))):
                current_state=2

        # 判定後の状態で現在の状態を記録する
        actual_states[i,t]=current_state

        # 記録した状態に基づいてetaを更新
        if current_state==1:
            eta_t=(B1_true_state1@eta_history_i)+q_dist_s1.sample().reshape(L1_state1,1)
            y_mean_t=Lambda1_true_state1@eta_t
        else:
            if not has_switched:
                eta_history_i=torch.tensor([eta_history_i.mean()],device=device).reshape(L1_state2,1)
                has_switched=True
            eta_t=(B1_true_state2@eta_history_i)+q_dist_s2.sample().reshape(L1_state2,1)
            y_mean_t=Lambda1_true_state2@eta_t

        Y_generated[i,t,:]=(y_mean_t+r_dist.sample().reshape(O,1)).squeeze()
        eta_history_i=eta_t

print("Simulation data generated.")
print(f"Saving simulation data to '{DATA_FILE}'...")
torch.save({'Y_generated':Y_generated.cpu(),'actual_states':actual_states, 'eta_true':eta_history_i.cpu()},DATA_FILE)
print("Data saving complete.")

# ----------------------------------------------------------------
# Part 3: DGP結果の可視化 (論文Figure 1との比較)
# ----------------------------------------------------------------
print("\n--- 3. Visualizing DGP Results vs. Paper's Figure 1 ---")

has_switched = np.cumsum(actual_states == 2, axis=1) > 0
cumulative_switches = np.sum(has_switched, axis=0)
cumulative_switch_percent = (cumulative_switches / N) * 100

# ★★★ 修正点: IMPLEMENT 2.0の参照データを完全なものに更新 ★★★
paper_data = {
    'IMPLEMENT': {
        'sessions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        'percent':  [0, 22, 35, 42, 45, 48, 50, 52, 56, 58, 60, 63, 65, 68, 68]
    },
    'IMPLEMENT 2.0': {
        'sessions': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        'percent':  [0, 15, 28, 38, 42, 45, 49, 52, 55, 57, 58, 60, 62, 63, 64, 65, 66, 67, 68]
    }
}

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

sessions = np.arange(1, Nt + 1)
ax.plot(sessions, cumulative_switch_percent,
        label=f'DGP ({DGP_MODE})', color='red', linewidth=2.5)

ref_mode = DGP_MODE
paper_sessions = np.array(paper_data[ref_mode]['sessions']) + 1
paper_percent = paper_data[ref_mode]['percent']
ax.plot(paper_sessions, paper_percent,
        label=f'Paper ({ref_mode})', color='black', linestyle='--', alpha=0.8)

ax.set_title(f"Cumulative Percentage of Switches: DGP vs. Paper ({DGP_MODE})")
ax.set_xlabel("Sessions")
ax.set_ylabel("Cumulative Percent of Switches")
ax.set_ylim(0, 100)
ax.set_xlim(left=1, right=Nt)
ax.legend()
plt.tight_layout()

# ★★★ 変更点: プロットの保存先をresultsディレクトリに指定 ★★★
plot_filename = os.path.join(RESULTS_DIR, f'dgp_switch_comparison_{DGP_MODE}.png')
plt.savefig(plot_filename)
plt.show()

print(f"\n✅ Plot saved as '{plot_filename}'")

# ----------------------------------------------------------------
# Part 4: Calculate Overall State Proportions
# ----------------------------------------------------------------
print("\n--- 4. Calculating Overall State Proportions ---")

# 総データポイント数
total_data_points = N * Nt

# 各状態の発生回数をカウント
num_state1 = np.sum(actual_states == 1)
num_state2 = np.sum(actual_states == 2)

# 各状態の割合を計算
percent_state1 = (num_state1 / total_data_points) * 100
percent_state2 = (num_state2 / total_data_points) * 100

print(f"\n📊 Analysis for DGP Mode: '{DGP_MODE}'")
print(f"Total Data Points (N x Nt): {total_data_points}")
print("-" * 40)
print(f"3-Factor State (State 1): {num_state1} points ({percent_state1:.2f}%)")
print(f"1-Factor State (State 2): {num_state2} points ({percent_state2:.2f}%)")
print("-" * 40)

# ----------------------------------------------------------------
# Part 5: Visualize Generated Y Data
# ----------------------------------------------------------------
print("\n--- 5. Visualizing The Trajectory of Generated Y ---")

# Calculate the mean of the first 3 observed variables across all individuals
y_mean_across_individuals = Y_generated[:, :, 0:3].mean(dim=0).cpu().numpy()

# Create the plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 6))

sessions = np.arange(1, Nt + 1)
ax.plot(sessions, y_mean_across_individuals[:, 0], 'o-', label='Avg. Y[0] (Task Indicator)', markersize=4)
ax.plot(sessions, y_mean_across_individuals[:, 1], 's-', label='Avg. Y[1] (Task Item 2)', markersize=4)
ax.plot(sessions, y_mean_across_individuals[:, 2], '^-', label='Avg. Y[2] (Task Item 3)', markersize=4)

# Formatting the plot
ax.set_title(f"Average Trajectory of Observed Variables (Mode: {DGP_MODE})", fontsize=16)
ax.set_xlabel("Sessions", fontsize=12)
ax.set_ylabel("Average Y Value", fontsize=12)
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save and show the plot
# ★★★ 変更点: プロットの保存先をresultsディレクトリに指定 ★★★
plot_filename_y = os.path.join(RESULTS_DIR, f'dgp_y_trajectory_{DGP_MODE}.png')
plt.savefig(plot_filename_y)
plt.show()

print(f"\n✅ Y trajectory plot saved as '{plot_filename_y}'")