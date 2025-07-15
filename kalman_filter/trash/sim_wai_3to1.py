import torch
import numpy as np
import random
import math
import sys
from torch.distributions.multivariate_normal import MultivariateNormal

# Teeクラス (コンソールとファイルへの同時出力)
class Tee:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

# Teeクラスを使って全体の処理を囲む
with Tee("sim_wai_3to1_result.txt"):
    # ----------------------------------------------------------------
    # Part 0: モデルパラメータの定義 (3因子 -> 1因子モデル)
    # ----------------------------------------------------------------
    print("--- 0. Defining Parameters for the 3-to-1 Factor Regime-Switching Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 全体の設定 (Symptom項目を削除)
    N = 100
    Nt = 20
    O = 9  # 観測変数 (Alliance項目x9)

    # --- State 1: 3因子モデル (治療初期, Symptom因子を削除) ---
    L1_state1 = 3  # 潜在変数の数 (Task, Goal, Bond)

    B1_state1_matrix = np.array([
        [0.4, 0.0, 0.0],
        [0.0, 0.45, 0.0],
        [0.0, 0.0, 0.5]
    ], dtype=np.float32)
    B1_true_state1 = torch.tensor(B1_state1_matrix, device=device)

    Lambda1_state1_matrix = np.zeros((O, L1_state1), dtype=np.float32)
    Lambda1_state1_matrix[0:3, 0] = [1.0, 0.8, 0.7] # Task loadings
    Lambda1_state1_matrix[3:6, 1] = [1.0, 1.2, 0.8] # Goal loadings
    Lambda1_state1_matrix[6:9, 2] = [1.0, 0.9, 0.7] # Bond loadings
    Lambda1_true_state1 = torch.tensor(Lambda1_state1_matrix, device=device)

    # --- State 2: 1因子モデル (治療後期, 統合WAI因子のみ) ---
    L1_state2 = 1
    B1_state2_matrix = np.array([[0.6]], dtype=np.float32) # WAI -> WAI
    B1_true_state2 = torch.tensor(B1_state2_matrix, device=device)

    Lambda1_state2_matrix = np.zeros((O, L1_state2), dtype=np.float32)
    Lambda1_state2_matrix[0:9, 0] = [0.9, 0.8, 0.7, 1.0, 1.1, 0.9, 0.9, 0.8, 0.7] # WAI loadings
    Lambda1_true_state2 = torch.tensor(Lambda1_state2_matrix, device=device)

    # 共通パラメータ (Symptom関連を削除)
    Q_state1 = torch.eye(L1_state1, device=device)
    Q_state2 = torch.eye(L1_state2, device=device)
    R_true = torch.eye(O, device=device)
    gamma_intercept = -2.5
    gamma_task = 0.1
    gamma_goal = 0.1
    gamma_bond = 0.1

    # ----------------------------------------------------------------
    # Part 1: データ生成 (体制スイッチモデル)
    # ----------------------------------------------------------------
    print("\n--- 1. Generating Simulation Data from Regime-Switching Model ---")

    Y_generated = torch.zeros(N, Nt, O, device=device)
    q_dist_s1 = MultivariateNormal(torch.zeros(L1_state1, device=device), Q_state1)
    q_dist_s2 = MultivariateNormal(torch.zeros(L1_state2, device=device), Q_state2)
    r_dist = MultivariateNormal(torch.zeros(O, device=device), R_true)

    switch_points = []
    for i in range(N):
        eta_history = torch.zeros(L1_state1, 1, device=device)
        current_state = 1
        has_switched = False
        for t in range(Nt):
            if current_state == 1 and t > 0:
                task_prev, goal_prev, bond_prev = eta_history.squeeze().tolist()
                # zの計算からSymptom項を削除
                z = gamma_intercept + gamma_task * task_prev + gamma_goal * goal_prev + gamma_bond * bond_prev
                switch_prob = 1 / (1 + math.exp(-z))
                if random.random() < switch_prob:
                    current_state = 2
                    if not has_switched:
                        switch_points.append(t)
                        has_switched = True
                    
                    wai_prev = (task_prev + goal_prev + bond_prev) / 3
                    # eta_historyを1因子用に変更
                    eta_history = torch.tensor([wai_prev], device=device).reshape(L1_state2, 1)

            if current_state == 1:
                eta_mean_t = B1_true_state1 @ eta_history
                noise_q = q_dist_s1.sample().reshape(L1_state1, 1)
                eta_t = eta_mean_t + noise_q
                y_mean_t = Lambda1_true_state1 @ eta_t
            else: # current_state == 2
                eta_mean_t = B1_true_state2 @ eta_history
                noise_q = q_dist_s2.sample().reshape(L1_state2, 1)
                eta_t = eta_mean_t + noise_q
                y_mean_t = Lambda1_true_state2 @ eta_t
            
            noise_r = r_dist.sample().reshape(O, 1)
            y_t = y_mean_t + noise_r
            Y_generated[i, t, :] = y_t.squeeze()
            eta_history = eta_t

    print("Simulation data with regime-switching process generated.")
    print(f"Number of individuals switched to State 2: {len(switch_points)} / {N}")
    if switch_points:
        print(f"Average switch time point (for those who switched): {np.mean(switch_points):.4f}")


    # ----------------------------------------------------------------
    # Part 2: 共通の関数定義
    # ----------------------------------------------------------------
    print("\n--- 2. Defining Common Functions ---")

    def kalman_filter_torch(Y1, b0, B1, Lambda1, Q, R, eta1_i0_0, P_i0_0):
        N, Nt, O1 = Y1.shape
        L1 = B1.shape[0]
        total_log_likelihood = torch.tensor(0.0, device=Y1.device)
        I_mat = torch.eye(L1, device=Y1.device)
        for i in range(N):
            eta_prev = eta1_i0_0.clone()
            P_prev = P_i0_0.clone()
            y1_i = Y1[i, :, :]
            for t in range(Nt):
                y1_it = y1_i[t, :].reshape(O1, 1)
                eta_pred = b0 + B1 @ eta_prev
                P_pred = B1 @ P_prev @ B1.T + Q
                v_it = y1_it - Lambda1 @ eta_pred
                F_it = Lambda1 @ P_pred @ Lambda1.T + R
                F_it_inv = torch.linalg.pinv(F_it)
                K_it = P_pred @ Lambda1.T @ F_it_inv
                eta_updated = eta_pred + K_it @ v_it
                term1 = I_mat - K_it @ Lambda1
                P_updated = term1 @ P_pred @ term1.T + K_it @ R @ K_it.T
                log_det_F_it = torch.logdet(F_it)
                exponent_term = -0.5 * v_it.T @ F_it_inv @ v_it
                log_likelihood_it = -0.5 * O1 * math.log(2 * math.pi) - 0.5 * log_det_F_it + exponent_term
                total_log_likelihood += log_likelihood_it.squeeze()
                eta_prev = eta_updated
                P_prev = P_updated
        return total_log_likelihood

    # --- 推定の共通設定 ---
    num_runs = 5
    learning_rate = 0.001
    num_epochs = 2000
    l2_decay = 0.01

    # ----------------------------------------------------------------
    # Part 3: 3因子モデルによるパラメータ推定
    # ----------------------------------------------------------------
    print("\n\n--- 3. Parameter Estimation using a 3-Factor Model ---")
    L1_3fac = 3
    best_loss_3fac = float('inf')
    best_params_3fac = {}
    Q_est_3fac = (torch.eye(L1_3fac) * 0.5).to(device)
    R_est_3fac = (torch.eye(O) * 1.0).to(device)

    for run in range(num_runs):
        print(f"\n--- Starting Run (3-Factor): {run + 1} / {num_runs} ---")
        b0 = torch.randn(L1_3fac, 1, requires_grad=True, device=device)
        # B1は対角行列と仮定し、対角成分のみを推定
        b1_free_params = torch.randn(3, requires_grad=True, device=device)
        lambda1_free_params = torch.randn(6, requires_grad=True, device=device)
        params_to_optimize = [b0, b1_free_params, lambda1_free_params]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=l2_decay)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            B1 = torch.diag(b1_free_params) # 対角行列を生成
            Lambda1 = torch.zeros(O, L1_3fac, device=device)
            Lambda1[0, 0] = 1; Lambda1[1, 0] = lambda1_free_params[0]; Lambda1[2, 0] = lambda1_free_params[1]
            Lambda1[3, 1] = 1; Lambda1[4, 1] = lambda1_free_params[2]; Lambda1[5, 1] = lambda1_free_params[3]
            Lambda1[6, 2] = 1; Lambda1[7, 2] = lambda1_free_params[4]; Lambda1[8, 2] = lambda1_free_params[5]
            
            log_likelihood = kalman_filter_torch(
                Y1=Y_generated, b0=b0, B1=B1, Lambda1=Lambda1, Q=Q_est_3fac, R=R_est_3fac,
                eta1_i0_0=torch.zeros(L1_3fac, 1, device=device),
                P_i0_0=torch.eye(L1_3fac, device=device) * 1e3
            )
            loss = -log_likelihood
            if torch.isnan(loss) or torch.isinf(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"Run: {run + 1}, Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
        
        if not (torch.isnan(loss) or torch.isinf(loss)) and loss.item() < best_loss_3fac:
            best_loss_3fac = loss.item()
            best_params_3fac = {
                'b1_free_params': b1_free_params.detach().clone(),
                'lambda1_free_params': lambda1_free_params.detach().clone()
            }
            print(f"!!! New best result for 3-Factor Model found in run {run + 1} !!!")

    # ----------------------------------------------------------------
    # Part 4: 1因子モデルによるパラメータ推定
    # ----------------------------------------------------------------
    print("\n\n--- 4. Parameter Estimation using a 1-Factor Model ---")
    L1_1fac = 1
    best_loss_1fac = float('inf')
    best_params_1fac = {}
    Q_est_1fac = (torch.eye(L1_1fac) * 0.5).to(device)
    R_est_1fac = (torch.eye(O) * 1.0).to(device)

    for run in range(num_runs):
        print(f"\n--- Starting Run (1-Factor): {run + 1} / {num_runs} ---")
        b0 = torch.randn(L1_1fac, 1, requires_grad=True, device=device)
        B1 = torch.randn(L1_1fac, L1_1fac, requires_grad=True, device=device) # 1x1行列
        lambda1_free_params = torch.randn(8, requires_grad=True, device=device)
        params_to_optimize = [b0, B1, lambda1_free_params]
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=l2_decay)
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            Lambda1 = torch.zeros(O, L1_1fac, device=device)
            Lambda1[0, 0] = 1.0
            Lambda1[1:9, 0] = lambda1_free_params[0:8]
            
            log_likelihood = kalman_filter_torch(
                Y1=Y_generated, b0=b0, B1=B1, Lambda1=Lambda1, Q=Q_est_1fac, R=R_est_1fac,
                eta1_i0_0=torch.zeros(L1_1fac, 1, device=device),
                P_i0_0=torch.eye(L1_1fac, device=device) * 1e3
            )
            loss = -log_likelihood
            if torch.isnan(loss) or torch.isinf(loss): break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
            optimizer.step()
            if (epoch + 1) % 500 == 0 or epoch == 0:
                print(f"Run: {run + 1}, Epoch: {epoch + 1}, Loss: {loss.item():.4f}")
                
        if not (torch.isnan(loss) or torch.isinf(loss)) and loss.item() < best_loss_1fac:
            best_loss_1fac = loss.item()
            best_params_1fac = {
                'B1': B1.detach().clone(),
                'lambda1_free_params': lambda1_free_params.detach().clone()
            }
            print(f"!!! New best result for 1-Factor Model found in run {run + 1} !!!")

    # ----------------------------------------------------------------
    # Part 5: 結果の表示とモデル比較
    # ----------------------------------------------------------------
    print("\n\n--- 5. Final Results and Model Comparison ---\n")

    if best_params_3fac:
        print("--- Best 3-Factor Model Estimation Results ---")
        print(f"Best Negative Log-Likelihood (Loss): {best_loss_3fac:.4f}")
        B1_est_3fac = torch.diag(best_params_3fac['b1_free_params'])
        Lambda1_est_3fac = torch.zeros(O, L1_3fac)
        lambda_params = best_params_3fac['lambda1_free_params']
        Lambda1_est_3fac[0,0]=1; Lambda1_est_3fac[1,0]=lambda_params[0]; Lambda1_est_3fac[2,0]=lambda_params[1]
        Lambda1_est_3fac[3,1]=1; Lambda1_est_3fac[4,1]=lambda_params[2]; Lambda1_est_3fac[5,1]=lambda_params[3]
        Lambda1_est_3fac[6,2]=1; Lambda1_est_3fac[7,2]=lambda_params[4]; Lambda1_est_3fac[8,2]=lambda_params[5]
        print("True B1 (State 1):\n", B1_true_state1.cpu().numpy())
        print("Estimated B1 (3-Factor):\n", B1_est_3fac.cpu().numpy())
        print("---")
        print("True Lambda1 (State 1):\n", Lambda1_true_state1.cpu().numpy())
        print("Estimated Lambda1 (3-Factor):\n", Lambda1_est_3fac.cpu().numpy())

    if best_params_1fac:
        print("\n--- Best 1-Factor Model Estimation Results ---")
        print(f"Best Negative Log-Likelihood (Loss): {best_loss_1fac:.4f}")
        Lambda1_est_1fac = torch.zeros(O, L1_1fac)
        lambda_params_1fac = best_params_1fac['lambda1_free_params']
        Lambda1_est_1fac[0,0]=1
        Lambda1_est_1fac[1:9, 0] = lambda_params_1fac[0:8]
        print("True B1 (State 2):\n", B1_true_state2.cpu().numpy())
        print("Estimated B1 (1-Factor):\n", best_params_1fac['B1'].cpu().numpy())
        print("---")
        print("True Lambda1 (State 2):\n", Lambda1_true_state2.cpu().numpy())
        print("Estimated Lambda1 (1-Factor):\n", Lambda1_est_1fac.cpu().numpy())

    if best_params_3fac and best_params_1fac:
        print("\n--- Model Fit Comparison ---")
        print(f"3-Factor Model Best Loss: {best_loss_3fac:.4f}")
        print(f"1-Factor Model Best Loss: {best_loss_1fac:.4f}")
        if best_loss_3fac < best_loss_1fac:
            print("The 3-Factor model provides a better fit to the data based on log-likelihood.")
        elif best_loss_1fac < best_loss_3fac:
            print("The 1-Factor model provides a better fit to the data based on log-likelihood.")
        else:
            print("The models have comparable fit based on log-likelihood.")