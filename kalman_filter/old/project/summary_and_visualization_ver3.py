import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- 設定 ---
DGP_MODE = 'IMPLEMENT'
NUM_REALIZATIONS = 30
BASE_RESULTS_DIR = 'results'
MODE_DIR = DGP_MODE.replace(' 2.0', '_2.0')
SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, MODE_DIR, 'summary')
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"--- 結果の集計と可視化を開始します ---")
print(f"モード: {DGP_MODE}")
print(f"リアライゼーション数: {NUM_REALIZATIONS}")
print(f"結果の保存先: {SUMMARY_DIR}")

# --- データ収集用の関数 (この関数全体を置き換え) ---
def parse_log_file(log_path, model_type):
    metrics = {}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return metrics

    # 既存の評価指標パース
    patterns = {
        'Likelihood': r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)",
        'RMSE': r"Y Prediction RMSE\s*\|\s*(\d+\.\d+)",
        'Sensitivity': r"State Detection Sensitivity\s*\|\s*(\d+\.\d+)",
        'Specificity': r"State Detection Specificity\s*\|\s*(\d+\.\d+)",
        'Duration': r"Training Duration \(s\)\s*\|\s*(\d+\.\d+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))

    # パラメータパース用の内部関数
    def parse_matrix(matrix_str):
        cleaned_rows = [row.strip().replace('[', '').replace(']', '') for row in matrix_str.strip().split('\n')]
        return np.array([[float(v) for v in row.split()] for row in cleaned_rows if row])

    def parse_vector_from_log(vector_str):
        # Cleans strings like '[0. 0. 0.]' or '[-0.  0.  0.]'
        cleaned_str = vector_str.strip().replace('[', '').replace(']', '').replace(',', '')
        return np.fromstring(cleaned_str, sep=' ')

    # モデルタイプに応じたパース処理
    if model_type == 'VAR':
        try:
            s3_content = content.split("--- 3-Factor Model Estimated vs. True Parameters ---")[1].split("--- 1-Factor Model Estimated vs. True Parameters ---")[0]
            s1_content = content.split("--- 1-Factor Model Estimated vs. True Parameters ---")[1]

            # --- 3-Factor VAR ---
            if (m := re.search(r"b0 \(Intercept\): Estimated = (\[.*?\])", s3_content)):
                metrics['b0_3fac'] = parse_vector_from_log(m.group(1))
            if (m := re.search(r"-- Estimated B1 \(State Transition\) --\s*\n(.*?)\n-- True B1 --", s3_content, re.DOTALL)):
                metrics['B1_3fac'] = parse_matrix(m.group(1))
            if (m := re.search(r"-- Estimated Q diag \(System Noise\): --\s*\n\s*(.*?)\n-- True Q diag --", s3_content, re.DOTALL)):
                metrics['Q_3fac_diag'] = parse_vector_from_log(m.group(1))
            if (m := re.search(r"-- Estimated R diag \(Measurement Noise\): --\s*\n\s*(.*?)\n-- True R diag --", s3_content, re.DOTALL)):
                metrics['R_3fac_diag'] = parse_vector_from_log(m.group(1))
            if (m := re.search(r"-- Estimated Lambda1 free \(Factor Loading\): --\s*\n\s*(.*?)\n-- True Lambda1 free --", s3_content, re.DOTALL)):
                metrics['lambda_free_3fac'] = parse_vector_from_log(m.group(1))
            
            # --- 1-Factor VAR ---
            if (m := re.search(r"b0 \(Intercept\): Estimated = (\[.*?\])", s1_content)):
                 metrics['b0_1fac'] = parse_vector_from_log(m.group(1))
            if (m := re.search(r"-- Estimated B1 \(State Transition\): --\s*\n\s*\[\[(.*?)\]\]", s1_content, re.DOTALL)):
                metrics['B1_1fac'] = np.array([[float(m.group(1))]])
            if (m := re.search(r"-- Estimated Q diag \(System Noise\): --\s*\n\s*(.*?)\n-- True Q diag --", s1_content, re.DOTALL)):
                metrics['Q_1fac_diag'] = parse_vector_from_log(m.group(1))
            if (m := re.search(r"-- Estimated R diag \(Measurement Noise\): --\s*\n\s*(.*?)\n-- True R diag --", s1_content, re.DOTALL)):
                metrics['R_1fac_diag'] = parse_vector_from_log(m.group(1))
            if (m := re.search(r"-- Estimated Lambda1 free \(Factor Loading\): --\s*\n\s*(.*?)\n-- True Lambda1 free --", s1_content, re.DOTALL)):
                metrics['lambda_free_1fac'] = parse_vector_from_log(m.group(1))
        except IndexError:
            pass

    elif model_type == 'FRS':
        if (m := re.search(r"-- Estimated B1 \(3-Factor\) --\s*\n(.*?)\n\n-- Estimated B1 \(1-Factor\) --", content, re.DOTALL)):
            metrics['B1_3fac'] = parse_matrix(m.group(1))
        if (m := re.search(r"-- Estimated B1 \(1-Factor\) --\s*\n\s*\[\[(.*?)\]\]", content, re.DOTALL)):
            metrics['B1_1fac'] = np.array([[float(m.group(1))]])
        if (m := re.search(r"-- Estimated Q diag --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            q_vec = parse_vector_from_log(m.group(1))
            metrics['Q_3fac_diag'] = q_vec[:3]
            metrics['Q_1fac_diag'] = q_vec[3:]
        if (m := re.search(r"-- Estimated R diag --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            metrics['R_diag'] = parse_vector_from_log(m.group(1))
        if (m := re.search(r"-- Estimated Lambda free \(State 1\) --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            metrics['lambda_free_3fac'] = parse_vector_from_log(m.group(1))
        if (m := re.search(r"-- Estimated Lambda free \(State 2\) --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            metrics['lambda_free_1fac'] = parse_vector_from_log(m.group(1))     
            
    return metrics

# --- データ収集メインループ (このブロック全体を置き換え) ---
all_params_data = []
for i in range(1, NUM_REALIZATIONS + 1):
    frs_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'frs_results_run_{i}', f'frs_log_{DGP_MODE}_run_{i}.txt')
    frs_params = parse_log_file(frs_log_file, 'FRS')
    
    var_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'var_results_run_{i}', f'var_log_{DGP_MODE}_run_{i}.txt')
    var_params = parse_log_file(var_log_file, 'VAR')

    # FRS データを整形
    for p_name, p_val in frs_params.items():
        if isinstance(p_val, (np.ndarray, float)):
            p_val = np.atleast_1d(p_val) 
            for index, val in np.ndenumerate(p_val):
                all_params_data.append({'Realization': i, 'Model': 'FRS', 'ParamGroup': p_name, 'ParamName': f'{p_name}{index}', 'Value': val})

    # VAR データを整形
    for p_name, p_val in var_params.items():
        if isinstance(p_val, (np.ndarray, float)):
            model_suffix = '_3fac' if '3fac' in p_name else '_1fac'
            param_group = p_name 
            p_val = np.atleast_1d(p_val)
            for index, val in np.ndenumerate(p_val):
                all_params_data.append({'Realization': i, 'Model': f'VAR{model_suffix}', 'ParamGroup': param_group, 'ParamName': f'{p_name}{index}', 'Value': val})

# --- 全パラメータを格納したDataFrameを作成 ---
params_df = pd.DataFrame(all_params_data)

# --- 真値の定義 (動的に切り替え) ---
true_params_common = {
    'Q_3fac_diag': np.array([0.5, 0.5, 0.5]),
    'Q_1fac_diag': np.array([0.1]),
    'R_diag': np.array([0.5] * 9),
    'lambda_free_3fac': [1.2, 0.8, 1.1, 0.9, 1.3, 0.7],
    'lambda_free_1fac': [1.2, 0.8, 1.0, 1.1, 0.9, 1.0, 1.3, 0.7]
}

if DGP_MODE == 'IMPLEMENT':
    b1_3fac_true = np.array([[0.04, 0.01, -0.11], [-0.01, 0.07, 0.13], [0.02, 0.11, 0.16]])
    b1_1fac_true = np.array([[0.50]])
elif DGP_MODE == 'IMPLEMENT 2.0':
    b1_3fac_true = np.array([[0.17, -0.06, 0.00], [0.14, 0.21, -0.10], [-0.29, -0.22, 0.11]])
    b1_1fac_true = np.array([[0.24]])
else:
    raise ValueError(f"DGP_MODE '{DGP_MODE}' に対応する真値が定義されていません。")

true_params = {
    'B1_3fac': b1_3fac_true,
    'B1_1fac': b1_1fac_true,
    'b0_3fac': np.array([0.0, 0.0, 0.0]), # ★ b0の真値を追加
    'b0_1fac': np.array([0.0]),           # ★ b0の真値を追加
    **true_params_common
}
print(f"\n'{DGP_MODE}' モードに対応する真値を設定しました。")

# --- 色設定 ---
color_palette = {
    'FRS': sns.color_palette("muted")[1],
    'VAR_1fac': sns.color_palette("muted")[3],
    'VAR_3fac': sns.color_palette("muted")[2]
}

# --- プロット作成 ---
sns.set_theme(style="whitegrid")

# ★★★ 2x2のサブプロットに変更 ★★★
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Structural Model Parameter Estimates', fontsize=20, y=1.0)

# --- データの前処理：プロットごとにデータを分割 ---
# プロット1: B1 対角成分
b1_diag_data = params_df[(params_df['ParamGroup'] == 'B1_3fac') & (params_df['ParamName'].str.contains('3fac\\(0, 0\\)|3fac\\(1, 1\\)|3fac\\(2, 2\\)'))].copy()
b1_1fac_data = params_df[params_df['ParamGroup'] == 'B1_1fac'].copy()
b1_1fac_data['ParamName'] = 'B1_1fac(0, 0)'
b1_diag_combined = pd.concat([b1_diag_data, b1_1fac_data])
b1_diag_combined['sort_key'] = b1_diag_combined['ParamName'].apply(lambda x: 0 if '3fac(0, 0)' in x else 1 if '3fac(1, 1)' in x else 2 if '3fac(2, 2)' in x else 3)
b1_diag_combined = b1_diag_combined.sort_values(by='sort_key')
y_order_b1_diag = b1_diag_combined['ParamName'].unique().tolist()

# プロット2: B1 非対角成分
b1_off_diag_data = params_df[(params_df['ParamGroup'] == 'B1_3fac') & (~params_df['ParamName'].str.contains('3fac\\(0, 0\\)|3fac\\(1, 1\\)|3fac\\(2, 2\\)'))].copy()
b1_off_diag_data['sort_key'] = b1_off_diag_data['ParamName'].apply(lambda x: 0 if '3fac(0, 1)' in x else 1 if '3fac(0, 2)' in x else 2 if '3fac(1, 0)' in x else 3 if '3fac(1, 2)' in x else 4 if '3fac(2, 0)' in x else 5)
b1_off_diag_data = b1_off_diag_data.sort_values(by='sort_key')
y_order_b1_off_diag = b1_off_diag_data['ParamName'].unique().tolist()

# プロット3: Q Diagonals
q_data_combined = params_df[(params_df['ParamGroup'] == 'Q_3fac_diag') | (params_df['ParamGroup'] == 'Q_1fac_diag')].copy()
q_data_combined['sort_key'] = q_data_combined['ParamName'].apply(lambda x: 0 if '3fac(0,)' in x else 1 if '3fac(1,)' in x else 2 if '3fac(2,)' in x else 3)
q_data_combined = q_data_combined.sort_values(by='sort_key')
y_order_q_diag = q_data_combined['ParamName'].unique().tolist()

# ★ プロット4: b0 Intercepts
b0_data = params_df[(params_df['ParamGroup'] == 'b0_3fac') | (params_df['ParamGroup'] == 'b0_1fac')].copy()
b0_data['sort_key'] = b0_data['ParamName'].apply(lambda x: 0 if '3fac(0,)' in x else 1 if '3fac(1,)' in x else 2 if '3fac(2,)' in x else 3)
b0_data = b0_data.sort_values(by='sort_key')
y_order_b0 = b0_data['ParamName'].unique().tolist()

# --- プロットの描画 ---
# プロット1: B1 Diagonals
ax0 = axes[0, 0]
sns.boxplot(data=b1_diag_combined, y='ParamName', x='Value', hue='Model', ax=ax0, palette=color_palette, hue_order=['FRS', 'VAR_3fac', 'VAR_1fac'], order=y_order_b1_diag, boxprops=dict(alpha=0.6))
true_vals_b1_diag = {'B1_3fac(0, 0)': true_params['B1_3fac'][0, 0], 'B1_3fac(1, 1)': true_params['B1_3fac'][1, 1], 'B1_3fac(2, 2)': true_params['B1_3fac'][2, 2], 'B1_1fac(0, 0)': true_params['B1_1fac'][0, 0]}
num_cats = len(y_order_b1_diag)
for param_name, true_val in true_vals_b1_diag.items():
    if param_name in y_order_b1_diag:
        y_pos = y_order_b1_diag.index(param_name)
        ax0.axvline(true_val, ymin=(y_pos + 0.1) / num_cats, ymax=(y_pos + 0.9) / num_cats, color='black', ls='--', lw=1.5)
ax0.set_title('B1 Diagonals')
if ax0.get_legend() is not None: ax0.get_legend().remove()

# プロット2: B1 Off-Diagonals
ax1 = axes[0, 1]
sns.boxplot(data=b1_off_diag_data, y='ParamName', x='Value', hue='Model', ax=ax1, palette=color_palette, hue_order=['FRS', 'VAR_3fac'], order=y_order_b1_off_diag, boxprops=dict(alpha=0.6))
true_vals_b1_off = {'B1_3fac(0, 1)': true_params['B1_3fac'][0, 1], 'B1_3fac(0, 2)': true_params['B1_3fac'][0, 2], 'B1_3fac(1, 0)': true_params['B1_3fac'][1, 0], 'B1_3fac(1, 2)': true_params['B1_3fac'][1, 2], 'B1_3fac(2, 0)': true_params['B1_3fac'][2, 0], 'B1_3fac(2, 1)': true_params['B1_3fac'][2, 1]}
num_cats_off = len(y_order_b1_off_diag)
for param_name, true_val in true_vals_b1_off.items():
    if param_name in y_order_b1_off_diag:
        y_pos = y_order_b1_off_diag.index(param_name)
        ax1.axvline(true_val, ymin=(y_pos + 0.1) / num_cats_off, ymax=(y_pos + 0.9) / num_cats_off, color='black', ls='--', lw=1.5)
ax1.set_title('B1 Off-Diagonals')
if ax1.get_legend() is not None: ax1.get_legend().remove()

# プロット3: Q Diagonals
ax2 = axes[1, 0]
sns.boxplot(data=q_data_combined, y='ParamName', x='Value', hue='Model', ax=ax2, palette=color_palette, hue_order=['FRS', 'VAR_3fac', 'VAR_1fac'], order=y_order_q_diag, boxprops=dict(alpha=0.6))
true_q_vals = {'Q_3fac_diag(0,)': true_params['Q_3fac_diag'][0], 'Q_3fac_diag(1,)': true_params['Q_3fac_diag'][1], 'Q_3fac_diag(2,)': true_params['Q_3fac_diag'][2], 'Q_1fac_diag(0,)': true_params['Q_1fac_diag'][0]}
num_cats_q = len(y_order_q_diag)
for param_name, true_val in true_q_vals.items():
    if param_name in y_order_q_diag:
        y_pos = y_order_q_diag.index(param_name)
        ax2.axvline(true_val, ymin=(y_pos + 0.1) / num_cats_q, ymax=(y_pos + 0.9) / num_cats_q, color='black', ls='--', lw=1.5)
ax2.set_title('Q Diagonals')
if ax2.get_legend() is not None: ax2.get_legend().remove()

# ★ プロット4: b0 Intercepts
ax3 = axes[1, 1]
sns.boxplot(data=b0_data, y='ParamName', x='Value', hue='Model', ax=ax3, palette=color_palette, hue_order=['VAR_3fac', 'VAR_1fac'], order=y_order_b0, boxprops=dict(alpha=0.6))
true_b0_vals = {'b0_3fac(0,)': true_params['b0_3fac'][0], 'b0_3fac(1,)': true_params['b0_3fac'][1], 'b0_3fac(2,)': true_params['b0_3fac'][2], 'b0_1fac(0,)': true_params['b0_1fac'][0]}
num_cats_b0 = len(y_order_b0)
for param_name, true_val in true_b0_vals.items():
    if param_name in y_order_b0:
        y_pos = y_order_b0.index(param_name)
        ax3.axvline(true_val, ymin=(y_pos + 0.1) / num_cats_b0, ymax=(y_pos + 0.9) / num_cats_b0, color='black', ls='--', lw=1.5)
ax3.set_title('b0 Intercepts')
if ax3.get_legend() is not None: ax3.get_legend().remove()

# --- 凡例と保存 (変更なし) ---
all_models_in_plot = pd.concat([b1_diag_combined, b1_off_diag_data, q_data_combined, b0_data])['Model'].unique()
handles = [mpatches.Patch(color=color_palette[model], label=model, alpha=0.6) for model in color_palette if model in all_models_in_plot]
handles.append(Line2D([0], [0], color='black', ls='--', lw=1.5, label='True Value'))
axes[1, 1].legend(handles=handles, loc='upper right') # 凡例を最後のプロットに移動

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(SUMMARY_DIR, 'structural_parameters_comparison.png'))
plt.close(fig)
print("Structural parameter comparison plot saved.")

# --- (The rest of the script for measurement model plots remains unchanged) ---