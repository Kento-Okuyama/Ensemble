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

    def parse_vector(vector_str):
        return np.array([float(v) for v in vector_str.strip().replace('[', '').replace(']', '').replace(',', '').split()])

    # モデルタイプに応じたパース処理
    if model_type == 'VAR':
        try:
            s3_content = content.split("--- 3-Factor Model Estimated vs. True Parameters ---")[1].split("--- 1-Factor Model Estimated vs. True Parameters ---")[0]
            s1_content = content.split("--- 1-Factor Model Estimated vs. True Parameters ---")[1]

            if (m := re.search(r"-- Estimated B1 \(State Transition\) --\s*\n(.*?)\n-- True B1 --", s3_content, re.DOTALL)):
                metrics['B1_3fac'] = parse_matrix(m.group(1))
            if (m := re.search(r"-- Estimated Q diag \(System Noise\): --\s*\n\s*(.*?)\n-- True Q diag --", s3_content, re.DOTALL)):
                metrics['Q_3fac_diag'] = parse_vector(m.group(1))
            if (m := re.search(r"-- Estimated R diag \(Measurement Noise\): --\s*\n\s*(.*?)\n-- True R diag --", s3_content, re.DOTALL)):
                metrics['R_3fac_diag'] = parse_vector(m.group(1))
            if (m := re.search(r"-- Estimated Lambda1 free \(Factor Loading\): --\s*\n\s*(.*?)\n-- True Lambda1 free --", s3_content, re.DOTALL)):
                metrics['lambda_free_3fac'] = parse_vector(m.group(1))
            
            if (m := re.search(r"-- Estimated B1 \(State Transition\): --\s*\n\s*\[\[(.*?)\]\]", s1_content, re.DOTALL)):
                metrics['B1_1fac'] = np.array([[float(m.group(1))]])
            if (m := re.search(r"-- Estimated Q diag \(System Noise\): --\s*\n\s*(.*?)\n-- True Q diag --", s1_content, re.DOTALL)):
                metrics['Q_1fac_diag'] = parse_vector(m.group(1))
            if (m := re.search(r"-- Estimated R diag \(Measurement Noise\): --\s*\n\s*(.*?)\n-- True R diag --", s1_content, re.DOTALL)):
                metrics['R_1fac_diag'] = parse_vector(m.group(1))
            if (m := re.search(r"-- Estimated Lambda1 free \(Factor Loading\): --\s*\n\s*(.*?)\n-- True Lambda1 free --", s1_content, re.DOTALL)):
                metrics['lambda_free_1fac'] = parse_vector(m.group(1))
        except IndexError:
            pass

    elif model_type == 'FRS':
        if (m := re.search(r"-- Estimated B1 \(3-Factor\) --\s*\n(.*?)\n\n-- Estimated B1 \(1-Factor\) --", content, re.DOTALL)):
            metrics['B1_3fac'] = parse_matrix(m.group(1))
        if (m := re.search(r"-- Estimated B1 \(1-Factor\) --\s*\n\s*\[\[(.*?)\]\]", content, re.DOTALL)):
            metrics['B1_1fac'] = np.array([[float(m.group(1))]])
        if (m := re.search(r"-- Estimated Q diag --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            q_vec = parse_vector(m.group(1))
            metrics['Q_3fac_diag'] = q_vec[:3]
            metrics['Q_1fac_diag'] = q_vec[3:]
        if (m := re.search(r"-- Estimated R diag --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            metrics['R_diag'] = parse_vector(m.group(1))
        # ★★★ FRSのLambdaパース処理 (正しい位置) ★★★
        if (m := re.search(r"-- Estimated Lambda free \(State 1\) --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            metrics['lambda_free_3fac'] = parse_vector(m.group(1))
        if (m := re.search(r"-- Estimated Lambda free \(State 2\) --\s*\n\s*(.*?)\n", content, re.DOTALL)):
            metrics['lambda_free_1fac'] = parse_vector(m.group(1))     
            
    return metrics

print("\n--- 全ての処理が完了しました ---")

# --- データ収集メインループ (このブロック全体を置き換え) ---
all_params_data = []
for i in range(1, NUM_REALIZATIONS + 1):
    # FRSログのパース
    frs_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'frs_results_run_{i}', f'frs_log_{DGP_MODE}_run_{i}.txt')
    frs_params = parse_log_file(frs_log_file, 'FRS')
    
    # VARログのパース
    var_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'var_results_run_{i}', f'var_log_{DGP_MODE}_run_{i}.txt')
    var_params = parse_log_file(var_log_file, 'VAR')

    # FRS データを整形
    for p_name, p_val in frs_params.items():
        if isinstance(p_val, (np.ndarray, float)):
            p_val = np.atleast_1d(p_val) # スカラーを配列に変換
            for index, val in np.ndenumerate(p_val):
                all_params_data.append({'Realization': i, 'Model': 'FRS', 'ParamGroup': p_name, 'ParamName': f'{p_name}{index}', 'Value': val})

    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    #  FIX: VARのパラメータを正しく整形し、リストに追加するロジックを修正
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
    # VAR データを整形
    for p_name, p_val in var_params.items():
        if isinstance(p_val, (np.ndarray, float)):
            model_suffix = '_3fac' if '3fac' in p_name else '_1fac'
            # FIX: Use the original parameter name for ParamGroup to ensure consistency with the FRS model.
            param_group = p_name 
            p_val = np.atleast_1d(p_val)
            for index, val in np.ndenumerate(p_val):
                all_params_data.append({'Realization': i, 'Model': f'VAR{model_suffix}', 'ParamGroup': param_group, 'ParamName': f'{p_name}{index}', 'Value': val})

# --- 全パラメータを格納したDataFrameを作成 ---
params_df = pd.DataFrame(all_params_data)

# --- 真値の定義 ---
true_params = {
    'B1_3fac': np.array([[0.04, 0.01, -0.11], [-0.01, 0.07, 0.13], [0.02, 0.11, 0.16]]),
    'B1_1fac': np.array([[0.50]]),
    'Q_3fac_diag': np.array([0.5, 0.5, 0.5]),
    'Q_1fac_diag': np.array([0.1]),
    'R_diag': np.array([0.5] * 9),
    # ★★★ FIX: Lambdaの真値を辞書内に移動 ★★★
    'lambda_free_3fac': [1.2, 0.8, 1.1, 0.9, 1.3, 0.7],
    'lambda_free_1fac': [1.2, 0.8, 1.1, 0.9, 1.3, 0.7, 0.6, 1.0]
}

# --- 色設定 ---
color_palette = {
    'FRS': sns.color_palette("muted")[1],
    'VAR_1fac': sns.color_palette("muted")[3],
    'VAR_3fac': sns.color_palette("muted")[2]
}

# ★★★ FIX: 新しい関数を定義してパレットに透過度を適用 ★★★
def apply_alpha_to_palette(palette, alpha):
    new_palette = {}
    for key, rgb_tuple in palette.items():
        new_palette[key] = (rgb_tuple[0], rgb_tuple[1], rgb_tuple[2], alpha)
    return new_palette

transparent_color_palette = apply_alpha_to_palette(color_palette, 0.6)


# --- プロット作成 ---
sns.set_theme(style="whitegrid")

# 新しい1x3のサブプロットを作成し、横長にする
fig, axes = plt.subplots(1, 3, figsize=(28, 12))
fig.suptitle('Structural Model Parameter Estimates', fontsize=20, y=1.0)

# --- データの前処理：プロットごとにデータを分割 ---
# プロット1: B1 対角成分
b1_diag_data = params_df[
    (params_df['ParamGroup'] == 'B1_3fac') & 
    (params_df['ParamName'].str.contains('3fac\\(0, 0\\)|3fac\\(1, 1\\)|3fac\\(2, 2\\)'))
].copy()
b1_1fac_data = params_df[params_df['ParamGroup'] == 'B1_1fac'].copy()
b1_1fac_data['ParamName'] = 'B1_1fac(0, 0)'
b1_diag_combined = pd.concat([b1_diag_data, b1_1fac_data])

# Y軸ラベルの順序を修正してB1_1facを一番下にする
b1_diag_combined['sort_key'] = b1_diag_combined['ParamName'].apply(lambda x: 0 if '3fac(0, 0)' in x else 1 if '3fac(1, 1)' in x else 2 if '3fac(2, 2)' in x else 3 if '1fac(0, 0)' in x else 9)
b1_diag_combined = b1_diag_combined.sort_values(by='sort_key', ascending=True)
y_order_b1_diag = b1_diag_combined['ParamName'].unique().tolist()


# プロット2: B1 非対角成分
b1_off_diag_data = params_df[
    (params_df['ParamGroup'] == 'B1_3fac') & 
    (~params_df['ParamName'].str.contains('3fac\\(0, 0\\)|3fac\\(1, 1\\)|3fac\\(2, 2\\)'))
].copy()
b1_off_diag_data['sort_key'] = b1_off_diag_data['ParamName'].apply(lambda x: 0 if '3fac(0, 1)' in x else 1 if '3fac(0, 2)' in x else 2 if '3fac(1, 0)' in x else 3 if '3fac(1, 2)' in x else 4 if '3fac(2, 0)' in x else 5 if '3fac(2, 1)' in x else 9)
b1_off_diag_data = b1_off_diag_data.sort_values(by='sort_key', ascending=True)
y_order_b1_off_diag = b1_off_diag_data['ParamName'].unique().tolist()


# プロット3: Q Diagonals
q_data_combined = params_df[(params_df['ParamGroup'] == 'Q_3fac_diag') | (params_df['ParamGroup'] == 'Q_1fac_diag')].copy()
q_data_combined['sort_key'] = q_data_combined['ParamName'].apply(lambda x: 0 if '3fac(0, 0)' in x else 1 if '3fac(0, 1)' in x else 2 if '3fac(0, 2)' in x else 3 if '1fac_diag' in x else 9)
q_data_combined = q_data_combined.sort_values(by='sort_key', ascending=False)
y_order_q_diag = q_data_combined['ParamName'].unique().tolist()

# --- プロットの描画 ---
# プロット1: B1 Diagonals
ax0 = axes[0]
sns.boxplot(data=b1_diag_combined, y='ParamName', x='Value', hue='Model', ax=ax0, palette=color_palette, hue_order=['VAR_3fac', 'VAR_1fac', 'FRS'], order=y_order_b1_diag, boxprops=dict(alpha=0.6))
true_vals = {
    'B1_3fac(0, 0)': true_params['B1_3fac'][0, 0],
    'B1_3fac(1, 1)': true_params['B1_3fac'][1, 1],
    'B1_3fac(2, 2)': true_params['B1_3fac'][2, 2],
    'B1_1fac(0, 0)': true_params['B1_1fac'][0, 0],
}
for i, param_name in enumerate(y_order_b1_diag):
    if param_name in true_vals:
        y_pos = y_order_b1_diag.index(param_name)
        ax0.axvline(true_vals[param_name], ymin=(y_pos + 0.1) / len(y_order_b1_diag), ymax=(y_pos + 0.9) / len(y_order_b1_diag), color='black', ls='--', lw=1)
ax0.set_title('B1 Diagonals')
if ax0.get_legend() is not None:
    ax0.get_legend().remove()

# プロット2: B1 Off-Diagonals
ax1 = axes[1]
sns.boxplot(data=b1_off_diag_data, y='ParamName', x='Value', hue='Model', ax=ax1, palette=color_palette, hue_order=['VAR_3fac', 'FRS'], order=y_order_b1_off_diag, boxprops=dict(alpha=0.6))
true_b1_off_diag_map = {
    'B1_3fac(0, 1)': true_params['B1_3fac'][0, 1],
    'B1_3fac(0, 2)': true_params['B1_3fac'][0, 2],
    'B1_3fac(1, 0)': true_params['B1_3fac'][1, 0],
    'B1_3fac(1, 2)': true_params['B1_3fac'][1, 2],
    'B1_3fac(2, 0)': true_params['B1_3fac'][2, 0],
    'B1_3fac(2, 1)': true_params['B1_3fac'][2, 1],
}
for i, param_name in enumerate(y_order_b1_off_diag):
    if param_name in true_b1_off_diag_map:
        y_pos = y_order_b1_off_diag.index(param_name)
        ax1.axvline(true_b1_off_diag_map[param_name], ymin=(y_pos + 0.1) / len(y_order_b1_off_diag), ymax=(y_pos + 0.9) / len(y_order_b1_off_diag), color='black', ls='--', lw=1)
ax1.set_title('B1 Off-Diagonals')
if ax1.get_legend() is not None:
    ax1.get_legend().remove()

# プロット3: Q Diagonals
ax2 = axes[2]
sns.boxplot(data=q_data_combined, y='ParamName', x='Value', hue='Model', ax=ax2, palette=color_palette, hue_order=['VAR_3fac', 'VAR_1fac', 'FRS'], order=y_order_q_diag, boxprops=dict(alpha=0.6))
true_q_vals = {
    'Q_3fac_diag(0, 0)': true_params['Q_3fac_diag'][0],
    'Q_3fac_diag(0, 1)': true_params['Q_3fac_diag'][1],
    'Q_3fac_diag(0, 2)': true_params['Q_3fac_diag'][2],
    'Q_1fac_diag(0, 0)': true_params['Q_1fac_diag'][0]
}
for i, param_name in enumerate(y_order_q_diag):
    if param_name in true_q_vals:
        y_pos = y_order_q_diag.index(param_name)
        ax2.axvline(true_q_vals[param_name], ymin=(y_pos + 0.1) / len(y_order_q_diag), ymax=(y_pos + 0.9) / len(y_order_q_diag), color='black', ls='--', lw=1)
ax2.set_title('Q Diagonals')
if ax2.get_legend() is not None:
    ax2.get_legend().remove()

# --- 凡例と保存 ---
handles = [mpatches.Patch(color=color_palette[model], label=model, alpha=0.6) for model in color_palette.keys()]
handles.append(Line2D([0], [0], color='black', ls='--', lw=1, label='True Value'))
# 凡例を最後のプロット内に配置
axes[2].legend(handles=handles, loc='upper right')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(SUMMARY_DIR, 'structural_parameters_comparison.png'))
plt.close(fig)
print("Structural parameter comparison plot saved.")

# --- 観測モデルのプロット (このブロック全体を置き換え) ---
fig_meas, axes_meas = plt.subplots(1, 1, figsize=(18, 10))
fig_meas.suptitle('Measurement Model Parameter Estimates', fontsize=20)

# Combine R data from all models into a single DataFrame for easier plotting
r_data_combined = []

# Process R_3fac_diag from VAR
r_3fac_var_data = params_df[(params_df['ParamGroup'] == 'R_3fac_diag') & (params_df['Model'] == 'VAR_3fac')]
for _, row in r_3fac_var_data.iterrows():
    r_data_combined.append({'ParamName': f"R_diag({row['ParamName'].split('(')[1]}", 'Model': row['Model'], 'Value': row['Value']})

# Process R_1fac_diag from VAR
r_1fac_var_data = params_df[(params_df['ParamGroup'] == 'R_1fac_diag') & (params_df['Model'] == 'VAR_1fac')]
for _, row in r_1fac_var_data.iterrows():
    r_data_combined.append({'ParamName': f"R_diag({row['ParamName'].split('(')[1]}", 'Model': row['Model'], 'Value': row['Value']})

# Process R_diag from FRS
r_frs_data = params_df[(params_df['ParamGroup'] == 'R_diag') & (params_df['Model'] == 'FRS')]
for _, row in r_frs_data.iterrows():
    r_data_combined.append({'ParamName': row['ParamName'], 'Model': row['Model'], 'Value': row['Value']})

r_combined_df = pd.DataFrame(r_data_combined)

# Sort the data by ParamName to ensure consistent plotting order
r_combined_df['sort_key'] = r_combined_df['ParamName'].apply(lambda x: int(re.search(r'\d+', x).group()))
r_combined_df = r_combined_df.sort_values(by='sort_key', ascending=True)
r_combined_df = r_combined_df.drop('sort_key', axis=1)

# Plot the combined data
sns.boxplot(data=r_combined_df, y='ParamName', x='Value', hue='Model', ax=axes_meas, palette=color_palette, hue_order=['VAR_3fac', 'VAR_1fac', 'FRS'], boxprops=dict(alpha=0.6))

# Draw true value lines
y_ticks_labels = [label.get_text() for label in axes_meas.get_yticklabels()]
true_r_val = true_params['R_diag'][0]
num_unique_params = len(y_ticks_labels)
num_models = len(r_combined_df['Model'].unique())
box_width = 0.8
spacing = 0.2

for pos, label_text in enumerate(y_ticks_labels):
    # Calculate the position for the single true value line
    y_center = pos
    ax_y_min = (y_center + 0.1) / num_unique_params
    ax_y_max = (y_center + 0.9) / num_unique_params
    axes_meas.axvline(true_r_val, ymin=ax_y_min, ymax=ax_y_max, color='black', ls='--', lw=1)

# Create legend
handles = [mpatches.Patch(color=color_palette[model], label=model, alpha=0.6) for model in color_palette.keys() if model in r_combined_df['Model'].unique()]
handles.append(Line2D([0], [0], color='black', ls='--', lw=1, label='True Value'))
axes_meas.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0))

# Adjust layout to fit everything without overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(SUMMARY_DIR, 'measurement_parameters_comparison.png'))
plt.close(fig_meas)
print("Measurement parameter comparison plot saved.")
print("\n--- 全ての処理が完了しました ---")

# NEW: Lambdaパラメータのプロットを追加
fig_lambda, axes_lambda = plt.subplots(1, 2, figsize=(20, 12))
fig_lambda.suptitle('Measurement Model: Lambda (Factor Loadings) Estimates', fontsize=20, y=1.0)

# Lambda (3-Factor / State 1)
lambda_3fac_data = params_df[params_df['ParamGroup'] == 'lambda_free_3fac']
if not lambda_3fac_data.empty:
    ax = axes_lambda[0]
    sns.boxplot(data=lambda_3fac_data, y='ParamName', x='Value', hue='Model', ax=ax, palette=color_palette, hue_order=['VAR_3fac', 'FRS'], boxprops=dict(alpha=0.6))
    true_vals = true_params['lambda_free_3fac']
    num_items = len(true_vals)
    for i in range(num_items):
        inverted_pos = num_items - 1 - i
        ax.axvline(true_vals[i], ymin=(inverted_pos + 0.1) / num_items, ymax=(inverted_pos + 0.9) / num_items, color='black', ls='--', lw=1)
    ax.set_title('Lambda Free Parameters (3-Factor / State 1)')
    if ax.get_legend() is not None: ax.get_legend().remove()

# Lambda (1-Factor / State 2)
lambda_1fac_data = params_df[params_df['ParamGroup'] == 'lambda_free_1fac']
if not lambda_1fac_data.empty:
    ax = axes_lambda[1]
    sns.boxplot(data=lambda_1fac_data, y='ParamName', x='Value', hue='Model', ax=ax, palette=color_palette, hue_order=['VAR_1fac', 'FRS'], boxprops=dict(alpha=0.6))
    true_vals = true_params['lambda_free_1fac']
    num_items = len(true_vals)
    for i in range(num_items):
        inverted_pos = num_items - 1 - i
        ax.axvline(true_vals[i], ymin=(inverted_pos + 0.1) / num_items, ymax=(inverted_pos + 0.9) / num_items, color='black', ls='--', lw=1)
    ax.set_title('Lambda Free Parameters (1-Factor / State 2)')
    if ax.get_legend() is not None: ax.get_legend().remove()

# FIX: Common Legendをfig全体ではなく、右側のaxesに追加する
# fig_lambda.legendの代わりにaxes_lambda[1].legendを使用
# axes_lambda[1]のget_legend()を削除して、凡例が作成されるようにする
all_lambda_models = pd.concat([lambda_3fac_data, lambda_1fac_data])['Model'].unique()
handles = [mpatches.Patch(color=color_palette[model], label=model, alpha=0.6) for model in color_palette.keys() if model in all_lambda_models]
handles.append(Line2D([0], [0], color='black', ls='--', lw=1, label='True Value'))

# FIX: legendを右側のaxesに直接追加し、位置を調整
axes_lambda[1].legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), borderaxespad=0.)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(SUMMARY_DIR, 'lambda_parameters_comparison.png'))
plt.close(fig_lambda)
print("Lambda parameter comparison plot saved.")