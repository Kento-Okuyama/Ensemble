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
DGP_MODE = 'IMPLEMENT 2.0'
NUM_REALIZATIONS = 30
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# 修正箇所: dataとresuldf_melted = pd.concatsのベースディレクトリを別々に定義
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
BASE_DATA_DIR = 'data'
BASE_RESULTS_DIR = 'results'
MODE_DIR = DGP_MODE.replace(' 2.0', '_2.0')
SUMMARY_DIR = os.path.join(BASE_RESULTS_DIR, MODE_DIR, 'summary') # サマリーはresults内に保存
os.makedirs(SUMMARY_DIR, exist_ok=True)

print(f"--- 結果の集計と可視化を開始します ---")
print(f"モード: {DGP_MODE}")
print(f"リアライゼーション数: {NUM_REALIZATIONS}")
print(f"結果の保存先: {SUMMARY_DIR}")

# --- データ収集用の関数 (変更なし) ---
def parse_log_file(log_path, model_type):
    # ... (この関数は変更ありません) ...
    metrics = {}
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return metrics
    patterns = {
        'Likelihood': r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)",
        'RMSE': r"Y Prediction RMSE \(Median\)\s*\|\s*(\d+\.\d+)" if model_type == 'BPS' else r"Y Prediction RMSE\s*\|\s*(\d+\.\d+)",
        'Sensitivity': r"State Detection Sensitivity\s*\|\s*(\d+\.\d+)",
        'Specificity': r"State Detection Specificity\s*\|\s*(\d+\.\d+)",
        'Duration': r"Training Duration \(s\)\s*\|\s*(\d+\.\d+)"
    }
    if model_type == 'BPS':
        if (m := re.search(r"Y Prediction RMSE \(Mode\)\s*\|\s*(\d+\.\d+)", content)):
            metrics['RMSE_Mode'] = float(m.group(1))
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    if model_type in ['FRS', 'BPS']:
        if model_type == 'FRS':
            if (m := re.search(r"gamma_intercept:.*?Estimated = (-?\d+\.\d+)", content)):
                metrics['gamma_intercept'] = float(m.group(1)) 
            if (m := re.search(r"gamma_task:.*?Estimated = (-?\d+\.\d+)", content, re.DOTALL)):
                metrics['gamma_task'] = float(m.group(1))
            if (m := re.search(r"gamma_goal:.*?Estimated = (-?\d+\.\d+)", content, re.DOTALL)):
                metrics['gamma_goal'] = float(m.group(1))
            if (m := re.search(r"gamma_bond:.*?Estimated = (-?\d+\.\d+)", content, re.DOTALL)):
                metrics['gamma_bond'] = float(m.group(1))
        else: # BPS
            if (m := re.search(r"gamma_intercept:.*?Estimated = (-?\d+\.\d+)", content)):
                metrics['gamma_intercept'] = float(m.group(1))
            if (m := re.search(r"gamma_task:.*?Estimated = (-?\d+\.\d+)", content, re.DOTALL)):
                metrics['gamma_task'] = float(m.group(1))
            if (m := re.search(r"gamma_goal:.*?Estimated = (-?\d+\.\d+)", content, re.DOTALL)):
                metrics['gamma_goal'] = float(m.group(1))
            if (m := re.search(r"gamma_bond:.*?Estimated = (-?\d+\.\d+)", content, re.DOTALL)):
                metrics['gamma_bond'] = float(m.group(1))
    return metrics

# --- データ収集メインループ ---
all_data = []
dgp_data = {
    'switch_proportions_over_time': [],
    'overall_regime_proportions_s2': [],
    'y_trajectories': []
}

print("\n--- 1. 全リアライゼーションのデータを収集中... ---")
for i in range(1, NUM_REALIZATIONS + 1):
    realization_data = {'realization': i}
    
    # ★★★ 修正: DGPデータの読み込みに BASE_DATA_DIR を使用 ★★★
    dgp_data_file = os.path.join(BASE_DATA_DIR, MODE_DIR, f'dgp_data_run_{i}', f'simulation_data_{DGP_MODE}.pt')
    if os.path.exists(dgp_data_file):
        data = torch.load(dgp_data_file, map_location='cpu', weights_only=False)
        actual_states = data['actual_states']
        dgp_data['switch_proportions_over_time'].append((actual_states == 2).mean(axis=0))
        dgp_data['overall_regime_proportions_s2'].append((actual_states == 2).mean())
        dgp_data['y_trajectories'].append(data['Y_generated'])
    else:
        print(f"警告: DGPデータファイルが見つかりません: {dgp_data_file}")

    # ★★★ 修正: VAR, FRS, BPSログの読み込みに BASE_RESULTS_DIR を使用 ★★★
    var_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'var_results_run_{i}', f'var_log_{DGP_MODE}_run_{i}.txt')
    if os.path.exists(var_log_file):
        # ... (VARのログ解析ロジックは変更なし) ...
        with open(var_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                s3_content = content.split("--- 3-Factor Model Performance Summary ---")[1].split("--- 1-Factor Model Performance Summary ---")[0]
                s1_content = content.split("--- 1-Factor Model Performance Summary ---")[1]
                if (m := re.search(r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)", s3_content)):
                    realization_data['VAR_3fac_Likelihood'] = float(m.group(1))
                if (m := re.search(r"Y Prediction RMSE\s*\|\s*(\d+\.\d+)", s3_content)):
                    realization_data['VAR_3fac_RMSE'] = float(m.group(1))
                if (m := re.search(r"Training Duration \(s\)\s*\|\s*(\d+\.\d+)", s3_content)):
                    realization_data['VAR_3fac_Duration'] = float(m.group(1))
                if (m := re.search(r"Final Log-Likelihood\s*\|\s*(-?\d+\.\d+)", s1_content)):
                    realization_data['VAR_1fac_Likelihood'] = float(m.group(1))
                if (m := re.search(r"Y Prediction RMSE\s*\|\s*(\d+\.\d+)", s1_content)):
                    realization_data['VAR_1fac_RMSE'] = float(m.group(1))
                if (m := re.search(r"Training Duration \(s\)\s*\|\s*(\d+\.\d+)", s1_content)):
                    realization_data['VAR_1fac_Duration'] = float(m.group(1))
            except (IndexError, AttributeError):
                print(f"警告: VARログの解析に失敗しました: {var_log_file}")

    frs_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'frs_results_run_{i}', f'frs_log_{DGP_MODE}_run_{i}.txt')
    frs_metrics = parse_log_file(frs_log_file, 'FRS')
    for k, v in frs_metrics.items():
        realization_data[f'FRS_{k}'] = v

    bps_log_file = os.path.join(BASE_RESULTS_DIR, MODE_DIR, f'bps_results_run_{i}', f'bps_log_{DGP_MODE}_run_{i}.txt')
    bps_metrics = parse_log_file(bps_log_file, 'BPS')
    for k, v in bps_metrics.items():
        realization_data[f'BPS_{k}'] = v
            
    all_data.append(realization_data)

df = pd.DataFrame(all_data)
df.set_index('realization', inplace=True)
print("データ収集完了。")

print("\n--- 2. DGPデータの分析と可視化中... ---")
sns.set_theme(style="whitegrid")

def get_summary_df(series, name=None):
    summary = series.describe(percentiles=[.25, .5, .75]).to_frame().T
    try:
        mode_val = series.mode().iloc[0]
    except IndexError:
        mode_val = np.nan
    summary['mode'] = mode_val
    summary = summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'mode']]
    if name:
        summary.index = [name]
    return summary

def save_summary_to_file(summary_dict, filename):
    with open(os.path.join(SUMMARY_DIR, filename), 'w', encoding='utf-8') as f:
        for title, df_summary in summary_dict.items():
            f.write(f"--- {title} ---\n")
            f.write(df_summary.to_string(index=True))
            f.write("\n\n")

dgp_summaries = {}
if dgp_data['overall_regime_proportions_s2']:
    s_regime_s2 = pd.Series(dgp_data['overall_regime_proportions_s2'])
    s_regime_s1 = 1 - s_regime_s2
    dgp_summaries['Overall Proportion of State 1 (3-fac)'] = get_summary_df(s_regime_s1)
    dgp_summaries['Overall Proportion of State 2 (1-fac)'] = get_summary_df(s_regime_s2, name="Proportion")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('DGP: Distribution of Overall State 2 (1-fac) Proportion Across Realizations', fontsize=16)
    sns.histplot(s_regime_s2, kde=True, ax=axes[0]).set_title('Histogram')
    sns.violinplot(y=s_regime_s2, ax=axes[1]).set_title('Violin Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(SUMMARY_DIR, 'dgp_overall_regime_proportion_dist.png'))
    plt.close()

if dgp_data['switch_proportions_over_time']:
    switch_props = np.array(dgp_data['switch_proportions_over_time'])
    mean_switch_props, std_switch_props = switch_props.mean(axis=0), switch_props.std(axis=0)
    time_points = np.arange(mean_switch_props.shape[0])
    plt.figure(figsize=(12, 7))
    plt.plot(time_points, mean_switch_props, 'o-', color='crimson', label='Mean Proportion in State 2')
    plt.fill_between(time_points, mean_switch_props - std_switch_props, mean_switch_props + std_switch_props, color='crimson', alpha=0.2, label='Mean ± 1 SD')
    plt.title(f'DGP: Proportion of Individuals in State 2 (1-Factor) Over Time\n(Mean over {NUM_REALIZATIONS} realizations)', fontsize=16)
    plt.xlabel('Time Point'); plt.ylabel('Proportion'); plt.ylim(0, 1); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, 'dgp_switch_proportion_over_time.png'))
    plt.close()

if dgp_data['y_trajectories']:
    y_tensor = torch.stack(dgp_data['y_trajectories'])
    y_mean_over_n = y_tensor.mean(dim=1)
    y_mean, y_std = y_mean_over_n.mean(dim=0).numpy(), y_mean_over_n.std(dim=0).numpy()
    time_points_y = np.arange(y_mean.shape[0])
    num_observed_vars = y_mean.shape[1]

    for o in range(num_observed_vars):
        plt.figure(figsize=(12, 6))
        plt.plot(time_points_y, y_mean[:, o], 'o-', label=f'Mean of Y[{o}]')
        plt.fill_between(time_points_y, y_mean[:, o] - y_std[:, o], y_mean[:, o] + y_std[:, o], alpha=0.2, label='Mean ± 1 SD')
        plt.title(f'DGP: Mean Trajectory of Observed Variable Y[{o}]\n(Mean over {NUM_REALIZATIONS} realizations)', fontsize=16)
        plt.xlabel('Time Point'); plt.ylabel('Value'); plt.legend(); plt.grid(True, which='both', linestyle='--', linewidth=0.5); plt.tight_layout()
        save_filename = os.path.join(SUMMARY_DIR, f'dgp_y_trajectory_Y{o}.png')
        plt.savefig(save_filename)
        plt.close()
    
    indices_to_plot = [0, 3, 6]
    factor_labels = ['Task Factor (Y[0])', 'Goal Factor (Y[3])', 'Bond Factor (Y[6])']

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
    fig.suptitle('DGP: Mean Trajectories of Factor Indicator Variables', fontsize=18)

    for i, o in enumerate(indices_to_plot):
        ax = axes[i]
        ax.plot(time_points_y, y_mean[:, o], 'o-', label=f'Mean of Y[{o}]')
        ax.fill_between(time_points_y, y_mean[:, o] - y_std[:, o], y_mean[:, o] + y_std[:, o], alpha=0.2)
        ax.set_title(factor_labels[i])
        ax.set_xlabel('Time Point')
        if i == 0:
            ax.set_ylabel('Value')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_filename = os.path.join(SUMMARY_DIR, 'dgp_y_trajectory_factor_indicators.png')
    plt.savefig(save_filename)
    plt.close()
    
    print("Factor indicator plot (Y[0], Y[3], Y[6]) saved.")

print("DGPデータの分析完了。")


print("\n--- 3. モデル比較の分析と可視化中... ---")
metrics_to_compare = ['Likelihood', 'RMSE', 'Sensitivity', 'Specificity', 'Duration']
models = ['VAR_3fac', 'VAR_1fac', 'FRS', 'BPS']
df_list = []
for metric in metrics_to_compare:
    for model in models:
        col_name = f'{model}_{metric}'
        if col_name in df.columns:
            temp_df = df[[col_name]].copy()
            temp_df.rename(columns={col_name: 'value'}, inplace=True)
            temp_df['Model'] = model
            temp_df['Metric'] = metric
            df_list.append(temp_df.reset_index())

df_melted = pd.concat(df_list, ignore_index=True)
model_summaries = {}
for metric in metrics_to_compare:
    metric_data = df_melted[df_melted['Metric'] == metric]
    if not metric_data.empty:
        summary_by_model = metric_data.groupby('Model')['value'].apply(get_summary_df).reset_index(level=1, drop=True)
        model_summaries[f"Metric: {metric}"] = summary_by_model
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Distribution of {metric} Across Models', fontsize=16)
        order = sorted(metric_data['Model'].unique())

        default_colors = sns.color_palette(n_colors=len(order))
        color_map = dict(zip(order, default_colors))
        muted_colors = sns.color_palette("muted", n_colors=len(order))
        muted_color_map = dict(zip(order, muted_colors))


        sns.histplot(data=metric_data, x='value', hue='Model', hue_order=order, kde=True, ax=axes[0], multiple="layer", palette=color_map).set(title='Histogram', xlabel=metric)
        sns.boxplot(data=metric_data, y='value', x='Model', hue='Model', order=order, ax=axes[1], legend=False, palette=muted_color_map).set(title='Boxplot', ylabel=metric)
        axes[1].tick_params(axis='x', rotation=15)
        sns.violinplot(data=metric_data, y='value', x='Model', hue='Model', order=order, ax=axes[2], legend=False, palette=muted_color_map).set(title='Violin Plot', ylabel=metric)
        axes[2].tick_params(axis='x', rotation=15)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(SUMMARY_DIR, f'model_comparison_{metric}.png'))
        plt.close()

# 'gamma'パラメータの比較設定
params_to_compare = ['gamma_intercept', 'gamma_task', 'gamma_goal', 'gamma_bond']
# BPSの代わりにVARモデルを使用
param_models = ['FRS', 'BPS']
true_params = {'gamma_intercept': -2.0, 'gamma_task': 0.5, 'gamma_goal': 0.5, 'gamma_bond': 0.5}

# データフレームを結合
param_df_list = []
for param in params_to_compare:
    for model in param_models:
        col_name = f'{model}_{param}'
        if col_name in df.columns:
            temp_df = df[[col_name]].copy()
            temp_df.rename(columns={col_name: 'value'}, inplace=True)
            temp_df['Model'] = model
            temp_df['Parameter'] = param
            param_df_list.append(temp_df.reset_index())

if param_df_list:
    param_df_melted = pd.concat(param_df_list, ignore_index=True)
else:
    param_df_melted = pd.DataFrame()

print("モデル比較の分析完了。")

print("\n--- 4. サマリー統計をファイルに保存中... ---")
all_summaries = {**dgp_summaries, **model_summaries}
save_summary_to_file(all_summaries, 'summary_statistics.txt')
print(f"サマリー統計が '{SUMMARY_DIR}/summary_statistics.txt' に保存されました。")

print("\n--- 5. 追加の結合プロットを作成中... ---")

order_all = sorted(df_melted['Model'].unique())
color_map_all = dict(zip(order_all, sns.color_palette(n_colors=len(order_all))))

if 'Likelihood' in df_melted['Metric'].unique() and 'RMSE' in df_melted['Metric'].unique():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Likelihood and RMSE Distributions', fontsize=18)

    sns.histplot(data=df_melted[df_melted['Metric'] == 'Likelihood'], x='value', hue='Model', hue_order=order_all, kde=True, ax=axes[0], multiple="layer", palette=color_map_all)
    axes[0].set_title('Likelihood')
    
    sns.histplot(data=df_melted[df_melted['Metric'] == 'RMSE'], x='value', hue='Model', hue_order=order_all, kde=True, ax=axes[1], multiple="layer", palette=color_map_all)
    axes[1].set_title('RMSE')
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(SUMMARY_DIR, 'combined_plot_likelihood_rmse.png'))
    plt.close()

if 'Sensitivity' in df_melted['Metric'].unique() and 'Specificity' in df_melted['Metric'].unique():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle('Sensitivity and Specificity Distributions', fontsize=18)
    
    sens_spec_order = sorted(df_melted[df_melted['Metric'].isin(['Sensitivity', 'Specificity'])]['Model'].unique())
    sens_spec_color_map = {model: color_map_all.get(model) for model in sens_spec_order}

    sns.histplot(data=df_melted[df_melted['Metric'] == 'Sensitivity'], x='value', hue='Model', hue_order=sens_spec_order, kde=True, ax=axes[0], multiple="layer", palette=sens_spec_color_map)
    axes[0].set_title('Sensitivity')

    sns.histplot(data=df_melted[df_melted['Metric'] == 'Specificity'], x='value', hue='Model', hue_order=sens_spec_order, kde=True, ax=axes[1], multiple="layer", palette=sens_spec_color_map)
    axes[1].set_title('Specificity')
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(SUMMARY_DIR, 'combined_plot_sens_spec.png'))
    plt.close()

# 凡例表示に必要なライブラリをスクリプトの先頭でインポートしてください
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D


if 'param_df_melted' in locals() and not param_df_melted.empty:
    
    # ここから元のコード
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Gamma Parameter Distributions (FRS vs. VAR)', fontsize=18)
    axes = axes.flatten()
    
    param_order = sorted(param_df_melted['Model'].unique())
    param_colors = sns.color_palette("muted", n_colors=len(param_order))
    param_color_map = dict(zip(param_order, param_colors))

    for i, param in enumerate(params_to_compare):
        ax = axes[i]
        param_data = param_df_melted[param_df_melted['Parameter'] == param]
        
        sns.histplot(data=param_data, x='value', hue='Model', hue_order=param_order, kde=True, ax=ax, multiple="layer", palette=param_color_map, legend=False)
        ax.axvline(true_params[param], color='r', linestyle='--')
        ax.set_title(param)

        # 凡例の手動作成（最終版）
        true_val_handle = Line2D([0], [0], color='r', linestyle='--', label=f'True Value ({true_params[param]})')

        if i == 0:
            model_handles = [mpatches.Patch(color=param_color_map[model], label=model, alpha=0.6) for model in param_order]
            ax.legend(handles=model_handles + [true_val_handle])
        else:
            ax.legend(handles=[true_val_handle])
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(SUMMARY_DIR, 'combined_plot_gamma_params.png'))
    plt.close()

else:
    print("警告: Gammaパラメータのデータが見つかりませんでした。")

print("追加の結合プロット作成完了。")

print("\n--- 全ての処理が完了しました ---")
