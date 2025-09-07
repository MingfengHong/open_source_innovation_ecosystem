import pandas as pd
import os
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np

# ==============================================================================
# --- 1. 配置区域 (保持不变) ---
# ==============================================================================
FINAL_DATA_DIR = "final_analysis_data"
ANALYSIS_OUTPUT_DIR = "analysis_output"
CLASSIFICATION_DIR = "classification_output"
PANEL_DATA_FILENAME = "monthly_panel_data.csv"
START_DATE = "2022-11-01"
END_DATE = "2024-12-31"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责构建用于最终分析的月度面板数据。
    """
    print("--- RQ3 分析: 步骤 1 - 构建月度面板数据集 (V4 - 最终修复版) ---")

    # --- 1. 加载所有需要的数据 ---
    print("[*] 正在加载所需的数据文件...")
    try:
        repos_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "repos.csv"))
        prs_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "prs.csv"))
        issues_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "issues.csv"))
        stars_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "stars.csv"))
        classified_repos_df = pd.read_csv(os.path.join(CLASSIFICATION_DIR, "repos_classified.csv"))
    except FileNotFoundError as e:
        print(f"[!] 错误: 缺少数据文件: {e}")
        return
    print("[✓] 所有数据加载成功。")

    # --- 2. 数据预处理 ---
    print("[*] 正在预处理数据...")
    repos_df = pd.merge(repos_df, classified_repos_df[['repo_name', 'primary_role']], on='repo_name', how='left')
    print("[✓] 已成功将仓库角色信息合并。")

    # --- 关键修复：更新辅助函数，移除时区信息 ---
    def convert_to_datetime_and_naive(df, column_name):
        # 转换为datetime对象
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        # 删除转换失败的行
        df.dropna(subset=[column_name], inplace=True)
        # 关键修复：如果存在时区信息，则将其移除，统一为naive时间
        if df[column_name].dt.tz is not None:
            df[column_name] = df[column_name].dt.tz_localize(None)
        return df

    repos_df = convert_to_datetime_and_naive(repos_df, 'created_at')
    prs_df = convert_to_datetime_and_naive(prs_df, 'created_at')
    issues_df = convert_to_datetime_and_naive(issues_df, 'created_at')
    issues_df = convert_to_datetime_and_naive(issues_df, 'closed_at')
    stars_df = convert_to_datetime_and_naive(stars_df, 'starred_at')
    print("[✓] 所有时间列已成功转换为统一的 timezone-naive 格式。")

    # --- 3. 初始化面板DataFrame ---
    date_index = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    panel_df = pd.DataFrame(index=date_index)

    # --- 4. 计算指标 (后续逻辑保持不变) ---
    print("[*] 正在计算月度指标...")
    panel_df['attract_stars_growth'] = stars_df.set_index('starred_at').resample('MS').size()
    panel_df['attract_new_app_repo_count'] = repos_df[repos_df['primary_role'] == 'Application'].set_index(
        'created_at').resample('MS').size()
    monthly_issues_created = issues_df.set_index('created_at').resample('MS').size()
    monthly_issues_closed = issues_df.set_index('closed_at').resample('MS').size()
    panel_df['robust_issue_closure_rate'] = monthly_issues_closed / monthly_issues_created
    bug_issues_df = issues_df[issues_df['labels'].str.contains('bug', na=False, case=False)]
    monthly_bugs_created = bug_issues_df.set_index('created_at').resample('MS').size()
    panel_df['robust_bug_report_ratio'] = monthly_bugs_created / monthly_issues_created
    monthly_diversity = []
    repos_by_month = repos_df.set_index('created_at').groupby(pd.Grouper(freq='MS'))
    for month, group in tqdm(repos_by_month, desc="计算主题多样性"):
        if group.empty: continue
        topics = group['topics'].dropna().str.split(',').explode()
        if topics.empty: continue
        topic_counts = topics.value_counts()
        monthly_diversity.append(entropy(topic_counts, base=2))  # 使用base=2更标准
    diversity_series = pd.Series(monthly_diversity, index=repos_by_month.groups.keys()).reindex(date_index,
                                                                                                fill_value=0)
    panel_df['innovate_topic_diversity'] = diversity_series
    panel_df['mech_app_creation'] = panel_df['attract_new_app_repo_count']
    panel_df['mech_code_contrib'] = prs_df[prs_df['contribution_type'] == 'code'].set_index('created_at').resample(
        'MS').size()
    panel_df['mech_problem_solving'] = monthly_issues_created
    panel_df['mech_knowledge_sharing'] = prs_df[prs_df['contribution_type'] == 'doc'].set_index('created_at').resample(
        'MS').size()

    # --- 5. 清理并保存 ---
    print("[*] 正在清理并保存最终的面板数据...")
    panel_df.fillna(0, inplace=True)
    panel_df.replace([np.inf, -np.inf], 0, inplace=True)
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, PANEL_DATA_FILENAME)
    panel_df.to_csv(output_path)

    print(f"\n--- 面板数据构建完成 ---")
    print(f"[*] 最终的面板数据已保存至: '{output_path}'")
    print("\n--- 数据预览 (前5行) ---")
    print(panel_df.head())
    print("\n--- 数据统计描述 ---")
    print(panel_df.describe())


if __name__ == "__main__":
    main()