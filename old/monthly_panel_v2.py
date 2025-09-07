import pandas as pd
import os
from scipy.stats import entropy
from tqdm import tqdm
import numpy as np

# ==============================================================================
# --- 1. 配置区域 ---
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
    print("--- RQ3 分析: 构建月度面板数据集 (V5 - 活跃度版) ---")

    # --- 1. 加载所有需要的数据 ---
    print("[*] 正在加载所需的数据文件...")
    try:
        repos_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "repos.csv"))
        prs_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "prs.csv"))
        issues_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "issues.csv"))
        stars_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "stars.csv"))
        # --- 新增：加载comments.csv用于计算活跃度 ---
        comments_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "comments.csv"))
        classified_repos_df = pd.read_csv(os.path.join(CLASSIFICATION_DIR, "repos_classified.csv"))
    except FileNotFoundError as e:
        print(f"[!] 错误: 缺少数据文件: {e}")
        return
    print("[✓] 所有数据加载成功。")

    # --- 2. 数据预处理 ---
    print("[*] 正在预处理数据...")
    repos_df = pd.merge(repos_df, classified_repos_df[['repo_name', 'primary_role']], on='repo_name', how='left')

    def convert_to_datetime_and_naive(df, column_name):
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        df.dropna(subset=[column_name], inplace=True)
        if df[column_name].dt.tz is not None:
            df[column_name] = df[column_name].dt.tz_localize(None)
        return df

    repos_df = convert_to_datetime_and_naive(repos_df, 'created_at')
    prs_df = convert_to_datetime_and_naive(prs_df, 'created_at')
    # --- 新增：处理prs_df的merged_at时间戳 ---
    prs_df = convert_to_datetime_and_naive(prs_df, 'merged_at')
    issues_df = convert_to_datetime_and_naive(issues_df, 'created_at')
    stars_df = convert_to_datetime_and_naive(stars_df, 'starred_at')
    # --- 新增：处理comments_df的时间戳 ---
    comments_df = convert_to_datetime_and_naive(comments_df, 'created_at')
    print("[✓] 所有时间列已成功转换为统一的 timezone-naive 格式。")

    # --- 3. 初始化面板DataFrame ---
    date_index = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')
    panel_df = pd.DataFrame(index=date_index)

    # --- 4. 计算生态健康度指标 (Y 变量) ---
    print("[*] 正在计算月度生态健康度指标 (Y)...")

    # a. 吸引力 (Attractiveness) - 保持不变
    panel_df['attract_stars_growth'] = stars_df.set_index('starred_at').resample('MS').size()
    panel_df['attract_new_app_repo_count'] = repos_df[repos_df['primary_role'] == 'Application'].set_index(
        'created_at').resample('MS').size()

    # --- 修改：计算 b. 活跃度 (Activeness) ---
    # 计算每月合并的PR数
    monthly_merged_prs = prs_df.set_index('merged_at').resample('MS').size()
    # 计算每月Issue下的评论数
    monthly_issue_comments = comments_df[comments_df['parent_type'] == 'issue'].set_index('created_at').resample(
        'MS').size()
    # 两者相加得到总活跃度指标
    panel_df['activeness_total_events'] = monthly_merged_prs.add(monthly_issue_comments, fill_value=0)

    # c. 创新性 (Innovativeness) - 保持不变
    monthly_diversity = []
    repos_by_month = repos_df.set_index('created_at').groupby(pd.Grouper(freq='MS'))
    for month, group in repos_by_month:
        if group.empty: continue
        topics = group['topics'].dropna().str.split(',').explode()
        if topics.empty: continue
        monthly_diversity.append(entropy(topics.value_counts(), base=2))
    diversity_series = pd.Series(monthly_diversity, index=repos_by_month.groups.keys()).reindex(date_index,
                                                                                                fill_value=0)
    panel_df['innovate_topic_diversity'] = diversity_series

    # --- 5. 计算创新机制指标 (X 变量) - 保持不变 ---
    print("[*] 正在计算月度创新机制指标 (X)...")
    panel_df['mech_app_creation'] = panel_df['attract_new_app_repo_count']
    panel_df['mech_code_contrib'] = prs_df[prs_df['contribution_type'] == 'code'].set_index('created_at').resample(
        'MS').size()
    panel_df['mech_problem_solving'] = issues_df.set_index('created_at').resample('MS').size()
    panel_df['mech_knowledge_sharing'] = prs_df[prs_df['contribution_type'] == 'doc'].set_index('created_at').resample(
        'MS').size()

    # --- 6. 清理并保存 ---
    print("[*] 正在清理并保存最终的面板数据...")
    panel_df.fillna(0, inplace=True)
    panel_df.replace([np.inf, -np.inf], 0, inplace=True)
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, PANEL_DATA_FILENAME)
    panel_df.to_csv(output_path)

    print(f"\n--- 面板数据构建完成 ---")
    print(f"[*] 最终的面板数据已保存至: '{output_path}'")
    print("\n--- 数据预览 (前5行) ---")
    print(panel_df.head())


if __name__ == "__main__":
    main()