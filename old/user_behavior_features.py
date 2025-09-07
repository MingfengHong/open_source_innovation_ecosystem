import pandas as pd
import os
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录
# 包含最终筛选后、用于分析的数据
FINAL_DATA_DIR = "final_analysis_data"
# 包含原始ETL产出的目录 (我们需要从中获取完整的用户列表和仓库列表)
ETL_DIR = "etl_output"
# 包含仓库角色分类结果的目录
CLASSIFICATION_DIR = "classification_output"

# 输出目录
# 用于存放本次分析产出的用户特征文件
ANALYSIS_OUTPUT_DIR = "analysis_output"
OUTPUT_FILENAME = "user_behavior_features.csv"

# LangChain核心仓库的全名
CORE_REPO_NAME = "langchain-ai/langchain"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责为每个用户计算行为特征向量。
    """
    print("--- RQ2 分析启动: 步骤 1 - 用户行为特征工程 ---")
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    # --- 1. 加载所有需要的数据 ---
    print("[*] 正在加载所需的数据文件...")
    try:
        # 主文件，包含所有参与分析的活跃实体
        repos_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "repos.csv"))
        prs_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "prs.csv"))
        comments_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "comments.csv"))
        stars_df = pd.read_csv(os.path.join(FINAL_DATA_DIR, "stars.csv"))

        # 辅助文件
        # 完整的用户列表，用于ID和login name的映射
        all_users_df = pd.read_csv(os.path.join(ETL_DIR, "users.csv"))
        # 完整的仓库列表，用于查找核心仓库的ID
        all_repos_df = pd.read_csv(os.path.join(ETL_DIR, "repos.csv"))
        # 仓库的角色分类结果
        classified_repos_df = pd.read_csv(os.path.join(CLASSIFICATION_DIR, "repos_classified.csv"))

    except FileNotFoundError as e:
        print(f"[!] 错误: 缺少数据文件: {e}")
        return
    print("[✓] 所有数据加载成功。")

    # --- 2. 数据预处理与准备 ---
    # a. 将仓库角色信息合并到我们的核心仓库列表中
    repos_with_roles_df = pd.merge(repos_df, classified_repos_df[['repo_name', 'primary_role']], on='repo_name',
                                   how='left')

    # b. 从 repo_name 中解析出 owner 的 login name
    repos_with_roles_df['owner_login'] = repos_with_roles_df['repo_name'].str.split('/').str[0]

    # c. 创建一个从 login name 到 user_id 的映射字典，用于快速查找
    login_to_id_map = all_users_df.set_index('login')['user_id']

    # d. 将 owner_login 映射为 owner_id
    repos_with_roles_df['owner_id'] = repos_with_roles_df['owner_login'].map(login_to_id_map)

    # e. 查找LangChain核心仓库的repo_id
    core_repo_id = all_repos_df[all_repos_df['repo_name'] == CORE_REPO_NAME]['repo_id'].iloc[0]

    print(f"[*] 预处理完成。核心仓库 '{CORE_REPO_NAME}' 的ID为: {core_repo_id}")

    # --- 3. 计算各项行为特征 ---
    # 我们将使用 pandas 的 groupby 和 value_counts 功能进行高效计算
    print("[*] 正在计算各项用户行为特征...")

    # 特征1: 创建应用倾向 (create_app_count)
    app_creators = repos_with_roles_df[repos_with_roles_df['primary_role'] == 'Application'][
        'owner_id'].value_counts().rename('create_app_count')

    # 特征2: 创建工具倾向 (create_tool_count)
    tool_creators = \
    repos_with_roles_df[repos_with_roles_df['primary_role'].isin(['Infrastructure/Tool', 'Library/Plugin'])][
        'owner_id'].value_counts().rename('create_tool_count')

    # 特征3: 核心贡献度 (core_contrib_count)
    core_contributors = prs_df[prs_df['repo_id'] == core_repo_id]['author_id'].value_counts().rename(
        'core_contrib_count')

    # 特征4: 文档贡献度 (doc_contrib_count)
    # 我们认为 'doc' 和 'mixed' 类型的PR都体现了文档贡献的意愿
    doc_contributors = prs_df[prs_df['contribution_type'].isin(['doc', 'mixed'])]['author_id'].value_counts().rename(
        'doc_contrib_count')

    # 特征5: 社区支持度 (support_count) - 只计算在Issue下的评论
    issue_commenters = comments_df[comments_df['parent_type'] == 'issue']['author_id'].value_counts().rename(
        'support_count')

    # 特征6: 生态探索度 (exploration_count)
    explorers = stars_df['user_id'].value_counts().rename('exploration_count')

    print("[✓] 所有特征计算完毕。")

    # --- 4. 合并所有特征到一个DataFrame ---
    print("[*] 正在将所有特征合并到一个表中...")

    # 我们以完整的用户列表为基础，确保每个用户都有一行
    user_features_df = all_users_df.set_index('user_id')

    # 依次合并所有特征列
    feature_series = [app_creators, tool_creators, core_contributors, doc_contributors, issue_commenters, explorers]
    for series in feature_series:
        user_features_df = user_features_df.join(series)

    # 将所有NaN值（即该用户在该项行为上计数为0）填充为0
    user_features_df.fillna(0, inplace=True)

    # 将计数值转换为整数类型
    for col in user_features_df.columns:
        if col != 'login':  # login列保持为字符串
            user_features_df[col] = user_features_df[col].astype(int)

    # 筛选掉所有特征都为0的“僵尸”用户，只保留在生态中有实际行为的用户
    feature_cols = [s.name for s in feature_series]
    active_users_df = user_features_df[user_features_df[feature_cols].sum(axis=1) > 0]

    print(f"[✓] 特征合并完成。共找到 {len(active_users_df)} 个在生态系统中有活跃行为的用户。")

    # --- 5. 保存结果 ---
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, OUTPUT_FILENAME)
    active_users_df.to_csv(output_path)

    print(f"\n--- 用户行为特征工程完成 ---")
    print(f"[*] 最终的用户特征数据已保存至: '{output_path}'")
    print("\n--- 数据预览 ---")
    print(active_users_df.sort_values(by='create_app_count', ascending=False).head())


if __name__ == "__main__":
    main()