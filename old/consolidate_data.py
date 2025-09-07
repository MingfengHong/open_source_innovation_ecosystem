import pandas as pd
import os

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录
# 包含ETL处理后、未经过滤的CSV文件的目录
ETL_DIR = "etl_output"
# 包含LLM相关性判断结果的目录
JUDGMENT_DIR = "relevance_judgment_output"

# 输出目录
# 用于存放最终筛选后的、可用于分析的干净数据
OUTPUT_DIR = "final_analysis_data"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责执行整个数据整合和筛选流程。
    """
    print("--- 步骤 1: 数据整合与最终筛选开始 ---")

    # --- 准备工作: 创建输出目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[+] 输出目录 '{OUTPUT_DIR}' 已准备就绪。")

    # --- 核心步骤 1: 读取判断结果，生成“核心仓库列表” ---
    try:
        judgment_file_path = os.path.join(JUDGMENT_DIR, "repo_relevance_judgment.csv")
        judgment_df = pd.read_csv(judgment_file_path)
    except FileNotFoundError:
        print(f"[!] 错误: 无法找到相关性判断文件: {judgment_file_path}")
        print("[!] 请确保您已成功运行了之前的相关性判断脚本。")
        return

    # 筛选出所有 is_relevant_to_study 为 TRUE 的记录
    relevant_repos_df = judgment_df[judgment_df['is_relevant_to_study'] == True]

    # 提取 repo_name 列表，使用集合(set)可以极大提高后续的查询效率
    core_repo_names = set(relevant_repos_df['repo_name'])

    if not core_repo_names:
        print("[!] 警告: 在相关性判断文件中没有找到任何标记为 TRUE 的仓库。脚本将终止。")
        return

    print(f"\n[*] 从判断结果中成功筛选出 {len(core_repo_names)} 个核心仓库。")

    # --- 核心步骤 2: 获取核心仓库的ID ---
    # `repos.csv` 是连接 `repo_name` 和 `repo_id` 的桥梁
    repos_file_path = os.path.join(ETL_DIR, "repos.csv")
    repos_df = pd.read_csv(repos_file_path)

    # 从总的仓库表中，根据“核心仓库名称列表”筛选出核心仓库的信息
    final_repos_df = repos_df[repos_df['repo_name'].isin(core_repo_names)]

    # 获取这些核心仓库的唯一ID列表，这将是我们筛选其他文件的关键
    core_repo_ids = set(final_repos_df['repo_id'])

    # 保存筛选后的仓库信息表到新目录
    final_repos_output_path = os.path.join(OUTPUT_DIR, "repos.csv")
    final_repos_df.to_csv(final_repos_output_path, index=False, encoding='utf-8-sig')
    print(f"[✓] 已筛选 'repos.csv' 并保存，包含 {len(final_repos_df)} 条记录。")

    # --- 核心步骤 3: 遍历并筛选所有其他ETL数据表 ---
    # 定义需要按 `repo_id` 筛选的文件列表
    files_to_filter = [
        "prs.csv",
        "issues.csv",
        "comments.csv",
        "stars.csv",
        "forks.csv"
    ]

    print("\n[*] 开始筛选与核心仓库相关的活动记录...")

    for filename in files_to_filter:
        input_path = os.path.join(ETL_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        try:
            # 读取原始ETL文件
            df = pd.read_csv(input_path, low_memory=False)  # low_memory=False 防止因列类型推断不一致产生的警告
            original_count = len(df)

            # 使用 `core_repo_ids` 集合进行高效筛选
            filtered_df = df[df['repo_id'].isin(core_repo_ids)]
            filtered_count = len(filtered_df)

            # 保存筛选后的文件
            filtered_df.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"[✓] 已筛选 '{filename}': {original_count} 条记录 -> {filtered_count} 条记录。")

        except FileNotFoundError:
            print(f"[!] 警告: 未找到文件 {input_path}，已跳过。")
        except KeyError:
            print(f"[!] 警告: 文件 {input_path} 中不包含 'repo_id' 列，已跳过。")

    print("\n--- 所有数据表均已整合筛选完毕！ ---")
    print(f"[*] 您现在可以在 '{OUTPUT_DIR}' 目录下找到用于下一步网络构建的所有干净数据。")


if __name__ == "__main__":
    main()