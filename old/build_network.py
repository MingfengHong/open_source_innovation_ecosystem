import pandas as pd
import networkx as nx
import os
from tqdm import tqdm

# = a============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录: 包含最终筛选后数据的目录
INPUT_DIR = "final_analysis_data"
# 包含原始用户信息的目录 (因为users.csv未被筛选，它包含所有可能的用户)
ETL_DIR = "etl_output"
# 输出目录: 用于存放图文件
OUTPUT_DIR = "network_output"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责构建完整的异构信息网络。
    """
    print("--- 步骤 2: 构建异构信息网络 (HIN) 开始 ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. 加载所有经过筛选的数据 ---
    print("[*] 正在加载所有位于 'final_analysis_data' 下的已筛选数据...")
    try:
        repos_df = pd.read_csv(os.path.join(INPUT_DIR, "repos.csv"))
        prs_df = pd.read_csv(os.path.join(INPUT_DIR, "prs.csv"))
        issues_df = pd.read_csv(os.path.join(INPUT_DIR, "issues.csv"))
        comments_df = pd.read_csv(os.path.join(INPUT_DIR, "comments.csv"))
        stars_df = pd.read_csv(os.path.join(INPUT_DIR, "stars.csv"))
        forks_df = pd.read_csv(os.path.join(INPUT_DIR, "forks.csv"))

        # 用户列表需要从原始ETL目录加载，以确保包含所有参与者
        users_df = pd.read_csv(os.path.join(ETL_DIR, "users.csv"))

        # 加载我们之前做的仓库角色分类结果，以附加到节点属性上
        classified_repos_df = pd.read_csv("classification_output/repos_classified.csv")
        # 将分类信息合并到我们的核心仓库数据中
        repos_df = pd.merge(repos_df, classified_repos_df[['repo_name', 'primary_role']], on='repo_name', how='left')

    except FileNotFoundError as e:
        print(f"[!] 错误: 缺少数据文件: {e}")
        print("[!] 请确保所有必需的CSV文件都存在于正确的目录中。")
        return

    print("[✓] 所有数据加载成功。")

    # --- 2. 初始化网络图 ---
    # 我们创建一个简单的图。对于更复杂的分析，也可以使用 nx.DiGraph (有向图)
    # 或 nx.MultiGraph (节点间允许多条不同类型的边)
    G = nx.Graph()
    print("[*] 已初始化一个空的 NetworkX Graph 对象。")

    # --- 3. 添加节点 (Nodes) 及属性 ---
    print("[*] 正在添加节点到图中...")

    # a. 添加仓库 (Repository) 节点
    # 使用tqdm来显示进度条
    for _, row in tqdm(repos_df.iterrows(), total=repos_df.shape[0], desc="添加仓库节点"):
        # repo_id 作为节点的唯一标识符
        # 其他信息作为节点的属性存储
        G.add_node(row['repo_id'],
                   type='repo',
                   name=row['repo_name'],
                   stars=row['stargazer_count'],
                   forks=row['fork_count'],
                   created_at=row['created_at'],
                   primary_role=row.get('primary_role', 'Unknown'))  # 使用.get以防合并失败

    # b. 添加用户 (User) 节点
    # 为了效率，我们只添加在本次分析中实际出现的用户
    active_users = set(prs_df['author_id'].dropna()) | \
                   set(issues_df['author_id'].dropna()) | \
                   set(comments_df['author_id'].dropna()) | \
                   set(stars_df['user_id'].dropna()) | \
                   set(forks_df['user_id'].dropna())

    active_users_df = users_df[users_df['user_id'].isin(active_users)]
    for _, row in tqdm(active_users_df.iterrows(), total=active_users_df.shape[0], desc="添加用户节点"):
        G.add_node(row['user_id'], type='user', login=row['login'])

    print("[✓] 节点添加完成。")

    # --- 4. 添加边 (Edges) 及属性 ---
    print("[*] 正在添加边到图中 (这可能需要一些时间)...")

    # a. 添加 用户-Star->仓库 的边
    for _, row in tqdm(stars_df.iterrows(), total=stars_df.shape[0], desc="添加 Star 边"):
        if row['user_id'] in G and row['repo_id'] in G:
            G.add_edge(row['user_id'], row['repo_id'], type='stars', timestamp=row['starred_at'])

    # b. 添加 用户-Fork->仓库 的边
    for _, row in tqdm(forks_df.iterrows(), total=forks_df.shape[0], desc="添加 Fork 边"):
        if row['user_id'] in G and row['repo_id'] in G:
            G.add_edge(row['user_id'], row['repo_id'], type='forks', timestamp=row['forked_at'])

    # c. 添加 用户-创建->PR 的边，以及 PR-属于->仓库 的边
    for _, row in tqdm(prs_df.iterrows(), total=prs_df.shape[0], desc="添加 PR 边"):
        # PR本身也可以被看作一个节点，来连接用户和仓库
        pr_node_id = f"pr_{row['pr_id']}"  # 创建一个唯一的PR节点ID
        G.add_node(pr_node_id, type='pr', timestamp=row['created_at'])
        # 添加边
        if row['author_id'] in G:
            G.add_edge(row['author_id'], pr_node_id, type='creates_pr')
        if row['repo_id'] in G:
            G.add_edge(pr_node_id, row['repo_id'], type='pr_in_repo')

    # d. 添加 用户-创建->Issue 的边，以及 Issue-属于->仓库 的边
    for _, row in tqdm(issues_df.iterrows(), total=issues_df.shape[0], desc="添加 Issue 边"):
        issue_node_id = f"issue_{row['issue_id']}"
        G.add_node(issue_node_id, type='issue', timestamp=row['created_at'])
        if row['author_id'] in G:
            G.add_edge(row['author_id'], issue_node_id, type='creates_issue')
        if row['repo_id'] in G:
            G.add_edge(issue_node_id, row['repo_id'], type='issue_in_repo')

    # e. 添加 用户-评论于->PR/Issue 的边
    for _, row in tqdm(comments_df.iterrows(), total=comments_df.shape[0], desc="添加 Comment 边"):
        parent_node_id = f"{row['parent_type']}_{row['parent_id']}"
        if row['author_id'] in G and parent_node_id in G:
            G.add_edge(row['author_id'], parent_node_id, type='comments_on', timestamp=row['created_at'])

    print("[✓] 边添加完成。")

    # --- 5. 总结与保存 ---
    print("\n--- 网络构建完成 ---")
    print(f"图中总节点数: {G.number_of_nodes()}")
    print(f"图中总边数: {G.number_of_edges()}")

    # 将完整的图保存到文件，以便将来快速加载
    # GraphML是一个很好的格式，因为它可以保存节点和边的所有属性
    output_path = os.path.join(OUTPUT_DIR, "full_ecosystem_graph.graphml")
    nx.write_graphml(G, output_path)

    print(f"[✓] 完整的网络图已成功保存至: '{output_path}'")
    print("\n您现在拥有了整个生态系统的网络模型，可以开始进行下一步的动态分析了。")


if __name__ == "__main__":
    main()