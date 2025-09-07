import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录
NETWORK_DIR = "network_output"
ANALYSIS_OUTPUT_DIR = "analysis_output"

# 输入文件名
GRAPH_FILENAME = "full_ecosystem_graph.graphml"
ROLES_FILENAME = "user_roles_final.csv"

# 输出文件名
INTERACTION_MATRIX_FILENAME = "role_interaction_matrix.csv"
INTERACTION_HEATMAP_FILENAME = "role_interaction_heatmap.png"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责分析角色间的互动。
    """
    print("--- RQ2a 分析: 角色互动与共生关系 ---")

    # --- 1. 加载网络与角色数据 ---
    graph_path = os.path.join(NETWORK_DIR, GRAPH_FILENAME)
    roles_path = os.path.join(ANALYSIS_OUTPUT_DIR, ROLES_FILENAME)

    print(f"[*] 正在加载网络图: '{graph_path}'...")
    G = nx.read_graphml(graph_path)

    print(f"[*] 正在加载用户角色数据: '{roles_path}'...")
    roles_df = pd.read_csv(roles_path)
    print("[✓] 数据加载完毕。")

    # --- 2. 将角色信息丰富到网络图中 ---
    print("[*] 正在将角色标签添加到网络图的节点属性中...")

    # 创建一个从 user_id -> final_role 的映射字典
    # 我们只关注有明确角色的用户
    role_map = roles_df.set_index('user_id')['final_role'].to_dict()

    # 使用 set_node_attributes 将角色信息批量设置到图中
    nx.set_node_attributes(G, role_map, 'role')

    print("[✓] 网络图已成功丰富角色属性。")

    # --- 3. 预计算用于追踪关系的的映射表 ---
    print("[*] 正在预计算PR/Issue/Repo与其作者的映射关系...")
    # a. PR/Issue -> 作者ID 的映射
    item_to_author = {}
    for node, data in G.nodes(data=True):
        if data.get('type') in ['pr', 'issue']:
            # 找到连接到这个PR/Issue的 "creates" 类型的边
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor].get('type') == 'user':
                    # 假设创建者是唯一的user邻居
                    item_to_author[node] = neighbor
                    break

    # b. Repo -> 作者ID 的映射
    repo_to_owner = {}
    user_login_to_id = {data['login']: user_id for user_id, data in G.nodes(data=True) if data.get('type') == 'user'}
    for node, data in G.nodes(data=True):
        if data.get('type') == 'repo':
            owner_login = data.get('name', '/').split('/')[0]
            if owner_login in user_login_to_id:
                repo_to_owner[node] = user_login_to_id[owner_login]

    print("[✓] 预计算完成。")

    # --- 4. 遍历边，统计角色间互动 ---
    print("[*] 正在遍历网络，统计角色间的互动...")
    interactions = []

    for u, v, data in tqdm(G.edges(data=True), desc="分析互动"):
        u_data = G.nodes[u]
        v_data = G.nodes[v]

        source_user_id, target_user_id = None, None

        edge_type = data.get('type')

        # a. 评论互动：评论者 -> PR/Issue的创建者
        if edge_type == 'comments_on':
            commenter_id = u if u_data.get('type') == 'user' else v
            item_id = v if u_data.get('type') == 'user' else u
            if item_id in item_to_author:
                source_user_id = commenter_id
                target_user_id = item_to_author[item_id]

        # b. Star/Fork互动：操作者 -> 仓库的拥有者
        elif edge_type in ['stars', 'forks']:
            starrer_id = u if u_data.get('type') == 'user' else v
            repo_id = v if u_data.get('type') == 'user' else u
            if repo_id in repo_to_owner:
                source_user_id = starrer_id
                target_user_id = repo_to_owner[repo_id]

        # 如果成功解析出一次互动
        if source_user_id and target_user_id and source_user_id != target_user_id:
            source_role = G.nodes[source_user_id].get('role')
            target_role = G.nodes[target_user_id].get('role')

            if source_role and target_role:
                interactions.append({'source_role': source_role, 'target_role': target_role})

    interactions_df = pd.DataFrame(interactions)
    print(f"[✓] 互动统计完成，共发现 {len(interactions_df)} 次有效的跨角色互动。")

    # --- 5. 构建并保存互动矩阵 ---
    print("[*] 正在构建角色互动矩阵...")
    # 获取所有独特的角色名称，并排序
    role_names = sorted(roles_df['final_role'].unique())

    # 使用 crosstab 功能快速生成互动计数矩阵
    interaction_matrix = pd.crosstab(interactions_df['source_role'], interactions_df['target_role'])

    # 确保矩阵包含所有角色，即使某些角色没有互动
    interaction_matrix = interaction_matrix.reindex(index=role_names, columns=role_names, fill_value=0)

    matrix_path = os.path.join(ANALYSIS_OUTPUT_DIR, INTERACTION_MATRIX_FILENAME)
    interaction_matrix.to_csv(matrix_path)
    print(f"[✓] 互动矩阵已保存至: '{matrix_path}'")
    print("\n--- 角色互动矩阵 (原始计数) ---")
    print(interaction_matrix)

    # --- 6. 可视化互动矩阵 ---
    print("\n[*] 正在生成角色互动热力图...")

    # 指定中文字体，解决中文显示为方框的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方框的问题

    # 对角线置零
    matrix_no_diagonal = interaction_matrix.copy()
    for role in role_names:
        if role in matrix_no_diagonal.index and role in matrix_no_diagonal.columns:
            matrix_no_diagonal.loc[role, role] = 0

    plt.figure(figsize=(16, 12))
    sns.heatmap(
        matrix_no_diagonal,
        annot=True,  # 在格子上显示数字
        fmt='d',  # 数字格式为整数
        cmap='viridis',  # 使用 viridis 颜色映射
        linewidths=.5
    )
    plt.title('Role Interaction Matrix (Excluding Self-Interaction)', fontsize=16)
    plt.xlabel('Target Role (Interaction Receiver)', fontsize=12)
    plt.ylabel('Source Role (Interaction Initiator)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    heatmap_path = os.path.join(ANALYSIS_OUTPUT_DIR, INTERACTION_HEATMAP_FILENAME)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)

    print(f"[✓] 互动热力图已保存至: '{heatmap_path}'")
    plt.show()


if __name__ == "__main__":
    main()