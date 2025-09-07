import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from networkx.algorithms import community as nx_comm

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
NETWORK_DIR = "network_output"
ANALYSIS_OUTPUT_DIR = "analysis_output"
GRAPH_FILENAME = "full_ecosystem_graph.graphml"
START_DATE = "2022-11-01"
END_DATE = "2024-12-31"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责执行社区结构的动态网络分析，并导出详细成员数据。
    """
    print("--- 动态网络分析 (V2: 社区发现 + 详细数据导出) 开始 ---")
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    # --- 1. 加载完整的网络图 ---
    graph_path = os.path.join(NETWORK_DIR, GRAPH_FILENAME)
    print(f"[*] 正在从 '{graph_path}' 加载完整的网络图...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        print(f"[!] 错误: 未找到图文件 '{graph_path}'。")
        return
    print("[✓] 网络图加载成功。")

    # --- 2. 按月生成网络快照并进行社区分析 ---
    print("[*] 开始按月生成网络快照并进行社区分析...")

    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

    # 初始化两个列表，一个用于宏观统计，一个用于详细数据
    macro_analysis_results = []
    detailed_community_data = []

    for start_of_month in tqdm(date_range, desc="分析各月份社区结构"):
        end_of_month = start_of_month + pd.offsets.MonthEnd(0)

        # a. 创建当月的子图 (网络快照)
        G_month = nx.Graph()
        G_month.add_nodes_from(G.nodes(data=True))

        edges_in_month = []
        for u, v, data in G.edges(data=True):
            if 'timestamp' in data:
                try:
                    ts = pd.to_datetime(data['timestamp']).tz_localize(None)
                    if start_of_month <= ts <= end_of_month:
                        edges_in_month.append((u, v))
                except (ValueError, TypeError):
                    continue

        G_month.add_edges_from(edges_in_month)
        G_month.remove_nodes_from(list(nx.isolates(G_month)))

        if G_month.number_of_edges() == 0:
            continue

        # b. 运行 Louvain 社区发现算法
        communities = nx_comm.louvain_communities(G_month, seed=123)

        # c. 计算宏观指标
        num_communities = len(communities)
        modularity = nx_comm.modularity(G_month, communities)
        community_sizes = [len(c) for c in communities]
        largest_community_size = max(community_sizes) if community_sizes else 0

        macro_analysis_results.append({
            "month": start_of_month,
            "num_communities": num_communities,
            "modularity": modularity,
            "largest_community_size": largest_community_size,
            "nodes_in_month": G_month.number_of_nodes(),
            "edges_in_month": G_month.number_of_edges()
        })

        # --- 新增：导出详细社区成员信息 ---
        # 1. 按社区大小降序排序，方便我们定位大社区
        sorted_communities = sorted(communities, key=len, reverse=True)

        # 2. 遍历每个社区，记录其所有成员信息
        for i, community_nodes in enumerate(sorted_communities):
            community_id = f"{start_of_month.strftime('%Y-%m')}_C{i}"  # 例如: 2023-05_C0
            community_size = len(community_nodes)

            for node_id in community_nodes:
                node_data = G.nodes[node_id]  # 从主图中获取节点属性
                node_type = node_data.get('type')

                # 根据节点类型确定标签和其他属性
                node_label = ""
                repo_role = None  # 只有仓库节点有此属性
                if node_type == 'user':
                    node_label = node_data.get('login', '')
                elif node_type == 'repo':
                    node_label = node_data.get('name', '')
                    repo_role = node_data.get('primary_role', '')

                detailed_community_data.append({
                    'month': start_of_month.strftime('%Y-%m'),
                    'community_id': community_id,
                    'community_size': community_size,
                    'node_id': node_id,
                    'node_type': node_type,
                    'node_label': node_label,
                    'repo_primary_role': repo_role
                })

    print("[✓] 所有月份的社区分析完成。")

    # --- 3. 保存详细数据文件 ---
    print("[*] 正在保存详细的社区成员数据到CSV文件...")
    details_df = pd.DataFrame(detailed_community_data)
    details_output_path = os.path.join(ANALYSIS_OUTPUT_DIR, "community_details_by_month.csv")
    details_df.to_csv(details_output_path, index=False, encoding='utf-8-sig')
    print(f"[✓] 详细数据已成功保存至: '{details_output_path}'")

    # --- 4. 结果可视化 (宏观图表) ---
    if not macro_analysis_results:
        print("[!] 没有计算出任何结果，无法生成图表。")
        return

    results_df = pd.DataFrame(macro_analysis_results)

    print("\n--- 宏观分析结果预览 ---")
    print(results_df)

    # (这部分可视化代码与之前完全相同)
    print("\n[*] 正在生成社区结构变化趋势图...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    ax1.plot(results_df['month'], results_df['num_communities'], marker='o', linestyle='-', color='b')
    ax1.set_title('Number of Communities Over Time', fontsize=16)
    ax1.set_ylabel('Total Communities Detected', fontsize=12)
    ax1.grid(True)
    ax2.plot(results_df['month'], results_df['largest_community_size'], marker='s', linestyle='--', color='r')
    ax2.set_title('Size of the Largest Community Over Time', fontsize=16)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Number of Nodes in Largest Community', fontsize=12)
    ax2.grid(True)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, "rq1b_community_evolution_over_time.png")
    plt.savefig(output_path, dpi=300)
    print(f"[✓] 图表已成功保存至: '{output_path}'")
    plt.show()


if __name__ == "__main__":
    main()