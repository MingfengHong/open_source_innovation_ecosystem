import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录: 存放图文件的目录
NETWORK_DIR = "network_output"
# 输出目录: 用于存放分析结果图表
ANALYSIS_OUTPUT_DIR = "analysis_output"
# 图文件名
GRAPH_FILENAME = "full_ecosystem_graph.graphml"

# LangChain核心团队成员的GitHub login name

# 创始人
FOUNDERS = [
    "hwchase17",  # Harrison Chase
    "agola11",    # Ankush Gola
]

# 核心维护者（根据官方文档）
CORE_MAINTAINERS = [
    "baskaryan",
    "ccurme",
    "hinthornw",
    "rlancemartin",
    "nfcampos",
    "vbarda",
    "efriis",
    "eyurtsev",
]

# 合并所有核心团队成员
CORE_TEAM_LOGINS = FOUNDERS + CORE_MAINTAINERS

# 分析的时间范围
START_DATE = "2022-11-01"  # LangChain 项目初期
END_DATE = "2024-12-31"  # 数据收集的截止日期


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责执行动态网络分析。
    """
    print("--- 步骤 3: 动态网络分析 (RQ1a) 开始 ---")
    os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

    # --- 1. 加载完整的网络图 ---
    graph_path = os.path.join(NETWORK_DIR, GRAPH_FILENAME)
    print(f"[*] 正在从 '{graph_path}' 加载完整的网络图...")
    try:
        G = nx.read_graphml(graph_path)
    except FileNotFoundError:
        print(f"[!] 错误: 未找到图文件 '{graph_path}'。请确保您已成功运行 build_network.py 脚本。")
        return
    print("[✓] 网络图加载成功。")

    # --- 2. 准备工作：识别核心团队节点的ID ---
    # 我们需要从login name找到他们在图中的node_id
    core_team_node_ids = []
    # 创建一个从 login -> node_id 的映射以便快速查找
    login_to_node_id = {data['login']: node for node, data in G.nodes(data=True) if data.get('type') == 'user'}

    for login in CORE_TEAM_LOGINS:
        if login in login_to_node_id:
            core_team_node_ids.append(login_to_node_id[login])
        else:
            print(f"[!] 警告: 核心成员 '{login}' 未在图中找到，可能他/她在此生态中没有活动。")

    if not core_team_node_ids:
        print("[!] 错误: 核心团队成员ID列表为空，无法继续分析。请检查 CORE_TEAM_LOGINS 配置。")
        return

    print(f"[*] 已成功识别 {len(core_team_node_ids)} 位核心团队成员的节点ID。")

    # --- 3. 按月生成网络快照并计算中心性 ---
    print("[*] 开始按月生成网络快照并计算中心性...")

    # 生成月份范围
    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')  # 'MS' 表示月份的开始

    analysis_results = []

    # 遍历每个月份
    for start_of_month in tqdm(date_range, desc="分析各月份网络"):
        end_of_month = start_of_month + pd.offsets.MonthEnd(0)

        # a. 创建当月的子图 (网络快照)
        G_month = nx.Graph()
        # 首先，将所有节点添加到月度图中，确保节点集稳定
        G_month.add_nodes_from(G.nodes(data=True))

        # 然后，只添加时间戳在本月内的边
        edges_in_month = []
        for u, v, data in G.edges(data=True):
            # 检查边是否有时间戳属性
            if 'timestamp' in data:
                try:
                    # 使用 .tz_localize(None) 移除时区信息
                    ts = pd.to_datetime(data['timestamp']).tz_localize(None)
                    # 判断时间戳是否在本月范围内
                    if start_of_month <= ts <= end_of_month:
                        edges_in_month.append((u, v))
                except (ValueError, TypeError):
                    # 如果时间戳格式不正确，则忽略这条边
                    continue

        G_month.add_edges_from(edges_in_month)

        # 移除度为0的孤立节点，以获得更准确的中心性计算
        G_month.remove_nodes_from(list(nx.isolates(G_month)))

        if G_month.number_of_nodes() == 0:
            continue  # 如果这个月没有任何活动，则跳过

        # b. 计算度中心性 (Degree Centrality)
        # 度中心性衡量一个节点直接连接的数量，是影响力最直接的体现
        # NetworkX的函数会返回一个包含所有节点中心性的字典
        centrality = nx.degree_centrality(G_month)

        # c. 提取核心团队的中心性
        core_team_centrality = [centrality.get(node_id, 0) for node_id in core_team_node_ids]

        # 计算核心团队的平均中心性
        avg_core_centrality = sum(core_team_centrality) / len(core_team_centrality) if core_team_centrality else 0

        # d. 存储当月结果
        analysis_results.append({
            "month": start_of_month,
            "avg_core_centrality": avg_core_centrality,
            "nodes_in_month": G_month.number_of_nodes(),
            "edges_in_month": G_month.number_of_edges()
        })

    print("[✓] 所有月份的中心性计算完成。")

    # --- 4. 结果可视化 ---
    if not analysis_results:
        print("[!] 没有计算出任何结果，无法生成图表。")
        return

    results_df = pd.DataFrame(analysis_results)

    print("\n--- 分析结果预览 ---")
    print(results_df)

    print("\n[*] 正在生成中心性变化趋势图...")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # 绘制核心团队平均中心性
    ax.plot(results_df['month'], results_df['avg_core_centrality'], marker='o', linestyle='-',
            label='Core Team Avg. Degree Centrality')

    ax.set_title('LangChain Core Team Influence Over Time (Degree Centrality)', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Degree Centrality', fontsize=12)
    ax.legend()
    ax.grid(True)

    # 格式化X轴的日期显示
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # 每3个月一个主刻度
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, "rq1a_core_team_centrality_over_time.png")
    plt.savefig(output_path, dpi=300)

    print(f"[✓] 图表已成功保存至: '{output_path}'")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    main()