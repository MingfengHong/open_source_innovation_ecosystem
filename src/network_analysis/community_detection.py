"""
社区检测模块 (增强版)
支持多种社区检测算法，包括Louvain、Leiden、Infomap等
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None

# 导入社区检测算法
try:
    import leidenalg as la
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning("Leiden算法不可用，请安装leidenalg和igraph: pip install leidenalg igraph")

try:
    import infomap
    INFOMAP_AVAILABLE = True
except ImportError:
    INFOMAP_AVAILABLE = False
    logging.warning("Infomap算法不可用，请安装infomap: pip install infomap")

from networkx.algorithms import community as nx_comm
from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, VISUALIZATION_CONFIG, ANALYSIS_CONFIG
from ..network_analysis.centrality_measures import CentralityCalculator

# 设置日志
logger = setup_logger(__name__)


class CommunityDetector:
    """社区检测器"""
    
    def __init__(self, graph: nx.Graph, random_seed: int = 42):
        """
        初始化社区检测器
        
        Args:
            graph: NetworkX图对象
            random_seed: 随机种子
        """
        self.graph = graph
        self.random_seed = random_seed
        self.communities_cache = {}
        
        # 可用的算法列表
        self.available_algorithms = ['louvain']
        if LEIDEN_AVAILABLE:
            self.available_algorithms.append('leiden')
        if INFOMAP_AVAILABLE:
            self.available_algorithms.append('infomap')
    
    def detect_louvain_communities(self, resolution: float = 1.0) -> List[Set]:
        """
        使用Louvain算法检测社区
        
        Args:
            resolution: 分辨率参数
            
        Returns:
            List[Set]: 社区列表，每个社区是节点集合
        """
        cache_key = f'louvain_res{resolution}'
        if cache_key not in self.communities_cache:
            logger.info(f"运行Louvain算法 (resolution={resolution})...")
            communities = nx_comm.louvain_communities(
                self.graph, 
                resolution=resolution, 
                seed=self.random_seed
            )
            self.communities_cache[cache_key] = communities
        
        return self.communities_cache[cache_key]
    
    def detect_leiden_communities(self, resolution: float = 1.0) -> List[Set]:
        """
        使用Leiden算法检测社区
        
        Args:
            resolution: 分辨率参数
            
        Returns:
            List[Set]: 社区列表，每个社区是节点集合
        """
        if not LEIDEN_AVAILABLE:
            logger.error("Leiden算法不可用")
            return []
        
        cache_key = f'leiden_res{resolution}'
        if cache_key not in self.communities_cache:
            logger.info(f"运行Leiden算法 (resolution={resolution})...")
            
            # 将NetworkX图转换为igraph图
            ig_graph = self._networkx_to_igraph()
            
            # 运行Leiden算法
            partition = la.find_partition(
                ig_graph, 
                la.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=self.random_seed
            )
            
            # 转换回NetworkX格式
            node_list = list(self.graph.nodes())
            communities = []
            for community_nodes in partition:
                community_set = {node_list[i] for i in community_nodes}
                communities.append(community_set)
            
            self.communities_cache[cache_key] = communities
        
        return self.communities_cache[cache_key]
    
    def detect_infomap_communities(self, directed: bool = False) -> List[Set]:
        """
        使用Infomap算法检测社区
        
        Args:
            directed: 是否为有向图
            
        Returns:
            List[Set]: 社区列表，每个社区是节点集合
        """
        if not INFOMAP_AVAILABLE:
            logger.error("Infomap算法不可用")
            return []
        
        cache_key = f'infomap_directed{directed}'
        if cache_key not in self.communities_cache:
            logger.info(f"运行Infomap算法 (directed={directed})...")
            
            # 创建Infomap实例
            im = infomap.Infomap(directed=directed, silent=True)
            
            # 添加节点和边
            node_to_id = {node: i for i, node in enumerate(self.graph.nodes())}
            id_to_node = {i: node for node, i in node_to_id.items()}
            
            for u, v in self.graph.edges():
                im.add_link(node_to_id[u], node_to_id[v])
            
            # 运行算法
            im.run()
            
            # 提取社区
            communities = defaultdict(set)
            for node in im.tree:
                if node.is_leaf:
                    module_id = node.module_id
                    original_node = id_to_node[node.physical_id]
                    communities[module_id].add(original_node)
            
            self.communities_cache[cache_key] = list(communities.values())
        
        return self.communities_cache[cache_key]
    
    def detect_all_algorithms(self, resolution: float = 1.0) -> Dict[str, List[Set]]:
        """
        使用所有可用算法检测社区
        
        Args:
            resolution: 分辨率参数（仅适用于支持的算法）
            
        Returns:
            Dict[str, List[Set]]: 算法名到社区列表的映射
        """
        results = {}
        
        # Louvain算法
        try:
            results['louvain'] = self.detect_louvain_communities(resolution)
        except Exception as e:
            logger.error(f"Louvain算法执行失败: {e}")
        
        # Leiden算法
        if LEIDEN_AVAILABLE:
            try:
                results['leiden'] = self.detect_leiden_communities(resolution)
            except Exception as e:
                logger.error(f"Leiden算法执行失败: {e}")
        
        # Infomap算法
        if INFOMAP_AVAILABLE:
            try:
                results['infomap'] = self.detect_infomap_communities()
            except Exception as e:
                logger.error(f"Infomap算法执行失败: {e}")
        
        return results
    
    def calculate_modularity(self, communities: List[Set]) -> float:
        """
        计算社区划分的模块度
        
        Args:
            communities: 社区列表
            
        Returns:
            float: 模块度值
        """
        try:
            return nx_comm.modularity(self.graph, communities)
        except Exception as e:
            logger.error(f"计算模块度失败: {e}")
            return 0.0
    
    def compare_algorithms(self, resolution: float = 1.0) -> pd.DataFrame:
        """
        比较不同算法的性能
        
        Args:
            resolution: 分辨率参数
            
        Returns:
            pd.DataFrame: 比较结果表格
        """
        logger.info("比较不同社区检测算法...")
        
        all_results = self.detect_all_algorithms(resolution)
        comparison_data = []
        
        for algorithm_name, communities in all_results.items():
            if not communities:
                continue
            
            modularity = self.calculate_modularity(communities)
            num_communities = len(communities)
            community_sizes = [len(c) for c in communities]
            largest_community_size = max(community_sizes) if community_sizes else 0
            avg_community_size = np.mean(community_sizes) if community_sizes else 0
            
            comparison_data.append({
                'algorithm': algorithm_name,
                'modularity': modularity,
                'num_communities': num_communities,
                'largest_community_size': largest_community_size,
                'avg_community_size': avg_community_size,
                'total_nodes': sum(community_sizes)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        output_path = ANALYSIS_OUTPUT_DIR / "community_algorithms_comparison.csv"
        comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"算法比较结果已保存至: {output_path}")
        
        return comparison_df
    
    def visualize_algorithm_comparison(self, comparison_df: pd.DataFrame):
        """
        可视化算法比较结果
        
        Args:
            comparison_df: 比较结果数据框
        """
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 模块度比较
        axes[0,0].bar(comparison_df['algorithm'], comparison_df['modularity'])
        axes[0,0].set_title('Modularity Comparison', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,0].set_ylabel('Modularity')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 社区数量比较
        axes[0,1].bar(comparison_df['algorithm'], comparison_df['num_communities'])
        axes[0,1].set_title('Number of Communities', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,1].set_ylabel('Number of Communities')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 最大社区大小比较
        axes[1,0].bar(comparison_df['algorithm'], comparison_df['largest_community_size'])
        axes[1,0].set_title('Largest Community Size', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,0].set_ylabel('Size of Largest Community')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 平均社区大小比较
        axes[1,1].bar(comparison_df['algorithm'], comparison_df['avg_community_size'])
        axes[1,1].set_title('Average Community Size', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,1].set_ylabel('Average Community Size')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "community_algorithms_comparison.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"算法比较图表已保存至: {save_path}")
        
        plt.show()
    
    def _networkx_to_igraph(self) -> 'ig.Graph':
        """将NetworkX图转换为igraph图（私有方法）"""
        node_list = list(self.graph.nodes())
        node_to_id = {node: i for i, node in enumerate(node_list)}
        
        edges = [(node_to_id[u], node_to_id[v]) for u, v in self.graph.edges()]
        
        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(node_list))
        ig_graph.add_edges(edges)
        
        return ig_graph
    
    def export_community_details(self, 
                                algorithm: str = 'louvain',
                                resolution: float = 1.0) -> pd.DataFrame:
        """
        导出社区详细信息
        
        Args:
            algorithm: 使用的算法
            resolution: 分辨率参数
            
        Returns:
            pd.DataFrame: 社区详细信息
        """
        if algorithm == 'louvain':
            communities = self.detect_louvain_communities(resolution)
        elif algorithm == 'leiden' and LEIDEN_AVAILABLE:
            communities = self.detect_leiden_communities(resolution)
        elif algorithm == 'infomap' and INFOMAP_AVAILABLE:
            communities = self.detect_infomap_communities()
        else:
            logger.error(f"不支持的算法或算法不可用: {algorithm}")
            return pd.DataFrame()
        
        # 创建详细信息列表
        detailed_data = []
        sorted_communities = sorted(communities, key=len, reverse=True)
        
        for i, community_nodes in enumerate(sorted_communities):
            community_id = f"{algorithm}_C{i}"
            community_size = len(community_nodes)
            
            for node_id in community_nodes:
                node_data = self.graph.nodes.get(node_id, {})
                node_type = node_data.get('type', 'unknown')
                
                # 根据节点类型确定标签
                node_label = ""
                if node_type == 'user':
                    node_label = node_data.get('login', '')
                elif node_type == 'repo':
                    node_label = node_data.get('name', '')
                
                detailed_data.append({
                    'algorithm': algorithm,
                    'community_id': community_id,
                    'community_size': community_size,
                    'node_id': node_id,
                    'node_type': node_type,
                    'node_label': node_label
                })
        
        details_df = pd.DataFrame(detailed_data)
        
        # 保存详细信息
        output_path = ANALYSIS_OUTPUT_DIR / f"community_details_{algorithm}.csv"
        details_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"社区详细信息已保存至: {output_path}")
        
        return details_df


class DynamicCommunityAnalyzer:
    """动态社区分析器"""
    
    def __init__(self, 
                 graph: nx.Graph,
                 start_date: str,
                 end_date: str,
                 algorithms: List[str] = None):
        """
        初始化动态社区分析器
        
        Args:
            graph: 完整的网络图
            start_date: 开始日期
            end_date: 结束日期
            algorithms: 要使用的算法列表
        """
        self.graph = graph
        self.start_date = start_date
        self.end_date = end_date
        self.algorithms = algorithms or ['louvain']
        
        # 确保只使用可用的算法
        available_algorithms = ['louvain']
        if LEIDEN_AVAILABLE:
            available_algorithms.append('leiden')
        if INFOMAP_AVAILABLE:
            available_algorithms.append('infomap')
        
        self.algorithms = [alg for alg in self.algorithms if alg in available_algorithms]
        
        if not self.algorithms:
            self.algorithms = ['louvain']  # 至少使用Louvain作为后备
        
        logger.info(f"将使用以下算法进行动态社区分析: {self.algorithms}")
    
    def create_monthly_snapshot(self, start_of_month: pd.Timestamp, end_of_month: pd.Timestamp) -> nx.Graph:
        """创建月度网络快照"""
        G_month = nx.Graph()
        G_month.add_nodes_from(self.graph.nodes(data=True))
        
        edges_in_month = []
        for u, v, data in self.graph.edges(data=True):
            if 'timestamp' in data:
                try:
                    ts = pd.to_datetime(data['timestamp']).tz_localize(None)
                    if start_of_month <= ts <= end_of_month:
                        edges_in_month.append((u, v))
                except (ValueError, TypeError):
                    continue
        
        G_month.add_edges_from(edges_in_month)
        G_month.remove_nodes_from(list(nx.isolates(G_month)))
        
        return G_month
    
    def analyze_temporal_communities(self) -> pd.DataFrame:
        """
        分析时间序列社区结构
        
        Returns:
            pd.DataFrame: 时间序列社区分析结果
        """
        logger.info("开始动态社区分析...")
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        results = []
        
        for start_of_month in tqdm(date_range, desc="分析各月份社区结构"):
            end_of_month = start_of_month + pd.offsets.MonthEnd(0)
            
            # 创建月度快照
            G_month = self.create_monthly_snapshot(start_of_month, end_of_month)
            
            if G_month.number_of_edges() == 0:
                continue
            
            # 使用各种算法检测社区
            detector = CommunityDetector(G_month)
            
            for algorithm in self.algorithms:
                try:
                    if algorithm == 'louvain':
                        communities = detector.detect_louvain_communities()
                    elif algorithm == 'leiden':
                        communities = detector.detect_leiden_communities()
                    elif algorithm == 'infomap':
                        communities = detector.detect_infomap_communities()
                    else:
                        continue
                    
                    if not communities:
                        continue
                    
                    modularity = detector.calculate_modularity(communities)
                    num_communities = len(communities)
                    community_sizes = [len(c) for c in communities]
                    largest_community_size = max(community_sizes) if community_sizes else 0
                    
                    results.append({
                        'month': start_of_month,
                        'algorithm': algorithm,
                        'modularity': modularity,
                        'num_communities': num_communities,
                        'largest_community_size': largest_community_size,
                        'nodes_in_month': G_month.number_of_nodes(),
                        'edges_in_month': G_month.number_of_edges()
                    })
                    
                except Exception as e:
                    logger.error(f"算法 {algorithm} 在月份 {start_of_month} 执行失败: {e}")
        
        results_df = pd.DataFrame(results)
        
        # 保存结果
        output_path = ANALYSIS_OUTPUT_DIR / "dynamic_community_analysis_multi_algorithm.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"动态社区分析结果已保存至: {output_path}")
        
        return results_df
    
    def visualize_temporal_trends(self, results_df: pd.DataFrame):
        """可视化时间序列趋势"""
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        # 创建子图
        metrics = ['modularity', 'num_communities', 'largest_community_size']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4*len(metrics)), sharex=True)
        
        if len(metrics) == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.algorithms)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for j, algorithm in enumerate(self.algorithms):
                algo_data = results_df[results_df['algorithm'] == algorithm]
                if not algo_data.empty:
                    ax.plot(algo_data['month'], algo_data[metric], 
                           marker='o', linestyle='-', label=algorithm.title(),
                           color=colors[j], linewidth=2, markersize=4)
            
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time', 
                        fontsize=VISUALIZATION_CONFIG["title_font_size"])
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Month')
        axes[-1].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "dynamic_community_trends_comparison.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"动态社区趋势图已保存至: {save_path}")
        
        plt.show()


class SubCenterDetector:
    """
    子中心识别器
    识别网络中的非官方创新子中心及其演化过程
    """
    
    def __init__(self, 
                 graph: nx.Graph,
                 core_team_logins: List[str] = None,
                 min_subcenter_size: int = 5,
                 innovation_threshold: float = 0.1):
        """
        初始化子中心识别器
        
        Args:
            graph: 网络图
            core_team_logins: 核心团队成员登录名列表
            min_subcenter_size: 子中心最小规模
            innovation_threshold: 创新产出阈值
        """
        self.graph = graph
        self.core_team_logins = core_team_logins or ANALYSIS_CONFIG["core_team_logins"]
        self.min_subcenter_size = min_subcenter_size
        self.innovation_threshold = innovation_threshold
        
        # 识别核心团队节点
        self.core_team_nodes = self._identify_core_team_nodes()
        
        logger.info(f"初始化子中心识别器: 核心团队节点 {len(self.core_team_nodes)} 个")
    
    def _identify_core_team_nodes(self) -> Set[str]:
        """识别核心团队节点"""
        core_nodes = set()
        
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'user':
                login = data.get('login', '')
                if login in self.core_team_logins:
                    core_nodes.add(node)
        
        return core_nodes
    
    def identify_sub_centers(self, 
                           communities: List[Set],
                           algorithm_name: str = 'louvain') -> List[Dict]:
        """
        识别功能性子中心
        
        Args:
            communities: 社区列表
            algorithm_name: 社区检测算法名称
            
        Returns:
            List[Dict]: 子中心信息列表
        """
        logger.info(f"基于 {algorithm_name} 社区检测结果识别子中心...")
        
        sub_centers = []
        
        for i, community in enumerate(communities):
            # 跳过过小的社区
            if len(community) < self.min_subcenter_size:
                continue
            
            # 检查是否为官方核心团队主导的社区
            core_members_in_community = community.intersection(self.core_team_nodes)
            is_official_center = len(core_members_in_community) > len(community) * 0.3
            
            if is_official_center:
                continue  # 跳过官方主导的社区
            
            # 计算子中心的创新指标
            innovation_metrics = self._calculate_innovation_metrics(community)
            
            # 判断是否达到创新阈值
            if innovation_metrics['innovation_score'] < self.innovation_threshold:
                continue
            
            # 识别子中心的关键节点（局部领导者）
            key_nodes = self._identify_key_nodes_in_community(community)
            
            # 分析子中心的功能特征
            functional_profile = self._analyze_functional_profile(community)
            
            sub_center_info = {
                'subcenter_id': f"{algorithm_name}_SC{i}",
                'algorithm': algorithm_name,
                'size': len(community),
                'nodes': list(community),
                'key_nodes': key_nodes,
                'core_members_count': len(core_members_in_community),
                'is_official_dominated': is_official_center,
                'innovation_score': innovation_metrics['innovation_score'],
                'app_creation_rate': innovation_metrics['app_creation_rate'],
                'code_contribution_rate': innovation_metrics['code_contribution_rate'],
                'knowledge_sharing_rate': innovation_metrics['knowledge_sharing_rate'],
                'functional_type': functional_profile['primary_function'],
                'specialization_index': functional_profile['specialization_index'],
                'external_connectivity': self._calculate_external_connectivity(community),
                'internal_density': self._calculate_internal_density(community)
            }
            
            sub_centers.append(sub_center_info)
        
        logger.info(f"识别出 {len(sub_centers)} 个潜在子中心")
        return sub_centers
    
    def _calculate_innovation_metrics(self, community: Set) -> Dict[str, float]:
        """计算社区的创新指标"""
        # 获取社区中的用户节点
        user_nodes = [node for node in community 
                     if self.graph.nodes[node].get('type') == 'user']
        
        if not user_nodes:
            return {'innovation_score': 0.0, 'app_creation_rate': 0.0, 
                   'code_contribution_rate': 0.0, 'knowledge_sharing_rate': 0.0}
        
        # 计算应用创建率（连接到application类型仓库的比例）
        app_connections = 0
        code_contributions = 0
        doc_contributions = 0
        total_connections = 0
        
        for user_node in user_nodes:
            for neighbor in self.graph.neighbors(user_node):
                neighbor_data = self.graph.nodes[neighbor]
                if neighbor_data.get('type') == 'repo':
                    total_connections += 1
                    # 检查仓库类型
                    if neighbor_data.get('primary_role') == 'application':
                        app_connections += 1
                elif neighbor_data.get('type') == 'pr':
                    # 检查PR类型
                    edge_data = self.graph.edges.get((user_node, neighbor), {})
                    if edge_data.get('contribution_type') == 'code':
                        code_contributions += 1
                    elif edge_data.get('contribution_type') == 'doc':
                        doc_contributions += 1
        
        # 计算各种创新指标
        app_creation_rate = app_connections / max(total_connections, 1)
        code_contribution_rate = code_contributions / max(len(user_nodes), 1)
        knowledge_sharing_rate = doc_contributions / max(len(user_nodes), 1)
        
        # 综合创新得分
        innovation_score = (app_creation_rate * 0.4 + 
                          code_contribution_rate * 0.4 + 
                          knowledge_sharing_rate * 0.2)
        
        return {
            'innovation_score': innovation_score,
            'app_creation_rate': app_creation_rate,
            'code_contribution_rate': code_contribution_rate,
            'knowledge_sharing_rate': knowledge_sharing_rate
        }
    
    def _identify_key_nodes_in_community(self, community: Set) -> List[str]:
        """识别社区内的关键节点（局部领导者）"""
        # 创建子图
        subgraph = self.graph.subgraph(community)
        
        if subgraph.number_of_nodes() == 0:
            return []
        
        # 计算子图内的中心性
        calculator = CentralityCalculator(subgraph)
        
        try:
            degree_centrality = calculator.calculate_degree_centrality()
            betweenness_centrality = calculator.calculate_betweenness_centrality()
            
            # 综合中心性得分
            combined_centrality = {}
            for node in community:
                if node in degree_centrality and node in betweenness_centrality:
                    combined_centrality[node] = (
                        degree_centrality[node] * 0.6 + 
                        betweenness_centrality[node] * 0.4
                    )
            
            # 选择前20%或至少1个节点作为关键节点
            num_key_nodes = max(1, int(len(community) * 0.2))
            key_nodes = sorted(combined_centrality.keys(), 
                             key=lambda x: combined_centrality[x], 
                             reverse=True)[:num_key_nodes]
            
            return key_nodes
            
        except Exception as e:
            logger.warning(f"计算社区关键节点时出错: {e}")
            # 回退方案：选择度最高的节点
            degrees = dict(subgraph.degree())
            if degrees:
                top_node = max(degrees.keys(), key=lambda x: degrees[x])
                return [top_node]
            return []
    
    def _analyze_functional_profile(self, community: Set) -> Dict[str, Any]:
        """分析社区的功能特征"""
        # 统计社区连接的不同类型实体
        connected_repos_by_type = defaultdict(int)
        connected_prs_by_type = defaultdict(int)
        
        for node in community:
            if self.graph.nodes[node].get('type') == 'user':
                for neighbor in self.graph.neighbors(node):
                    neighbor_data = self.graph.nodes[neighbor]
                    if neighbor_data.get('type') == 'repo':
                        repo_type = neighbor_data.get('primary_role', 'unknown')
                        connected_repos_by_type[repo_type] += 1
                    elif neighbor_data.get('type') == 'pr':
                        edge_data = self.graph.edges.get((node, neighbor), {})
                        pr_type = edge_data.get('contribution_type', 'unknown')
                        connected_prs_by_type[pr_type] += 1
        
        # 确定主要功能类型
        total_repo_connections = sum(connected_repos_by_type.values())
        total_pr_connections = sum(connected_prs_by_type.values())
        
        if total_repo_connections == 0 and total_pr_connections == 0:
            return {'primary_function': 'unknown', 'specialization_index': 0.0}
        
        # 计算专业化指数（基尼系数的简化版本）
        all_connections = list(connected_repos_by_type.values()) + list(connected_prs_by_type.values())
        if len(all_connections) > 1:
            max_connections = max(all_connections)
            total_connections = sum(all_connections)
            specialization_index = max_connections / total_connections
        else:
            specialization_index = 1.0
        
        # 确定主要功能
        if connected_repos_by_type['application'] > total_repo_connections * 0.4:
            primary_function = 'application_development'
        elif connected_prs_by_type['code'] > total_pr_connections * 0.5:
            primary_function = 'code_contribution'
        elif connected_prs_by_type['doc'] > total_pr_connections * 0.3:
            primary_function = 'knowledge_sharing'
        elif connected_repos_by_type['library'] > total_repo_connections * 0.3:
            primary_function = 'library_development'
        else:
            primary_function = 'mixed'
        
        return {
            'primary_function': primary_function,
            'specialization_index': specialization_index,
            'repo_connections': dict(connected_repos_by_type),
            'pr_connections': dict(connected_prs_by_type)
        }
    
    def _calculate_external_connectivity(self, community: Set) -> float:
        """计算社区的外部连接度"""
        internal_edges = 0
        external_edges = 0
        
        for node in community:
            for neighbor in self.graph.neighbors(node):
                if neighbor in community:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        total_edges = internal_edges + external_edges
        if total_edges == 0:
            return 0.0
        
        return external_edges / total_edges
    
    def _calculate_internal_density(self, community: Set) -> float:
        """计算社区内部密度"""
        subgraph = self.graph.subgraph(community)
        n = subgraph.number_of_nodes()
        
        if n < 2:
            return 0.0
        
        possible_edges = n * (n - 1) / 2
        actual_edges = subgraph.number_of_edges()
        
        return actual_edges / possible_edges


class SubCenterLifecycleTracker:
    """
    子中心生命周期追踪器
    追踪子中心的涌现、发展、衰退和消亡过程
    """
    
    def __init__(self, 
                 start_date: str,
                 end_date: str,
                 similarity_threshold: float = 0.3,
                 stability_threshold: int = 3):
        """
        初始化生命周期追踪器
        
        Args:
            start_date: 开始日期
            end_date: 结束日期  
            similarity_threshold: 子中心相似度阈值
            stability_threshold: 稳定性阈值（连续存在的月数）
        """
        self.start_date = start_date
        self.end_date = end_date
        self.similarity_threshold = similarity_threshold
        self.stability_threshold = stability_threshold
        
        # 存储历史子中心数据
        self.historical_subcenters = {}
        self.subcenter_lineages = []
        
        logger.info(f"初始化子中心生命周期追踪器: {start_date} 到 {end_date}")
    
    def track_subcenter_evolution(self, 
                                graph: nx.Graph,
                                algorithm: str = 'louvain') -> pd.DataFrame:
        """
        追踪子中心的演化过程
        
        Args:
            graph: 完整网络图
            algorithm: 社区检测算法
            
        Returns:
            pd.DataFrame: 子中心演化历史
        """
        logger.info(f"开始追踪子中心生命周期 ({algorithm} 算法)...")
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        
        for month_start in tqdm(date_range, desc="追踪子中心演化"):
            month_end = month_start + pd.offsets.MonthEnd(0)
            
            # 创建月度快照
            monthly_graph = self._create_monthly_snapshot(graph, month_start, month_end)
            
            if monthly_graph.number_of_nodes() < 10:  # 跳过节点过少的月份
                continue
            
            # 检测社区和子中心
            monthly_subcenters = self._detect_monthly_subcenters(
                monthly_graph, month_start, algorithm
            )
            
            # 与历史子中心进行匹配
            self._match_with_historical_subcenters(monthly_subcenters, month_start)
            
            # 存储当月子中心
            self.historical_subcenters[month_start] = monthly_subcenters
        
        # 分析生命周期模式
        lifecycle_data = self._analyze_lifecycle_patterns()
        
        # 转换为DataFrame并保存
        lifecycle_df = pd.DataFrame(lifecycle_data)
        
        if not lifecycle_df.empty:
            output_path = ANALYSIS_OUTPUT_DIR / f"subcenter_lifecycle_{algorithm}.csv"
            lifecycle_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"子中心生命周期数据已保存至: {output_path}")
        
        return lifecycle_df
    
    def _create_monthly_snapshot(self, 
                               graph: nx.Graph, 
                               start_date: pd.Timestamp, 
                               end_date: pd.Timestamp) -> nx.Graph:
        """创建月度网络快照"""
        monthly_graph = nx.Graph()
        monthly_graph.add_nodes_from(graph.nodes(data=True))
        
        # 添加时间范围内的边
        for u, v, data in graph.edges(data=True):
            if 'timestamp' in data:
                try:
                    ts = pd.to_datetime(data['timestamp']).tz_localize(None)
                    if start_date <= ts <= end_date:
                        monthly_graph.add_edge(u, v, **data)
                except (ValueError, TypeError):
                    continue
        
        # 移除孤立节点
        monthly_graph.remove_nodes_from(list(nx.isolates(monthly_graph)))
        
        return monthly_graph
    
    def _detect_monthly_subcenters(self, 
                                 graph: nx.Graph, 
                                 month: pd.Timestamp,
                                 algorithm: str) -> List[Dict]:
        """检测当月的子中心"""
        if graph.number_of_nodes() == 0:
            return []
        
        # 社区检测
        detector = CommunityDetector(graph)
        
        if algorithm == 'louvain':
            communities = detector.detect_louvain_communities()
        elif algorithm == 'leiden' and LEIDEN_AVAILABLE:
            communities = detector.detect_leiden_communities()
        elif algorithm == 'infomap' and INFOMAP_AVAILABLE:
            communities = detector.detect_infomap_communities()
        else:
            return []
        
        # 子中心识别
        subcenter_detector = SubCenterDetector(graph)
        subcenters = subcenter_detector.identify_sub_centers(communities, algorithm)
        
        # 为每个子中心添加时间戳
        for subcenter in subcenters:
            subcenter['month'] = month
            subcenter['month_str'] = month.strftime('%Y-%m')
        
        return subcenters
    
    def _match_with_historical_subcenters(self, 
                                        current_subcenters: List[Dict], 
                                        current_month: pd.Timestamp):
        """将当前子中心与历史子中心进行匹配"""
        if not self.historical_subcenters:
            # 第一个月，所有子中心都是新的
            for i, subcenter in enumerate(current_subcenters):
                lineage_id = f"lineage_{len(self.subcenter_lineages)}"
                subcenter['lineage_id'] = lineage_id
                subcenter['lifecycle_stage'] = 'emergence'
                
                # 创建新的家族谱系
                self.subcenter_lineages.append({
                    'lineage_id': lineage_id,
                    'birth_month': current_month,
                    'death_month': None,
                    'peak_month': current_month,
                    'peak_size': subcenter['size'],
                    'total_lifespan': 1,
                    'history': [subcenter.copy()]
                })
            return
        
        # 获取上个月的子中心
        previous_months = [m for m in self.historical_subcenters.keys() if m < current_month]
        if not previous_months:
            return
        
        last_month = max(previous_months)
        previous_subcenters = self.historical_subcenters[last_month]
        
        # 计算相似度矩阵
        similarity_matrix = self._calculate_similarity_matrix(
            current_subcenters, previous_subcenters
        )
        
        # 匹配算法（贪心匹配）
        matched_pairs = []
        used_current = set()
        used_previous = set()
        
        # 按相似度从高到低排序
        similarities = []
        for i, current in enumerate(current_subcenters):
            for j, previous in enumerate(previous_subcenters):
                similarity = similarity_matrix[i][j]
                if similarity >= self.similarity_threshold:
                    similarities.append((similarity, i, j))
        
        similarities.sort(reverse=True)
        
        # 执行匹配
        for similarity, i, j in similarities:
            if i not in used_current and j not in used_previous:
                matched_pairs.append((i, j, similarity))
                used_current.add(i)
                used_previous.add(j)
        
        # 更新匹配的子中心
        for current_idx, previous_idx, similarity in matched_pairs:
            current_subcenter = current_subcenters[current_idx]
            previous_subcenter = previous_subcenters[previous_idx]
            
            # 继承家族谱系ID
            lineage_id = previous_subcenter['lineage_id']
            current_subcenter['lineage_id'] = lineage_id
            current_subcenter['similarity_to_previous'] = similarity
            
            # 更新生命周期阶段
            lineage = next(l for l in self.subcenter_lineages if l['lineage_id'] == lineage_id)
            
            if current_subcenter['size'] > previous_subcenter['size']:
                current_subcenter['lifecycle_stage'] = 'growth'
                if current_subcenter['size'] > lineage['peak_size']:
                    lineage['peak_month'] = current_month
                    lineage['peak_size'] = current_subcenter['size']
            elif current_subcenter['size'] < previous_subcenter['size']:
                current_subcenter['lifecycle_stage'] = 'decline'
            else:
                current_subcenter['lifecycle_stage'] = 'stable'
            
            # 更新家族谱系历史
            lineage['total_lifespan'] += 1
            lineage['history'].append(current_subcenter.copy())
        
        # 处理新出现的子中心
        for i, current_subcenter in enumerate(current_subcenters):
            if i not in used_current:
                lineage_id = f"lineage_{len(self.subcenter_lineages)}"
                current_subcenter['lineage_id'] = lineage_id
                current_subcenter['lifecycle_stage'] = 'emergence'
                current_subcenter['similarity_to_previous'] = 0.0
                
                # 创建新的家族谱系
                self.subcenter_lineages.append({
                    'lineage_id': lineage_id,
                    'birth_month': current_month,
                    'death_month': None,
                    'peak_month': current_month,
                    'peak_size': current_subcenter['size'],
                    'total_lifespan': 1,
                    'history': [current_subcenter.copy()]
                })
        
        # 标记消失的子中心
        for j, previous_subcenter in enumerate(previous_subcenters):
            if j not in used_previous:
                lineage_id = previous_subcenter['lineage_id']
                lineage = next(l for l in self.subcenter_lineages if l['lineage_id'] == lineage_id)
                if lineage['death_month'] is None:  # 只标记一次死亡
                    lineage['death_month'] = current_month
    
    def _calculate_similarity_matrix(self, 
                                   current_subcenters: List[Dict], 
                                   previous_subcenters: List[Dict]) -> List[List[float]]:
        """计算子中心之间的相似度矩阵"""
        similarity_matrix = []
        
        for current in current_subcenters:
            current_similarities = []
            current_nodes = set(current['nodes'])
            
            for previous in previous_subcenters:
                previous_nodes = set(previous['nodes'])
                
                # Jaccard相似度
                intersection = len(current_nodes.intersection(previous_nodes))
                union = len(current_nodes.union(previous_nodes))
                
                if union == 0:
                    jaccard_similarity = 0.0
                else:
                    jaccard_similarity = intersection / union
                
                # 功能相似度
                function_similarity = 1.0 if current['functional_type'] == previous['functional_type'] else 0.5
                
                # 综合相似度
                overall_similarity = jaccard_similarity * 0.7 + function_similarity * 0.3
                
                current_similarities.append(overall_similarity)
            
            similarity_matrix.append(current_similarities)
        
        return similarity_matrix
    
    def _analyze_lifecycle_patterns(self) -> List[Dict]:
        """分析生命周期模式"""
        lifecycle_data = []
        
        for lineage in self.subcenter_lineages:
            # 基本生命周期信息
            birth_month = lineage['birth_month']
            death_month = lineage['death_month']
            peak_month = lineage['peak_month']
            total_lifespan = lineage['total_lifespan']
            
            # 生命周期阶段统计
            stage_counts = defaultdict(int)
            size_history = []
            innovation_history = []
            
            for snapshot in lineage['history']:
                stage_counts[snapshot['lifecycle_stage']] += 1
                size_history.append(snapshot['size'])
                innovation_history.append(snapshot['innovation_score'])
            
            # 计算生命周期特征
            avg_size = np.mean(size_history) if size_history else 0
            max_size = max(size_history) if size_history else 0
            size_volatility = np.std(size_history) if len(size_history) > 1 else 0
            
            avg_innovation = np.mean(innovation_history) if innovation_history else 0
            max_innovation = max(innovation_history) if innovation_history else 0
            
            # 生命周期类型分类
            if total_lifespan == 1:
                lifecycle_type = 'ephemeral'  # 昙花一现
            elif total_lifespan < self.stability_threshold:
                lifecycle_type = 'short_lived'  # 短命
            elif death_month is None:
                lifecycle_type = 'persistent'  # 持续存在
            else:
                lifecycle_type = 'completed'  # 完整生命周期
            
            # 主导功能类型
            if lineage['history']:
                functional_types = [h['functional_type'] for h in lineage['history']]
                dominant_function = max(set(functional_types), key=functional_types.count)
            else:
                dominant_function = 'unknown'
            
            lifecycle_record = {
                'lineage_id': lineage['lineage_id'],
                'lifecycle_type': lifecycle_type,
                'birth_month': birth_month.strftime('%Y-%m'),
                'death_month': death_month.strftime('%Y-%m') if death_month else None,
                'peak_month': peak_month.strftime('%Y-%m'),
                'total_lifespan_months': total_lifespan,
                'avg_size': avg_size,
                'max_size': max_size,
                'peak_size': lineage['peak_size'],
                'size_volatility': size_volatility,
                'avg_innovation_score': avg_innovation,
                'max_innovation_score': max_innovation,
                'dominant_functional_type': dominant_function,
                'emergence_count': stage_counts['emergence'],
                'growth_count': stage_counts['growth'],
                'stable_count': stage_counts['stable'],
                'decline_count': stage_counts['decline'],
                'is_stable': total_lifespan >= self.stability_threshold,
                'final_status': 'active' if death_month is None else 'extinct'
            }
            
            lifecycle_data.append(lifecycle_record)
        
        return lifecycle_data
    
    def visualize_lifecycle_patterns(self, lifecycle_df: pd.DataFrame):
        """可视化生命周期模式"""
        if lifecycle_df.empty:
            logger.warning("没有生命周期数据可视化")
            return
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 生命周期类型分布
        lifecycle_counts = lifecycle_df['lifecycle_type'].value_counts()
        axes[0,0].pie(lifecycle_counts.values, labels=lifecycle_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Distribution of Lifecycle Types', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        
        # 2. 寿命分布
        axes[0,1].hist(lifecycle_df['total_lifespan_months'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Distribution of Subcenter Lifespan', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,1].set_xlabel('Lifespan (months)')
        axes[0,1].set_ylabel('Count')
        
        # 3. 规模与创新得分关系
        axes[1,0].scatter(lifecycle_df['avg_size'], lifecycle_df['avg_innovation_score'], alpha=0.6)
        axes[1,0].set_title('Size vs Innovation Score', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,0].set_xlabel('Average Size')
        axes[1,0].set_ylabel('Average Innovation Score')
        
        # 4. 功能类型分布
        function_counts = lifecycle_df['dominant_functional_type'].value_counts()
        axes[1,1].bar(function_counts.index, function_counts.values)
        axes[1,1].set_title('Distribution of Functional Types', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,1].set_xlabel('Functional Type')
        axes[1,1].set_ylabel('Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "subcenter_lifecycle_patterns.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"生命周期模式图已保存至: {save_path}")
        
        plt.show()
    
    def generate_lifecycle_summary(self, lifecycle_df: pd.DataFrame) -> Dict[str, Any]:
        """生成生命周期摘要统计"""
        if lifecycle_df.empty:
            return {}
        
        summary = {
            'total_subcenters': len(lifecycle_df),
            'active_subcenters': len(lifecycle_df[lifecycle_df['final_status'] == 'active']),
            'extinct_subcenters': len(lifecycle_df[lifecycle_df['final_status'] == 'extinct']),
            'stable_subcenters': len(lifecycle_df[lifecycle_df['is_stable'] == True]),
            'avg_lifespan': lifecycle_df['total_lifespan_months'].mean(),
            'max_lifespan': lifecycle_df['total_lifespan_months'].max(),
            'avg_peak_size': lifecycle_df['peak_size'].mean(),
            'most_common_function': lifecycle_df['dominant_functional_type'].mode().iloc[0] if not lifecycle_df['dominant_functional_type'].mode().empty else 'unknown',
            'lifecycle_type_distribution': lifecycle_df['lifecycle_type'].value_counts().to_dict(),
            'functional_type_distribution': lifecycle_df['dominant_functional_type'].value_counts().to_dict()
        }
        
        return summary


def main():
    """主函数入口"""
    from config.settings import NETWORK_OUTPUT_DIR, FILENAMES
    
    # 加载网络图
    graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
    try:
        G = nx.read_graphml(str(graph_path))
        logger.info(f"成功加载网络图，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    except FileNotFoundError:
        logger.error(f"未找到网络图文件: {graph_path}")
        return
    
    # 静态社区检测比较
    logger.info("开始静态社区检测算法比较...")
    detector = CommunityDetector(G)
    comparison_df = detector.compare_algorithms()
    detector.visualize_algorithm_comparison(comparison_df)
    
    # 动态社区分析
    logger.info("开始动态社区分析...")
    from config.settings import ANALYSIS_CONFIG
    
    dynamic_analyzer = DynamicCommunityAnalyzer(
        G, 
        ANALYSIS_CONFIG["start_date"], 
        ANALYSIS_CONFIG["end_date"],
        algorithms=['louvain', 'leiden'] if LEIDEN_AVAILABLE else ['louvain']
    )
    
    dynamic_results = dynamic_analyzer.analyze_temporal_communities()
    dynamic_analyzer.visualize_temporal_trends(dynamic_results)
    
    logger.info("社区检测分析完成！")


def analyze_subcenters_and_lifecycle():
    """子中心识别和生命周期追踪的主函数"""
    from config.settings import NETWORK_OUTPUT_DIR, FILENAMES, ANALYSIS_CONFIG
    
    # 加载网络图
    graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
    try:
        G = nx.read_graphml(str(graph_path))
        logger.info(f"成功加载网络图，节点数: {G.number_of_nodes()}, 边数: {G.number_of_edges()}")
    except FileNotFoundError:
        logger.error(f"未找到网络图文件: {graph_path}")
        return
    
    # === 第一部分：静态子中心识别 ===
    logger.info("=== 开始静态子中心识别分析 ===")
    
    # 使用Louvain算法进行社区检测
    detector = CommunityDetector(G)
    communities = detector.detect_louvain_communities()
    logger.info(f"检测到 {len(communities)} 个社区")
    
    # 识别子中心
    subcenter_detector = SubCenterDetector(
        G, 
        core_team_logins=ANALYSIS_CONFIG["core_team_logins"],
        min_subcenter_size=8,  # 调整最小规模
        innovation_threshold=0.05  # 调整创新阈值
    )
    
    subcenters = subcenter_detector.identify_sub_centers(communities, 'louvain')
    logger.info(f"识别出 {len(subcenters)} 个子中心")
    
    # 保存静态子中心结果
    if subcenters:
        subcenters_df = pd.DataFrame(subcenters)
        static_output_path = ANALYSIS_OUTPUT_DIR / "static_subcenters_analysis.csv"
        subcenters_df.to_csv(static_output_path, index=False, encoding='utf-8-sig')
        logger.info(f"静态子中心分析结果已保存至: {static_output_path}")
        
        # 打印子中心摘要
        print("\n=== 静态子中心分析摘要 ===")
        print(f"总子中心数: {len(subcenters)}")
        
        function_counts = {}
        for sc in subcenters:
            func_type = sc['functional_type']
            function_counts[func_type] = function_counts.get(func_type, 0) + 1
        
        print("功能类型分布:")
        for func_type, count in function_counts.items():
            print(f"  - {func_type}: {count}")
        
        avg_size = sum(sc['size'] for sc in subcenters) / len(subcenters)
        avg_innovation = sum(sc['innovation_score'] for sc in subcenters) / len(subcenters)
        print(f"平均规模: {avg_size:.1f}")
        print(f"平均创新得分: {avg_innovation:.3f}")
    else:
        print("\n⚠️  未识别出符合条件的子中心")
    
    # === 第二部分：动态生命周期追踪 ===
    logger.info("\n=== 开始子中心生命周期追踪 ===")
    
    # 创建生命周期追踪器
    lifecycle_tracker = SubCenterLifecycleTracker(
        start_date=ANALYSIS_CONFIG["start_date"],
        end_date=ANALYSIS_CONFIG["end_date"],
        similarity_threshold=0.3,
        stability_threshold=3
    )
    
    # 追踪生命周期演化
    lifecycle_df = lifecycle_tracker.track_subcenter_evolution(G, algorithm='louvain')
    
    if not lifecycle_df.empty:
        logger.info(f"追踪到 {len(lifecycle_df)} 个子中心家族谱系")
        
        # 生成摘要统计
        summary = lifecycle_tracker.generate_lifecycle_summary(lifecycle_df)
        
        print("\n=== 子中心生命周期摘要 ===")
        print(f"总子中心家族数: {summary['total_subcenters']}")
        print(f"当前活跃: {summary['active_subcenters']}")
        print(f"已消失: {summary['extinct_subcenters']}")
        print(f"稳定子中心: {summary['stable_subcenters']}")
        print(f"平均寿命: {summary['avg_lifespan']:.1f} 个月")
        print(f"最长寿命: {summary['max_lifespan']} 个月")
        print(f"平均峰值规模: {summary['avg_peak_size']:.1f}")
        print(f"最常见功能: {summary['most_common_function']}")
        
        print("\n生命周期类型分布:")
        for lc_type, count in summary['lifecycle_type_distribution'].items():
            print(f"  - {lc_type}: {count}")
        
        # 可视化生命周期模式
        lifecycle_tracker.visualize_lifecycle_patterns(lifecycle_df)
        
        # 保存详细的生命周期摘要
        import json
        summary_path = ANALYSIS_OUTPUT_DIR / "subcenter_lifecycle_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"生命周期摘要已保存至: {summary_path}")
        
    else:
        print("\n⚠️  未能追踪到子中心生命周期数据")
    
    logger.info("\n=== 子中心识别和生命周期追踪分析完成！ ===")


if __name__ == "__main__":
    main()
