"""
角色演化分析模块
实现用户角色的时间演化分析和共生关系量化
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Set
import logging
from collections import defaultdict, Counter
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from ..utils.logging_config import setup_logger
from .role_clustering import MultiAlgorithmClustering
from config.settings import (
    ANALYSIS_OUTPUT_DIR, FINAL_ANALYSIS_DATA_DIR, 
    VISUALIZATION_CONFIG, ANALYSIS_CONFIG
)

# 设置日志
logger = setup_logger(__name__)


class RoleEvolutionAnalyzer:
    """角色演化分析器"""
    
    def __init__(self, 
                 time_periods: List[Tuple[str, str]] = None,
                 clustering_algorithm: str = 'kmeans',
                 n_clusters: int = 6):
        """
        初始化角色演化分析器
        
        Args:
            time_periods: 时间段列表，每个元素为(开始日期, 结束日期)
            clustering_algorithm: 聚类算法
            n_clusters: 聚类数量
        """
        self.clustering_algorithm = clustering_algorithm
        self.n_clusters = n_clusters
        
        # 如果未提供时间段，则自动生成半年期时间段
        if time_periods is None:
            self.time_periods = self._generate_time_periods()
        else:
            self.time_periods = time_periods
        
        # 存储各时期的聚类结果
        self.period_clustering_results = {}
        self.role_transition_matrix = None
        self.role_networks = {}
        
        logger.info(f"初始化角色演化分析器: {len(self.time_periods)} 个时间段")
    
    def _generate_time_periods(self) -> List[Tuple[str, str]]:
        """生成半年期时间段"""
        start_date = pd.to_datetime(ANALYSIS_CONFIG["start_date"])
        end_date = pd.to_datetime(ANALYSIS_CONFIG["end_date"])
        
        periods = []
        current_start = start_date
        
        while current_start < end_date:
            # 半年期结束日期
            current_end = min(current_start + pd.DateOffset(months=6) - pd.Timedelta(days=1), end_date)
            periods.append((current_start.strftime('%Y-%m-%d'), current_end.strftime('%Y-%m-%d')))
            current_start = current_end + pd.Timedelta(days=1)
        
        return periods
    
    def extract_user_features_by_period(self, period_start: str, period_end: str) -> pd.DataFrame:
        """
        提取指定时间段的用户行为特征
        
        Args:
            period_start: 时间段开始日期
            period_end: 时间段结束日期
            
        Returns:
            pd.DataFrame: 用户特征数据框
        """
        logger.info(f"提取时间段 {period_start} 到 {period_end} 的用户特征...")
        
        try:
            # 加载原始数据
            prs_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "prs.csv")
            issues_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "issues.csv")
            comments_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "comments.csv")
            stars_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "stars.csv")
            users_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "users.csv")
            
            # 转换时间列
            prs_df['created_at'] = pd.to_datetime(prs_df['created_at'])
            issues_df['created_at'] = pd.to_datetime(issues_df['created_at'])
            comments_df['created_at'] = pd.to_datetime(comments_df['created_at'])
            stars_df['starred_at'] = pd.to_datetime(stars_df['starred_at'])
            
            period_start_dt = pd.to_datetime(period_start)
            period_end_dt = pd.to_datetime(period_end)
            
            # 过滤时间段内的数据
            period_prs = prs_df[
                (prs_df['created_at'] >= period_start_dt) & 
                (prs_df['created_at'] <= period_end_dt)
            ]
            period_issues = issues_df[
                (issues_df['created_at'] >= period_start_dt) & 
                (issues_df['created_at'] <= period_end_dt)
            ]
            period_comments = comments_df[
                (comments_df['created_at'] >= period_start_dt) & 
                (comments_df['created_at'] <= period_end_dt)
            ]
            period_stars = stars_df[
                (stars_df['starred_at'] >= period_start_dt) & 
                (stars_df['starred_at'] <= period_end_dt)
            ]
            
            # 获取活跃用户列表
            active_users = set()
            active_users.update(period_prs['author_id'].dropna())
            active_users.update(period_issues['author_id'].dropna())
            active_users.update(period_comments['author_id'].dropna())
            active_users.update(period_stars['user_id'].dropna())
            
            # 计算用户特征
            user_features = []
            
            for user_id in tqdm(active_users, desc="计算用户特征"):
                # PR相关特征
                user_prs = period_prs[period_prs['author_id'] == user_id]
                pr_count = len(user_prs)
                code_pr_count = len(user_prs[user_prs['contribution_type'] == 'code'])
                doc_pr_count = len(user_prs[user_prs['contribution_type'] == 'doc'])
                
                # Issue相关特征
                user_issues = period_issues[period_issues['author_id'] == user_id]
                issue_count = len(user_issues)
                
                # Comment相关特征
                user_comments = period_comments[period_comments['author_id'] == user_id]
                comment_count = len(user_comments)
                
                # Star相关特征
                user_stars = period_stars[period_stars['user_id'] == user_id]
                star_count = len(user_stars)
                
                # 获取用户login
                user_info = users_df[users_df['user_id'] == user_id]
                user_login = user_info['login'].iloc[0] if not user_info.empty else str(user_id)
                
                # 计算复合特征
                total_contributions = pr_count + issue_count + comment_count
                code_focus_ratio = code_pr_count / max(pr_count, 1)
                interaction_diversity = len([x for x in [pr_count, issue_count, comment_count, star_count] if x > 0])
                
                user_features.append({
                    'user_id': user_id,
                    'login': user_login,
                    'pr_count': pr_count,
                    'code_pr_count': code_pr_count,
                    'doc_pr_count': doc_pr_count,
                    'issue_count': issue_count,
                    'comment_count': comment_count,
                    'star_count': star_count,
                    'total_contributions': total_contributions,
                    'code_focus_ratio': code_focus_ratio,
                    'interaction_diversity': interaction_diversity,
                    'period_start': period_start,
                    'period_end': period_end
                })
            
            return pd.DataFrame(user_features)
            
        except Exception as e:
            logger.error(f"提取用户特征失败: {e}")
            return pd.DataFrame()
    
    def perform_periodic_clustering(self) -> Dict[str, Dict]:
        """
        对每个时间段执行聚类分析
        
        Returns:
            Dict[str, Dict]: 每个时间段的聚类结果
        """
        logger.info("对各时间段执行聚类分析...")
        
        self.period_clustering_results = {}
        
        for i, (period_start, period_end) in enumerate(self.time_periods):
            period_key = f"period_{i}_{period_start}_{period_end}"
            logger.info(f"处理时间段 {period_key}")
            
            # 提取用户特征
            features_df = self.extract_user_features_by_period(period_start, period_end)
            
            if features_df.empty:
                logger.warning(f"时间段 {period_key} 没有用户特征数据")
                continue
            
            # 过滤特征列
            feature_columns = [col for col in features_df.columns 
                             if col not in ['user_id', 'login', 'period_start', 'period_end']]
            
            if len(features_df) < self.n_clusters:
                logger.warning(f"时间段 {period_key} 用户数({len(features_df)})少于聚类数({self.n_clusters})")
                continue
            
            # 执行聚类
            try:
                clusterer = MultiAlgorithmClustering(features_df, feature_columns)
                
                if self.clustering_algorithm == 'kmeans':
                    cluster_labels = clusterer.perform_kmeans_clustering(self.n_clusters)
                elif self.clustering_algorithm == 'gmm':
                    cluster_labels = clusterer.perform_gmm_clustering(self.n_clusters)
                elif self.clustering_algorithm == 'hierarchical':
                    cluster_labels = clusterer.perform_hierarchical_clustering(self.n_clusters)
                else:
                    logger.error(f"不支持的聚类算法: {self.clustering_algorithm}")
                    continue
                
                # 存储结果
                result_df = features_df.copy()
                result_df['cluster'] = cluster_labels
                
                self.period_clustering_results[period_key] = {
                    'features_df': features_df,
                    'result_df': result_df,
                    'clusterer': clusterer,
                    'period_start': period_start,
                    'period_end': period_end,
                    'period_index': i
                }
                
                logger.info(f"时间段 {period_key} 聚类完成: {len(features_df)} 用户, {self.n_clusters} 个聚类")
                
            except Exception as e:
                logger.error(f"时间段 {period_key} 聚类失败: {e}")
        
        return self.period_clustering_results
    
    def analyze_role_transitions(self) -> pd.DataFrame:
        """
        分析角色转移模式
        
        Returns:
            pd.DataFrame: 角色转移数据框
        """
        logger.info("分析角色转移模式...")
        
        if len(self.period_clustering_results) < 2:
            logger.error("需要至少2个时间段的聚类结果才能分析角色转移")
            return pd.DataFrame()
        
        # 获取有序的时间段
        sorted_periods = sorted(self.period_clustering_results.keys(), 
                              key=lambda x: self.period_clustering_results[x]['period_index'])
        
        transitions = []
        
        # 分析相邻时间段之间的转移
        for i in range(len(sorted_periods) - 1):
            period1_key = sorted_periods[i]
            period2_key = sorted_periods[i + 1]
            
            period1_data = self.period_clustering_results[period1_key]['result_df']
            period2_data = self.period_clustering_results[period2_key]['result_df']
            
            # 找到两个时期都活跃的用户
            common_users = set(period1_data['user_id']) & set(period2_data['user_id'])
            
            logger.info(f"时间段 {period1_key} -> {period2_key}: {len(common_users)} 个共同用户")
            
            for user_id in common_users:
                user1_data = period1_data[period1_data['user_id'] == user_id].iloc[0]
                user2_data = period2_data[period2_data['user_id'] == user_id].iloc[0]
                
                transitions.append({
                    'user_id': user_id,
                    'login': user1_data['login'],
                    'period1_start': user1_data['period_start'],
                    'period1_end': user1_data['period_end'],
                    'period2_start': user2_data['period_start'],
                    'period2_end': user2_data['period_end'],
                    'role_from': user1_data['cluster'],
                    'role_to': user2_data['cluster'],
                    'transition_type': 'stable' if user1_data['cluster'] == user2_data['cluster'] else 'change'
                })
        
        transitions_df = pd.DataFrame(transitions)
        
        if not transitions_df.empty:
            # 计算转移矩阵
            self.role_transition_matrix = self._calculate_transition_matrix(transitions_df)
            
            # 保存转移数据
            output_path = ANALYSIS_OUTPUT_DIR / "role_transitions.csv"
            transitions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"角色转移数据已保存至: {output_path}")
        
        return transitions_df
    
    def _calculate_transition_matrix(self, transitions_df: pd.DataFrame) -> pd.DataFrame:
        """计算角色转移矩阵"""
        # 创建转移计数矩阵
        all_roles = list(range(self.n_clusters))
        transition_counts = pd.DataFrame(0, index=all_roles, columns=all_roles)
        
        for _, row in transitions_df.iterrows():
            role_from = row['role_from']
            role_to = row['role_to']
            transition_counts.loc[role_from, role_to] += 1
        
        # 转换为概率矩阵
        transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
        
        # 保存转移矩阵
        output_path = ANALYSIS_OUTPUT_DIR / "role_transition_matrix.csv"
        transition_matrix.to_csv(output_path, encoding='utf-8-sig')
        logger.info(f"角色转移矩阵已保存至: {output_path}")
        
        return transition_matrix
    
    def visualize_role_transitions(self, transitions_df: pd.DataFrame):
        """可视化角色转移"""
        if transitions_df.empty:
            logger.warning("没有转移数据可视化")
            return
        
        # 1. 转移矩阵热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.role_transition_matrix, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='Blues',
                   square=True)
        plt.title('Role Transition Probability Matrix', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.xlabel('Role To')
        plt.ylabel('Role From')
        
        save_path = ANALYSIS_OUTPUT_DIR / "role_transition_matrix_heatmap.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.show()
        
        # 2. 桑基图 (Sankey Diagram)
        self._create_sankey_diagram(transitions_df)
        
        # 3. 转移类型分布
        plt.figure(figsize=(10, 6))
        transition_counts = transitions_df['transition_type'].value_counts()
        plt.pie(transition_counts.values, labels=transition_counts.index, autopct='%1.1f%%')
        plt.title('Distribution of Transition Types', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        
        save_path = ANALYSIS_OUTPUT_DIR / "transition_types_distribution.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        plt.show()
    
    def _create_sankey_diagram(self, transitions_df: pd.DataFrame):
        """创建桑基图"""
        try:
            # 准备桑基图数据
            source = []
            target = []
            value = []
            
            # 计算角色间的转移流量
            role_transitions = transitions_df.groupby(['role_from', 'role_to']).size().reset_index(name='count')
            
            for _, row in role_transitions.iterrows():
                source.append(row['role_from'])
                target.append(row['role_to'] + self.n_clusters)  # 目标角色ID偏移
                value.append(row['count'])
            
            # 创建节点标签
            node_labels = []
            for i in range(self.n_clusters):
                node_labels.append(f"Role {i} (T1)")
            for i in range(self.n_clusters):
                node_labels.append(f"Role {i} (T2)")
            
            # 创建桑基图
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=node_labels
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value
                )
            )])
            
            fig.update_layout(
                title_text="Role Transition Flow (Sankey Diagram)",
                font_size=12
            )
            
            # 保存桑基图
            save_path = ANALYSIS_OUTPUT_DIR / "role_transition_sankey.html"
            fig.write_html(str(save_path))
            logger.info(f"桑基图已保存至: {save_path}")
            
            fig.show()
            
        except Exception as e:
            logger.error(f"创建桑基图失败: {e}")
    
    def analyze_role_symbiosis(self) -> pd.DataFrame:
        """
        分析角色共生关系
        
        Returns:
            pd.DataFrame: 角色交互网络数据
        """
        logger.info("分析角色共生关系...")
        
        # 加载交互数据
        try:
            interaction_data = self._load_interaction_data()
            if interaction_data.empty:
                logger.error("无法加载交互数据")
                return pd.DataFrame()
            
            # 为每个时间段构建角色交互网络
            self.role_networks = {}
            symbiosis_data = []
            
            for period_key, period_result in self.period_clustering_results.items():
                period_network = self._build_role_interaction_network(
                    period_result['result_df'], 
                    interaction_data,
                    period_result['period_start'],
                    period_result['period_end']
                )
                
                self.role_networks[period_key] = period_network
                
                # 计算网络指标
                network_metrics = self._calculate_network_metrics(period_network)
                network_metrics['period'] = period_key
                symbiosis_data.append(network_metrics)
            
            symbiosis_df = pd.DataFrame(symbiosis_data)
            
            # 保存共生关系数据
            output_path = ANALYSIS_OUTPUT_DIR / "role_symbiosis_analysis.csv"
            symbiosis_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"角色共生关系分析已保存至: {output_path}")
            
            return symbiosis_df
            
        except Exception as e:
            logger.error(f"角色共生关系分析失败: {e}")
            return pd.DataFrame()
    
    def _load_interaction_data(self) -> pd.DataFrame:
        """加载用户交互数据"""
        try:
            comments_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "comments.csv")
            prs_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "prs.csv")
            issues_df = pd.read_csv(FINAL_ANALYSIS_DATA_DIR / "issues.csv")
            
            interactions = []
            
            # PR作者与评论者的交互
            pr_comments = comments_df[comments_df['parent_type'] == 'pr']
            for _, comment in pr_comments.iterrows():
                pr_info = prs_df[prs_df['pr_id'] == comment['parent_id']]
                if not pr_info.empty:
                    pr_author = pr_info.iloc[0]['author_id']
                    commenter = comment['author_id']
                    if pr_author != commenter:  # 排除自评论
                        interactions.append({
                            'user1': pr_author,
                            'user2': commenter,
                            'interaction_type': 'pr_comment',
                            'timestamp': comment['created_at']
                        })
            
            # Issue作者与评论者的交互
            issue_comments = comments_df[comments_df['parent_type'] == 'issue']
            for _, comment in issue_comments.iterrows():
                issue_info = issues_df[issues_df['issue_id'] == comment['parent_id']]
                if not issue_info.empty:
                    issue_author = issue_info.iloc[0]['author_id']
                    commenter = comment['author_id']
                    if issue_author != commenter:  # 排除自评论
                        interactions.append({
                            'user1': issue_author,
                            'user2': commenter,
                            'interaction_type': 'issue_comment',
                            'timestamp': comment['created_at']
                        })
            
            return pd.DataFrame(interactions)
            
        except Exception as e:
            logger.error(f"加载交互数据失败: {e}")
            return pd.DataFrame()
    
    def _build_role_interaction_network(self, 
                                      role_assignments: pd.DataFrame,
                                      interaction_data: pd.DataFrame,
                                      period_start: str,
                                      period_end: str) -> nx.Graph:
        """构建角色交互网络"""
        # 过滤时间段内的交互
        interaction_data['timestamp'] = pd.to_datetime(interaction_data['timestamp'])
        period_start_dt = pd.to_datetime(period_start)
        period_end_dt = pd.to_datetime(period_end)
        
        period_interactions = interaction_data[
            (interaction_data['timestamp'] >= period_start_dt) & 
            (interaction_data['timestamp'] <= period_end_dt)
        ]
        
        # 创建用户到角色的映射
        user_to_role = dict(zip(role_assignments['user_id'], role_assignments['cluster']))
        
        # 构建角色间的交互网络
        role_network = nx.Graph()
        
        # 添加所有角色节点
        for role in range(self.n_clusters):
            role_network.add_node(role)
        
        # 添加角色间的交互边
        role_interactions = defaultdict(int)
        
        for _, interaction in period_interactions.iterrows():
            user1 = interaction['user1']
            user2 = interaction['user2']
            
            if user1 in user_to_role and user2 in user_to_role:
                role1 = user_to_role[user1]
                role2 = user_to_role[user2]
                
                if role1 != role2:  # 只考虑不同角色间的交互
                    role_pair = tuple(sorted([role1, role2]))
                    role_interactions[role_pair] += 1
        
        # 添加加权边
        for (role1, role2), weight in role_interactions.items():
            role_network.add_edge(role1, role2, weight=weight)
        
        return role_network
    
    def _calculate_network_metrics(self, role_network: nx.Graph) -> Dict:
        """计算角色网络指标"""
        metrics = {}
        
        # 基本网络指标
        metrics['num_nodes'] = role_network.number_of_nodes()
        metrics['num_edges'] = role_network.number_of_edges()
        metrics['density'] = nx.density(role_network)
        
        # 中心性指标
        if role_network.number_of_edges() > 0:
            degree_centrality = nx.degree_centrality(role_network)
            betweenness_centrality = nx.betweenness_centrality(role_network)
            
            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
            metrics['max_degree_centrality'] = np.max(list(degree_centrality.values()))
            metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
            metrics['max_betweenness_centrality'] = np.max(list(betweenness_centrality.values()))
        else:
            metrics.update({
                'avg_degree_centrality': 0,
                'max_degree_centrality': 0,
                'avg_betweenness_centrality': 0,
                'max_betweenness_centrality': 0
            })
        
        # 连通性指标
        metrics['is_connected'] = nx.is_connected(role_network)
        metrics['num_components'] = nx.number_connected_components(role_network)
        
        return metrics
    
    def run_complete_analysis(self) -> Dict:
        """
        运行完整的角色演化分析
        
        Returns:
            Dict: 完整的分析结果
        """
        logger.info("开始完整的角色演化分析...")
        
        # 1. 各时间段聚类分析
        periodic_results = self.perform_periodic_clustering()
        
        if len(periodic_results) < 2:
            logger.error("需要至少2个时间段的聚类结果")
            return {}
        
        # 2. 角色转移分析
        transitions_df = self.analyze_role_transitions()
        
        # 3. 角色共生关系分析
        symbiosis_df = self.analyze_role_symbiosis()
        
        # 4. 可视化结果
        if not transitions_df.empty:
            self.visualize_role_transitions(transitions_df)
        
        # 5. 生成分析报告
        analysis_report = {
            'num_periods': len(periodic_results),
            'time_periods': self.time_periods,
            'num_transitions': len(transitions_df) if not transitions_df.empty else 0,
            'transition_matrix': self.role_transition_matrix.to_dict() if self.role_transition_matrix is not None else {},
            'role_networks': {k: {'nodes': v.number_of_nodes(), 'edges': v.number_of_edges()} 
                            for k, v in self.role_networks.items()},
            'clustering_algorithm': self.clustering_algorithm,
            'n_clusters': self.n_clusters
        }
        
        # 保存分析报告
        import json
        report_path = ANALYSIS_OUTPUT_DIR / "role_evolution_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_report, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"角色演化分析报告已保存至: {report_path}")
        logger.info("角色演化分析完成！")
        
        return analysis_report


def main():
    """主函数入口"""
    # 确保输出目录存在
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建角色演化分析器
    analyzer = RoleEvolutionAnalyzer(
        clustering_algorithm='kmeans',
        n_clusters=6
    )
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    if results:
        logger.info("角色演化分析成功完成！")
        print(f"分析了 {results['num_periods']} 个时间段")
        print(f"发现 {results['num_transitions']} 个角色转移")
        print(f"使用算法: {results['clustering_algorithm']}")
    else:
        logger.error("角色演化分析失败")


if __name__ == "__main__":
    main()
