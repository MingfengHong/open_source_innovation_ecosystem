"""
角色共生关系分析模块
实现用户角色间的依赖关系量化验证，支持RQ2研究问题
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.metrics import mutual_info_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, VISUALIZATION_CONFIG

# 设置日志
logger = setup_logger(__name__)


class RoleSymbiosisAnalyzer:
    """
    角色共生关系分析器
    量化验证不同用户角色之间的依赖关系和共生模式
    """
    
    def __init__(self, 
                 user_roles_df: pd.DataFrame,
                 network_graph: nx.Graph = None,
                 time_window_months: int = 3):
        """
        初始化角色共生分析器
        
        Args:
            user_roles_df: 用户角色数据框，包含user_id, cluster等字段
            network_graph: 网络图（可选，用于网络位置分析）
            time_window_months: 分析时间窗口（月）
        """
        self.user_roles_df = user_roles_df.copy()
        self.network_graph = network_graph
        self.time_window_months = time_window_months
        
        # 定义角色映射（基于聚类结果）
        self.role_mapping = self._create_role_mapping()
        
        # 存储分析结果
        self.symbiosis_results = {}
        self.dependency_matrix = None
        self.transition_probabilities = None
        
        logger.info(f"初始化角色共生分析器: {len(user_roles_df)} 个用户, {len(self.role_mapping)} 种角色")
    
    def _create_role_mapping(self) -> Dict[int, str]:
        """
        基于聚类结果和行为特征创建角色映射
        
        Returns:
            Dict[int, str]: 聚类ID到角色名称的映射
        """
        role_mapping = {}
        
        if 'cluster' not in self.user_roles_df.columns:
            logger.warning("用户数据中缺少cluster列，使用默认角色分配")
            # 创建默认角色分配
            unique_users = len(self.user_roles_df)
            for i in range(min(6, unique_users)):  # 最多6种角色
                role_mapping[i] = f"role_{i}"
            return role_mapping
        
        # 分析每个聚类的特征来确定角色名称
        cluster_stats = self.user_roles_df.groupby('cluster').agg({
            'pr_count': 'mean',
            'issue_count': 'mean', 
            'star_count': 'mean',
            'repo_count': 'mean',
            'code_focus_ratio': 'mean',
            'interaction_diversity': 'mean'
        }).fillna(0)
        
        for cluster_id, stats in cluster_stats.iterrows():
            # 基于特征模式确定角色名称
            if stats['code_focus_ratio'] > 0.7 and stats['pr_count'] > stats.mean()['pr_count']:
                role_name = "core_developer"  # 核心开发者
            elif stats['interaction_diversity'] > 3 and stats['issue_count'] > stats.mean()['issue_count']:
                role_name = "community_facilitator"  # 社区促进者/布道者
            elif stats['repo_count'] > stats.mean()['repo_count'] and stats['code_focus_ratio'] > 0.5:
                role_name = "architect"  # 架构师/项目创建者
            elif stats['issue_count'] > stats['pr_count'] and stats['interaction_diversity'] > 2:
                role_name = "problem_solver"  # 问题解决者
            elif stats['star_count'] > stats.mean()['star_count'] and stats['pr_count'] < stats.mean()['pr_count']:
                role_name = "observer"  # 观察者/学习者
            else:
                role_name = "casual_contributor"  # 偶然贡献者
            
            role_mapping[cluster_id] = role_name
        
        logger.info(f"角色映射: {role_mapping}")
        return role_mapping
    
    def analyze_role_dependencies(self, 
                                activity_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析角色间的依赖关系
        
        Args:
            activity_data: 活动数据，包含user_id, activity_type, target_id, timestamp等
            
        Returns:
            Dict[str, Any]: 依赖关系分析结果
        """
        logger.info("分析角色间的依赖关系...")
        
        # 准备数据
        activity_with_roles = self._prepare_activity_data(activity_data)
        
        if activity_with_roles.empty:
            logger.error("活动数据为空，无法进行依赖关系分析")
            return {}
        
        # 1. 计算角色间的协作频率
        collaboration_matrix = self._calculate_collaboration_matrix(activity_with_roles)
        
        # 2. 分析角色间的时间依赖关系
        temporal_dependencies = self._analyze_temporal_dependencies(activity_with_roles)
        
        # 3. 计算角色互补性指数
        complementarity_scores = self._calculate_role_complementarity(activity_with_roles)
        
        # 4. 分析知识流动模式
        knowledge_flow = self._analyze_knowledge_flow_patterns(activity_with_roles)
        
        # 5. 验证特定的共生假设
        symbiosis_hypotheses = self._validate_symbiosis_hypotheses(activity_with_roles)
        
        results = {
            'collaboration_matrix': collaboration_matrix,
            'temporal_dependencies': temporal_dependencies,
            'complementarity_scores': complementarity_scores,
            'knowledge_flow': knowledge_flow,
            'symbiosis_hypotheses': symbiosis_hypotheses,
            'summary_statistics': self._calculate_dependency_summary(
                collaboration_matrix, temporal_dependencies, complementarity_scores
            )
        }
        
        self.symbiosis_results = results
        
        # 保存结果
        self._save_symbiosis_results(results)
        
        return results
    
    def _prepare_activity_data(self, activity_data: pd.DataFrame) -> pd.DataFrame:
        """准备分析用的活动数据"""
        # 合并用户角色信息
        activity_with_roles = pd.merge(
            activity_data, 
            self.user_roles_df[['user_id', 'cluster']], 
            on='user_id', 
            how='left'
        )
        
        # 添加角色名称
        activity_with_roles['role'] = activity_with_roles['cluster'].map(self.role_mapping)
        
        # 过滤掉没有角色信息的数据
        activity_with_roles = activity_with_roles.dropna(subset=['role'])
        
        # 转换时间戳
        if 'timestamp' in activity_with_roles.columns:
            activity_with_roles['timestamp'] = pd.to_datetime(activity_with_roles['timestamp'])
        
        logger.info(f"准备活动数据: {len(activity_with_roles)} 条记录, {activity_with_roles['role'].nunique()} 种角色")
        
        return activity_with_roles
    
    def _calculate_collaboration_matrix(self, activity_data: pd.DataFrame) -> pd.DataFrame:
        """计算角色间的协作频率矩阵"""
        logger.info("计算角色协作矩阵...")
        
        roles = list(self.role_mapping.values())
        collaboration_matrix = pd.DataFrame(0, index=roles, columns=roles)
        
        # 基于共同参与的项目/仓库计算协作
        if 'target_id' in activity_data.columns:
            # 按项目分组，统计角色共现
            project_groups = activity_data.groupby('target_id')
            
            for project_id, group in project_groups:
                project_roles = group['role'].value_counts()
                
                # 计算角色对之间的协作强度
                for role1 in project_roles.index:
                    for role2 in project_roles.index:
                        if role1 != role2:
                            # 协作强度 = min(role1_count, role2_count) * 项目重要性权重
                            weight = min(project_roles[role1], project_roles[role2])
                            collaboration_matrix.loc[role1, role2] += weight
        
        # 归一化
        for role in roles:
            total = collaboration_matrix.loc[role].sum()
            if total > 0:
                collaboration_matrix.loc[role] = collaboration_matrix.loc[role] / total
        
        return collaboration_matrix
    
    def _analyze_temporal_dependencies(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """分析角色间的时间依赖关系"""
        logger.info("分析时间依赖关系...")
        
        if 'timestamp' not in activity_data.columns:
            logger.warning("缺少时间戳信息，跳过时间依赖分析")
            return {}
        
        temporal_deps = {}
        roles = list(self.role_mapping.values())
        
        # 按时间窗口分析角色活动的先后关系
        activity_data = activity_data.sort_values('timestamp')
        
        # 计算角色活动的时间序列
        role_activity_series = {}
        for role in roles:
            role_data = activity_data[activity_data['role'] == role]
            if not role_data.empty:
                # 按月聚合活动数量
                monthly_activity = role_data.set_index('timestamp').resample('M').size()
                role_activity_series[role] = monthly_activity.fillna(0)
        
        # 计算角色间的格兰杰因果关系（简化版）
        granger_results = {}
        for role1 in roles:
            for role2 in roles:
                if role1 != role2 and role1 in role_activity_series and role2 in role_activity_series:
                    # 对齐时间序列
                    series1 = role_activity_series[role1]
                    series2 = role_activity_series[role2]
                    
                    if len(series1) > 3 and len(series2) > 3:
                        # 计算滞后相关性
                        lag_correlations = self._calculate_lag_correlations(series1, series2, max_lag=3)
                        granger_results[f"{role1}_to_{role2}"] = lag_correlations
        
        temporal_deps['granger_causality'] = granger_results
        
        # 分析活动模式的时间互补性
        complementarity_patterns = self._analyze_temporal_complementarity(role_activity_series)
        temporal_deps['complementarity_patterns'] = complementarity_patterns
        
        return temporal_deps
    
    def _calculate_lag_correlations(self, series1: pd.Series, series2: pd.Series, max_lag: int = 3) -> Dict[int, float]:
        """计算滞后相关性"""
        lag_correlations = {}
        
        # 对齐索引
        common_index = series1.index.intersection(series2.index)
        if len(common_index) < 4:
            return {}
        
        s1_aligned = series1.reindex(common_index).fillna(0)
        s2_aligned = series2.reindex(common_index).fillna(0)
        
        for lag in range(0, max_lag + 1):
            if lag == 0:
                # 同期相关性
                if len(s1_aligned) > 1 and s1_aligned.std() > 0 and s2_aligned.std() > 0:
                    corr, p_value = pearsonr(s1_aligned, s2_aligned)
                    lag_correlations[lag] = {'correlation': corr, 'p_value': p_value}
            else:
                # 滞后相关性
                if len(s1_aligned) > lag + 1:
                    s1_lagged = s1_aligned[:-lag]
                    s2_current = s2_aligned[lag:]
                    
                    if len(s1_lagged) > 1 and s1_lagged.std() > 0 and s2_current.std() > 0:
                        corr, p_value = pearsonr(s1_lagged, s2_current)
                        lag_correlations[lag] = {'correlation': corr, 'p_value': p_value}
        
        return lag_correlations
    
    def _analyze_temporal_complementarity(self, role_activity_series: Dict[str, pd.Series]) -> Dict[str, float]:
        """分析角色活动的时间互补性"""
        complementarity = {}
        roles = list(role_activity_series.keys())
        
        for i, role1 in enumerate(roles):
            for role2 in roles[i+1:]:
                if role1 in role_activity_series and role2 in role_activity_series:
                    series1 = role_activity_series[role1]
                    series2 = role_activity_series[role2]
                    
                    # 计算活动时间的互补性（负相关表示互补）
                    common_index = series1.index.intersection(series2.index)
                    if len(common_index) > 3:
                        s1 = series1.reindex(common_index).fillna(0)
                        s2 = series2.reindex(common_index).fillna(0)
                        
                        if s1.std() > 0 and s2.std() > 0:
                            corr, _ = pearsonr(s1, s2)
                            # 互补性 = 1 - |相关系数|，值越大表示越互补
                            complementarity[f"{role1}_{role2}"] = 1 - abs(corr)
        
        return complementarity
    
    def _calculate_role_complementarity(self, activity_data: pd.DataFrame) -> Dict[str, float]:
        """计算角色互补性指数"""
        logger.info("计算角色互补性指数...")
        
        complementarity_scores = {}
        roles = list(self.role_mapping.values())
        
        # 基于活动类型的互补性分析
        if 'activity_type' in activity_data.columns:
            # 计算每种角色在不同活动类型上的分布
            role_activity_distribution = pd.crosstab(
                activity_data['role'], 
                activity_data['activity_type'], 
                normalize='index'
            )
            
            # 计算角色间的Jensen-Shannon散度（衡量分布差异）
            for i, role1 in enumerate(roles):
                for role2 in roles[i+1:]:
                    if role1 in role_activity_distribution.index and role2 in role_activity_distribution.index:
                        dist1 = role_activity_distribution.loc[role1].values
                        dist2 = role_activity_distribution.loc[role2].values
                        
                        # Jensen-Shannon散度
                        js_divergence = self._jensen_shannon_divergence(dist1, dist2)
                        # 互补性得分：散度越大，互补性越强
                        complementarity_scores[f"{role1}_{role2}"] = js_divergence
        
        return complementarity_scores
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """计算Jensen-Shannon散度"""
        # 确保概率分布归一化
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q
        
        # 计算平均分布
        m = (p + q) / 2
        
        # 避免log(0)
        p = np.where(p == 0, 1e-10, p)
        q = np.where(q == 0, 1e-10, q) 
        m = np.where(m == 0, 1e-10, m)
        
        # Jensen-Shannon散度
        js_div = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
        
        return js_div
    
    def _analyze_knowledge_flow_patterns(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """分析知识流动模式"""
        logger.info("分析知识流动模式...")
        
        knowledge_flow = {}
        
        # 基于文档贡献和代码贡献的知识流动分析
        if 'activity_type' in activity_data.columns:
            doc_activities = activity_data[activity_data['activity_type'].str.contains('doc|documentation', case=False, na=False)]
            code_activities = activity_data[activity_data['activity_type'].str.contains('code|pr', case=False, na=False)]
            
            # 分析文档贡献者对代码贡献者的影响
            doc_to_code_flow = self._analyze_role_influence_flow(doc_activities, code_activities)
            knowledge_flow['documentation_to_code'] = doc_to_code_flow
            
            # 分析代码贡献者对问题解决的影响
            issue_activities = activity_data[activity_data['activity_type'].str.contains('issue', case=False, na=False)]
            code_to_issue_flow = self._analyze_role_influence_flow(code_activities, issue_activities)
            knowledge_flow['code_to_issue_resolution'] = code_to_issue_flow
        
        return knowledge_flow
    
    def _analyze_role_influence_flow(self, source_activities: pd.DataFrame, target_activities: pd.DataFrame) -> Dict[str, float]:
        """分析角色间的影响流动"""
        influence_flow = {}
        
        if source_activities.empty or target_activities.empty:
            return influence_flow
        
        # 按角色聚合活动
        source_by_role = source_activities['role'].value_counts()
        target_by_role = target_activities['role'].value_counts()
        
        # 计算影响强度（基于活动数量的相关性）
        common_roles = set(source_by_role.index).intersection(set(target_by_role.index))
        
        for role in common_roles:
            source_count = source_by_role.get(role, 0)
            target_count = target_by_role.get(role, 0)
            
            if source_count > 0 and target_count > 0:
                # 影响强度 = min(source, target) / max(source, target)
                influence_strength = min(source_count, target_count) / max(source_count, target_count)
                influence_flow[role] = influence_strength
        
        return influence_flow
    
    def _validate_symbiosis_hypotheses(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """验证特定的共生假设"""
        logger.info("验证角色共生假设...")
        
        hypotheses_results = {}
        
        # 假设1: 布道者(community_facilitator)的文档贡献降低架构师(architect)的入门门槛
        hypothesis1 = self._test_facilitator_architect_symbiosis(activity_data)
        hypotheses_results['facilitator_architect_symbiosis'] = hypothesis1
        
        # 假设2: 核心开发者(core_developer)与问题解决者(problem_solver)的互补关系
        hypothesis2 = self._test_developer_solver_complementarity(activity_data)
        hypotheses_results['developer_solver_complementarity'] = hypothesis2
        
        # 假设3: 观察者(observer)到贡献者的角色转换路径
        hypothesis3 = self._test_observer_contributor_transition(activity_data)
        hypotheses_results['observer_contributor_transition'] = hypothesis3
        
        return hypotheses_results
    
    def _test_facilitator_architect_symbiosis(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """测试布道者-架构师共生关系"""
        result = {'hypothesis': 'Facilitators enable architects through documentation', 'evidence': {}}
        
        facilitators = activity_data[activity_data['role'] == 'community_facilitator']
        architects = activity_data[activity_data['role'] == 'architect']
        
        if facilitators.empty or architects.empty:
            result['conclusion'] = 'Insufficient data'
            return result
        
        # 分析文档活动对新项目创建的时间滞后影响
        doc_activities = facilitators[facilitators.get('activity_type', '').str.contains('doc', case=False, na=False)]
        project_activities = architects[architects.get('activity_type', '').str.contains('create|repo', case=False, na=False)]
        
        if not doc_activities.empty and not project_activities.empty and 'timestamp' in activity_data.columns:
            # 按月统计活动
            doc_monthly = doc_activities.set_index('timestamp').resample('M').size()
            project_monthly = project_activities.set_index('timestamp').resample('M').size()
            
            # 计算滞后相关性
            lag_corrs = self._calculate_lag_correlations(doc_monthly, project_monthly, max_lag=2)
            
            result['evidence']['lag_correlations'] = lag_corrs
            
            # 寻找最强的正相关滞后
            max_corr_lag = max(lag_corrs.keys(), key=lambda k: lag_corrs[k].get('correlation', 0))
            max_corr = lag_corrs[max_corr_lag]['correlation']
            
            if max_corr > 0.3 and lag_corrs[max_corr_lag]['p_value'] < 0.05:
                result['conclusion'] = f'Strong positive evidence (r={max_corr:.3f}, lag={max_corr_lag})'
            elif max_corr > 0.1:
                result['conclusion'] = f'Weak positive evidence (r={max_corr:.3f}, lag={max_corr_lag})'
            else:
                result['conclusion'] = 'No significant evidence'
        else:
            result['conclusion'] = 'Insufficient temporal data'
        
        return result
    
    def _test_developer_solver_complementarity(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """测试开发者-问题解决者互补关系"""
        result = {'hypothesis': 'Developers and problem solvers have complementary activity patterns', 'evidence': {}}
        
        developers = activity_data[activity_data['role'] == 'core_developer']
        solvers = activity_data[activity_data['role'] == 'problem_solver']
        
        if developers.empty or solvers.empty:
            result['conclusion'] = 'Insufficient data'
            return result
        
        # 分析活动类型的互补性
        if 'activity_type' in activity_data.columns:
            dev_activities = developers['activity_type'].value_counts(normalize=True)
            solver_activities = solvers['activity_type'].value_counts(normalize=True)
            
            # 计算活动分布的重叠度
            common_activities = set(dev_activities.index).intersection(set(solver_activities.index))
            
            if common_activities:
                overlap_score = 0
                for activity in common_activities:
                    overlap_score += min(dev_activities[activity], solver_activities[activity])
                
                complementarity_score = 1 - overlap_score  # 互补性 = 1 - 重叠度
                result['evidence']['complementarity_score'] = complementarity_score
                result['evidence']['activity_overlap'] = overlap_score
                
                if complementarity_score > 0.6:
                    result['conclusion'] = f'Strong complementarity (score={complementarity_score:.3f})'
                elif complementarity_score > 0.3:
                    result['conclusion'] = f'Moderate complementarity (score={complementarity_score:.3f})'
                else:
                    result['conclusion'] = f'Weak complementarity (score={complementarity_score:.3f})'
            else:
                result['conclusion'] = 'No common activities found'
        else:
            result['conclusion'] = 'No activity type data'
        
        return result
    
    def _test_observer_contributor_transition(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """测试观察者到贡献者的转换路径"""
        result = {'hypothesis': 'Observers transition to contributors through specific pathways', 'evidence': {}}
        
        # 这个分析需要时间序列数据来追踪用户的角色变化
        # 由于当前数据结构限制，这里提供一个简化的分析框架
        
        observers = activity_data[activity_data['role'] == 'observer']
        contributors = activity_data[activity_data['role'].isin(['casual_contributor', 'core_developer'])]
        
        if observers.empty or contributors.empty:
            result['conclusion'] = 'Insufficient data'
            return result
        
        # 分析观察者和贡献者的活动模式差异
        if 'activity_type' in activity_data.columns:
            observer_patterns = observers['activity_type'].value_counts(normalize=True)
            contributor_patterns = contributors['activity_type'].value_counts(normalize=True)
            
            # 识别转换路径（观察者较多但贡献者也有的活动类型）
            transition_activities = []
            for activity in observer_patterns.index:
                if activity in contributor_patterns.index:
                    # 转换指数 = observer比例 * contributor比例
                    transition_index = observer_patterns[activity] * contributor_patterns[activity]
                    transition_activities.append((activity, transition_index))
            
            # 排序找到最可能的转换路径
            transition_activities.sort(key=lambda x: x[1], reverse=True)
            
            result['evidence']['transition_pathways'] = transition_activities[:5]  # 前5个最可能的路径
            
            if transition_activities:
                top_pathway = transition_activities[0]
                result['conclusion'] = f'Primary transition pathway: {top_pathway[0]} (index={top_pathway[1]:.3f})'
            else:
                result['conclusion'] = 'No clear transition pathways identified'
        else:
            result['conclusion'] = 'No activity type data for pathway analysis'
        
        return result
    
    def _calculate_dependency_summary(self, 
                                    collaboration_matrix: pd.DataFrame,
                                    temporal_dependencies: Dict[str, Any],
                                    complementarity_scores: Dict[str, float]) -> Dict[str, Any]:
        """计算依赖关系摘要统计"""
        summary = {}
        
        # 协作网络统计
        if not collaboration_matrix.empty:
            summary['collaboration_density'] = (collaboration_matrix > 0).sum().sum() / (len(collaboration_matrix) ** 2)
            summary['max_collaboration_strength'] = collaboration_matrix.max().max()
            summary['most_collaborative_role'] = collaboration_matrix.sum(axis=1).idxmax()
        
        # 时间依赖统计
        if temporal_dependencies and 'granger_causality' in temporal_dependencies:
            granger_results = temporal_dependencies['granger_causality']
            significant_dependencies = 0
            
            for dep_key, lag_corrs in granger_results.items():
                for lag, corr_data in lag_corrs.items():
                    if isinstance(corr_data, dict) and corr_data.get('p_value', 1) < 0.05:
                        significant_dependencies += 1
            
            summary['significant_temporal_dependencies'] = significant_dependencies
            summary['total_temporal_tests'] = sum(len(lc) for lc in granger_results.values())
        
        # 互补性统计
        if complementarity_scores:
            summary['avg_complementarity'] = np.mean(list(complementarity_scores.values()))
            summary['max_complementarity'] = max(complementarity_scores.values())
            summary['most_complementary_pair'] = max(complementarity_scores.keys(), 
                                                   key=lambda k: complementarity_scores[k])
        
        return summary
    
    def _save_symbiosis_results(self, results: Dict[str, Any]):
        """保存共生关系分析结果"""
        # 保存协作矩阵
        if 'collaboration_matrix' in results:
            collab_path = ANALYSIS_OUTPUT_DIR / "role_collaboration_matrix.csv"
            results['collaboration_matrix'].to_csv(collab_path, encoding='utf-8-sig')
            logger.info(f"协作矩阵已保存至: {collab_path}")
        
        # 保存摘要结果
        import json
        summary_path = ANALYSIS_OUTPUT_DIR / "role_symbiosis_analysis.json"
        
        # 处理不能JSON序列化的对象
        json_results = {}
        for key, value in results.items():
            if key == 'collaboration_matrix':
                json_results[key] = value.to_dict()
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"共生关系分析结果已保存至: {summary_path}")
    
    def visualize_symbiosis_relationships(self):
        """可视化角色共生关系"""
        if not self.symbiosis_results:
            logger.warning("没有共生关系分析结果可视化")
            return
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 协作矩阵热力图
        if 'collaboration_matrix' in self.symbiosis_results:
            collab_matrix = self.symbiosis_results['collaboration_matrix']
            if not collab_matrix.empty:
                sns.heatmap(collab_matrix, annot=True, cmap='YlOrRd', ax=axes[0,0], fmt='.3f')
                axes[0,0].set_title('Role Collaboration Matrix', fontsize=VISUALIZATION_CONFIG["title_font_size"])
                axes[0,0].set_xlabel('Target Role')
                axes[0,0].set_ylabel('Source Role')
        
        # 2. 互补性得分
        if 'complementarity_scores' in self.symbiosis_results:
            comp_scores = self.symbiosis_results['complementarity_scores']
            if comp_scores:
                pairs = list(comp_scores.keys())
                scores = list(comp_scores.values())
                
                axes[0,1].bar(range(len(pairs)), scores)
                axes[0,1].set_title('Role Complementarity Scores', fontsize=VISUALIZATION_CONFIG["title_font_size"])
                axes[0,1].set_xlabel('Role Pairs')
                axes[0,1].set_ylabel('Complementarity Score')
                axes[0,1].set_xticks(range(len(pairs)))
                axes[0,1].set_xticklabels(pairs, rotation=45, ha='right')
        
        # 3. 时间依赖关系网络图
        if 'temporal_dependencies' in self.symbiosis_results:
            self._plot_temporal_dependency_network(axes[1,0])
        
        # 4. 假设验证结果
        if 'symbiosis_hypotheses' in self.symbiosis_results:
            self._plot_hypothesis_validation_results(axes[1,1])
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "role_symbiosis_visualization.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"共生关系可视化图表已保存至: {save_path}")
        
        plt.show()
    
    def _plot_temporal_dependency_network(self, ax):
        """绘制时间依赖关系网络图"""
        ax.set_title('Temporal Dependency Network', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        
        temporal_deps = self.symbiosis_results.get('temporal_dependencies', {})
        granger_results = temporal_deps.get('granger_causality', {})
        
        if not granger_results:
            ax.text(0.5, 0.5, 'No temporal dependency data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 创建网络图
        G = nx.DiGraph()
        
        # 添加显著的因果关系作为边
        for dep_key, lag_corrs in granger_results.items():
            if '_to_' in dep_key:
                source_role, target_role = dep_key.split('_to_')
                
                # 找到最强的显著相关性
                max_corr = 0
                best_lag = 0
                for lag, corr_data in lag_corrs.items():
                    if isinstance(corr_data, dict):
                        corr = corr_data.get('correlation', 0)
                        p_val = corr_data.get('p_value', 1)
                        if p_val < 0.05 and abs(corr) > abs(max_corr):
                            max_corr = corr
                            best_lag = lag
                
                if abs(max_corr) > 0.2:  # 只显示较强的关系
                    G.add_edge(source_role, target_role, weight=abs(max_corr), lag=best_lag)
        
        if G.number_of_nodes() > 0:
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=1000, alpha=0.7, ax=ax)
            
            # 绘制边
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], 
                                 alpha=0.6, edge_color='red', 
                                 arrowsize=20, arrowstyle='->', ax=ax)
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No significant temporal dependencies', 
                   ha='center', va='center', transform=ax.transAxes)
        
        ax.set_axis_off()
    
    def _plot_hypothesis_validation_results(self, ax):
        """绘制假设验证结果"""
        ax.set_title('Symbiosis Hypothesis Validation', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        
        hypotheses = self.symbiosis_results.get('symbiosis_hypotheses', {})
        
        if not hypotheses:
            ax.text(0.5, 0.5, 'No hypothesis validation data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 提取验证结果
        hypothesis_names = []
        evidence_scores = []
        
        for hyp_name, hyp_result in hypotheses.items():
            conclusion = hyp_result.get('conclusion', 'No evidence')
            
            # 根据结论文本评估证据强度
            if 'Strong' in conclusion or 'strong' in conclusion:
                score = 3
            elif 'Moderate' in conclusion or 'moderate' in conclusion or 'Weak positive' in conclusion:
                score = 2
            elif 'Weak' in conclusion or 'weak' in conclusion:
                score = 1
            else:
                score = 0
            
            hypothesis_names.append(hyp_name.replace('_', ' ').title())
            evidence_scores.append(score)
        
        if hypothesis_names:
            colors = ['red' if s == 0 else 'orange' if s == 1 else 'yellow' if s == 2 else 'green' for s in evidence_scores]
            bars = ax.bar(range(len(hypothesis_names)), evidence_scores, color=colors, alpha=0.7)
            
            ax.set_xlabel('Hypotheses')
            ax.set_ylabel('Evidence Strength')
            ax.set_xticks(range(len(hypothesis_names)))
            ax.set_xticklabels(hypothesis_names, rotation=45, ha='right')
            ax.set_ylim(0, 3.5)
            
            # 添加图例
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Strong Evidence'),
                plt.Rectangle((0,0),1,1, facecolor='yellow', alpha=0.7, label='Moderate Evidence'),
                plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Weak Evidence'),
                plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='No Evidence')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No validation results', ha='center', va='center', transform=ax.transAxes)
    
    def generate_symbiosis_report(self) -> str:
        """生成角色共生关系分析报告"""
        if not self.symbiosis_results:
            return "No symbiosis analysis results available."
        
        report_lines = [
            "=" * 60,
            "角色共生关系分析报告",
            "Role Symbiosis Analysis Report",
            "=" * 60,
            ""
        ]
        
        # 摘要统计
        if 'summary_statistics' in self.symbiosis_results:
            summary = self.symbiosis_results['summary_statistics']
            report_lines.extend([
                "📊 摘要统计 (Summary Statistics):",
                f"  - 协作网络密度: {summary.get('collaboration_density', 'N/A'):.3f}",
                f"  - 最大协作强度: {summary.get('max_collaboration_strength', 'N/A'):.3f}",
                f"  - 最协作的角色: {summary.get('most_collaborative_role', 'N/A')}",
                f"  - 显著时间依赖关系数: {summary.get('significant_temporal_dependencies', 'N/A')}",
                f"  - 平均互补性得分: {summary.get('avg_complementarity', 'N/A'):.3f}",
                f"  - 最互补的角色对: {summary.get('most_complementary_pair', 'N/A')}",
                ""
            ])
        
        # 假设验证结果
        if 'symbiosis_hypotheses' in self.symbiosis_results:
            report_lines.extend([
                "🔬 共生假设验证结果 (Symbiosis Hypothesis Validation):",
                ""
            ])
            
            for hyp_name, hyp_result in self.symbiosis_results['symbiosis_hypotheses'].items():
                report_lines.extend([
                    f"假设: {hyp_result.get('hypothesis', 'Unknown')}",
                    f"结论: {hyp_result.get('conclusion', 'No conclusion')}",
                    ""
                ])
        
        # 关键发现
        report_lines.extend([
            "🔍 关键发现 (Key Findings):",
            ""
        ])
        
        # 基于结果生成关键发现
        if 'collaboration_matrix' in self.symbiosis_results:
            collab_matrix = self.symbiosis_results['collaboration_matrix']
            if not collab_matrix.empty:
                max_collab = collab_matrix.max().max()
                max_pair = collab_matrix.stack().idxmax()
                report_lines.append(f"  - 最强协作关系: {max_pair[0]} → {max_pair[1]} (强度: {max_collab:.3f})")
        
        if 'complementarity_scores' in self.symbiosis_results:
            comp_scores = self.symbiosis_results['complementarity_scores']
            if comp_scores:
                max_comp_pair = max(comp_scores.keys(), key=lambda k: comp_scores[k])
                max_comp_score = comp_scores[max_comp_pair]
                report_lines.append(f"  - 最强互补关系: {max_comp_pair} (得分: {max_comp_score:.3f})")
        
        report_lines.extend([
            "",
            "=" * 60,
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        # 保存报告
        report_path = ANALYSIS_OUTPUT_DIR / "role_symbiosis_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"共生关系分析报告已保存至: {report_path}")
        
        return report


def main():
    """主函数入口"""
    logger.info("角色共生关系分析模块测试")
    
    # 这里应该加载真实的用户角色数据和活动数据进行测试
    # 由于没有真实数据，这里提供一个使用示例
    
    print("角色共生关系分析器已实现以下功能:")
    print("1. 角色依赖关系分析")
    print("2. 时间序列因果关系检验")
    print("3. 角色互补性量化")
    print("4. 知识流动模式分析")
    print("5. 特定共生假设验证")
    print("6. 可视化和报告生成")
    print("\n使用方法:")
    print("analyzer = RoleSymbiosisAnalyzer(user_roles_df)")
    print("results = analyzer.analyze_role_dependencies(activity_data)")
    print("analyzer.visualize_symbiosis_relationships()")
    print("report = analyzer.generate_symbiosis_report()")


if __name__ == "__main__":
    main()
