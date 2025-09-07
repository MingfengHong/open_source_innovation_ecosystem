"""
角色转换路径分析模块
分析用户在开源生态系统中的角色演化和转换路径
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, VISUALIZATION_CONFIG

# 设置日志
logger = setup_logger(__name__)


class RoleTransitionAnalyzer:
    """
    角色转换路径分析器
    分析用户角色的演化模式和转换概率
    """
    
    def __init__(self, 
                 user_roles_df: pd.DataFrame,
                 activity_data: pd.DataFrame = None,
                 time_window_months: int = 6):
        """
        初始化角色转换分析器
        
        Args:
            user_roles_df: 用户角色数据框
            activity_data: 用户活动数据（可选，用于预测转换）
            time_window_months: 分析时间窗口
        """
        self.user_roles_df = user_roles_df.copy()
        self.activity_data = activity_data
        self.time_window_months = time_window_months
        
        # 角色映射（与共生分析保持一致）
        self.role_mapping = self._create_role_mapping()
        
        # 存储分析结果
        self.transition_matrix = None
        self.transition_probabilities = None
        self.pathway_analysis = {}
        self.prediction_model = None
        
        logger.info(f"初始化角色转换分析器: {len(user_roles_df)} 个用户, {len(self.role_mapping)} 种角色")
    
    def _create_role_mapping(self) -> Dict[int, str]:
        """创建角色映射"""
        role_mapping = {}
        
        if 'cluster' not in self.user_roles_df.columns:
            logger.warning("用户数据中缺少cluster列，使用默认角色分配")
            return {i: f"role_{i}" for i in range(6)}
        
        # 基于聚类特征分析确定角色名称
        cluster_stats = self.user_roles_df.groupby('cluster').agg({
            'pr_count': 'mean',
            'issue_count': 'mean', 
            'star_count': 'mean',
            'repo_count': 'mean',
            'code_focus_ratio': 'mean',
            'interaction_diversity': 'mean'
        }).fillna(0)
        
        for cluster_id, stats in cluster_stats.iterrows():
            if stats['code_focus_ratio'] > 0.7 and stats['pr_count'] > stats.mean()['pr_count']:
                role_name = "core_developer"
            elif stats['interaction_diversity'] > 3 and stats['issue_count'] > stats.mean()['issue_count']:
                role_name = "community_facilitator"
            elif stats['repo_count'] > stats.mean()['repo_count'] and stats['code_focus_ratio'] > 0.5:
                role_name = "architect"
            elif stats['issue_count'] > stats['pr_count'] and stats['interaction_diversity'] > 2:
                role_name = "problem_solver"
            elif stats['star_count'] > stats.mean()['star_count'] and stats['pr_count'] < stats.mean()['pr_count']:
                role_name = "observer"
            else:
                role_name = "casual_contributor"
            
            role_mapping[cluster_id] = role_name
        
        return role_mapping
    
    def analyze_role_transitions(self, 
                                temporal_user_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        分析角色转换模式
        
        Args:
            temporal_user_data: 时间序列用户数据，包含user_id, time_period, cluster等
            
        Returns:
            Dict[str, Any]: 转换分析结果
        """
        logger.info("开始分析角色转换模式...")
        
        if temporal_user_data is None:
            # 如果没有时间序列数据，创建模拟数据用于演示
            temporal_user_data = self._create_simulated_temporal_data()
        
        # 1. 计算转换矩阵
        transition_matrix = self._calculate_transition_matrix(temporal_user_data)
        
        # 2. 分析转换路径
        pathway_analysis = self._analyze_transition_pathways(temporal_user_data)
        
        # 3. 计算转换概率和稳定性
        stability_analysis = self._analyze_role_stability(temporal_user_data)
        
        # 4. 识别关键转换触发因素
        trigger_factors = self._identify_transition_triggers(temporal_user_data)
        
        # 5. 预测未来转换
        prediction_results = self._predict_role_transitions(temporal_user_data)
        
        results = {
            'transition_matrix': transition_matrix,
            'pathway_analysis': pathway_analysis,
            'stability_analysis': stability_analysis,
            'trigger_factors': trigger_factors,
            'prediction_results': prediction_results,
            'summary_statistics': self._calculate_transition_summary(transition_matrix, stability_analysis)
        }
        
        # 存储结果
        self.transition_matrix = transition_matrix
        self.pathway_analysis = pathway_analysis
        
        # 保存结果
        self._save_transition_results(results)
        
        return results
    
    def _create_simulated_temporal_data(self) -> pd.DataFrame:
        """创建模拟的时间序列用户数据"""
        logger.info("创建模拟时间序列数据...")
        
        temporal_data = []
        time_periods = pd.date_range('2023-01-01', '2023-12-31', freq='M')
        
        for user_id in self.user_roles_df['user_id'].unique()[:100]:  # 限制用户数量
            current_role = self.user_roles_df[self.user_roles_df['user_id'] == user_id]['cluster'].iloc[0]
            
            for period in time_periods:
                # 模拟角色转换概率
                transition_prob = np.random.random()
                
                if transition_prob < 0.1:  # 10%的概率发生转换
                    # 随机选择新角色（倾向于相邻角色）
                    possible_roles = list(self.role_mapping.keys())
                    # 移除当前角色
                    if current_role in possible_roles:
                        possible_roles.remove(current_role)
                    
                    if possible_roles:
                        new_role = np.random.choice(possible_roles)
                        current_role = new_role
                
                temporal_data.append({
                    'user_id': user_id,
                    'time_period': period,
                    'cluster': current_role,
                    'role': self.role_mapping[current_role]
                })
        
        return pd.DataFrame(temporal_data)
    
    def _calculate_transition_matrix(self, temporal_data: pd.DataFrame) -> pd.DataFrame:
        """计算角色转换矩阵"""
        logger.info("计算角色转换矩阵...")
        
        roles = list(self.role_mapping.values())
        transition_counts = pd.DataFrame(0, index=roles, columns=roles)
        
        # 按用户分组，分析角色转换
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            
            # 计算连续时期的角色转换
            for i in range(len(user_data) - 1):
                current_role = user_data.iloc[i]['role']
                next_role = user_data.iloc[i + 1]['role']
                
                if current_role in roles and next_role in roles:
                    transition_counts.loc[current_role, next_role] += 1
        
        # 转换为概率矩阵
        transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
        
        return transition_matrix
    
    def _analyze_transition_pathways(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """分析转换路径"""
        logger.info("分析角色转换路径...")
        
        pathway_analysis = {}
        
        # 1. 识别最常见的转换路径
        common_transitions = self._find_common_transition_sequences(temporal_data)
        pathway_analysis['common_sequences'] = common_transitions
        
        # 2. 分析新手到专家的路径
        novice_to_expert_paths = self._analyze_novice_expert_pathways(temporal_data)
        pathway_analysis['novice_expert_paths'] = novice_to_expert_paths
        
        # 3. 计算路径长度分布
        path_length_distribution = self._calculate_path_length_distribution(temporal_data)
        pathway_analysis['path_length_distribution'] = path_length_distribution
        
        # 4. 识别关键中介角色
        bridge_roles = self._identify_bridge_roles(temporal_data)
        pathway_analysis['bridge_roles'] = bridge_roles
        
        return pathway_analysis
    
    def _find_common_transition_sequences(self, temporal_data: pd.DataFrame, max_length: int = 4) -> List[Dict]:
        """找到最常见的转换序列"""
        sequences = defaultdict(int)
        
        # 提取每个用户的角色序列
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            role_sequence = user_data['role'].tolist()
            
            # 提取所有可能的子序列
            for length in range(2, min(max_length + 1, len(role_sequence) + 1)):
                for i in range(len(role_sequence) - length + 1):
                    subseq = tuple(role_sequence[i:i + length])
                    sequences[subseq] += 1
        
        # 转换为列表并排序
        common_sequences = [
            {'sequence': list(seq), 'count': count, 'length': len(seq)}
            for seq, count in sequences.items()
            if count >= 2  # 至少出现2次
        ]
        
        common_sequences.sort(key=lambda x: x['count'], reverse=True)
        
        return common_sequences[:20]  # 返回前20个最常见的序列
    
    def _analyze_novice_expert_pathways(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """分析新手到专家的路径"""
        # 定义新手和专家角色
        novice_roles = ['observer', 'casual_contributor']
        expert_roles = ['core_developer', 'architect', 'community_facilitator']
        
        pathways = []
        
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            roles = user_data['role'].tolist()
            
            # 寻找从新手角色开始，到专家角色结束的路径
            start_indices = [i for i, role in enumerate(roles) if role in novice_roles]
            end_indices = [i for i, role in enumerate(roles) if role in expert_roles]
            
            for start_idx in start_indices:
                for end_idx in end_indices:
                    if end_idx > start_idx:
                        pathway = roles[start_idx:end_idx + 1]
                        pathways.append({
                            'user_id': user_id,
                            'pathway': pathway,
                            'length': len(pathway),
                            'duration': end_idx - start_idx
                        })
                        break  # 只取第一个专家角色
        
        # 分析路径统计
        if pathways:
            avg_length = np.mean([p['length'] for p in pathways])
            avg_duration = np.mean([p['duration'] for p in pathways])
            
            # 最常见的路径
            pathway_counts = Counter(tuple(p['pathway']) for p in pathways)
            most_common_paths = [
                {'pathway': list(path), 'count': count}
                for path, count in pathway_counts.most_common(10)
            ]
        else:
            avg_length = 0
            avg_duration = 0
            most_common_paths = []
        
        return {
            'total_pathways_found': len(pathways),
            'average_pathway_length': avg_length,
            'average_transition_duration': avg_duration,
            'most_common_pathways': most_common_paths
        }
    
    def _calculate_path_length_distribution(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """计算路径长度分布"""
        path_lengths = []
        
        for user_id, user_data in temporal_data.groupby('user_id'):
            # 计算每个用户经历的不同角色数量
            unique_roles = user_data['role'].nunique()
            path_lengths.append(unique_roles)
        
        if path_lengths:
            distribution = {
                'mean_length': np.mean(path_lengths),
                'std_length': np.std(path_lengths),
                'min_length': min(path_lengths),
                'max_length': max(path_lengths),
                'length_counts': dict(Counter(path_lengths))
            }
        else:
            distribution = {
                'mean_length': 0,
                'std_length': 0,
                'min_length': 0,
                'max_length': 0,
                'length_counts': {}
            }
        
        return distribution
    
    def _identify_bridge_roles(self, temporal_data: pd.DataFrame) -> Dict[str, float]:
        """识别在角色转换中起桥梁作用的角色"""
        role_bridge_scores = defaultdict(int)
        total_transitions = 0
        
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            roles = user_data['role'].tolist()
            
            # 分析三元组转换模式 A -> B -> C
            for i in range(len(roles) - 2):
                role_a, role_b, role_c = roles[i], roles[i + 1], roles[i + 2]
                
                if role_a != role_b and role_b != role_c and role_a != role_c:
                    # B是从A到C的桥梁角色
                    role_bridge_scores[role_b] += 1
                    total_transitions += 1
        
        # 计算桥梁得分（归一化）
        if total_transitions > 0:
            bridge_roles = {
                role: score / total_transitions
                for role, score in role_bridge_scores.items()
            }
        else:
            bridge_roles = {}
        
        return bridge_roles
    
    def _analyze_role_stability(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """分析角色稳定性"""
        logger.info("分析角色稳定性...")
        
        stability_metrics = {}
        
        for role in self.role_mapping.values():
            role_data = temporal_data[temporal_data['role'] == role]
            
            if role_data.empty:
                stability_metrics[role] = {
                    'retention_rate': 0,
                    'avg_tenure': 0,
                    'transition_rate': 0
                }
                continue
            
            # 计算保留率（连续时期保持同一角色的比例）
            retention_count = 0
            total_periods = 0
            
            # 计算平均任期长度
            tenure_lengths = []
            
            for user_id, user_data in role_data.groupby('user_id'):
                user_data = user_data.sort_values('time_period')
                user_all_data = temporal_data[temporal_data['user_id'] == user_id].sort_values('time_period')
                
                # 找到该角色的连续时期
                current_tenure = 0
                for _, row in user_all_data.iterrows():
                    if row['role'] == role:
                        current_tenure += 1
                    else:
                        if current_tenure > 0:
                            tenure_lengths.append(current_tenure)
                        current_tenure = 0
                
                if current_tenure > 0:
                    tenure_lengths.append(current_tenure)
                
                # 计算保留率
                for i in range(len(user_all_data) - 1):
                    if user_all_data.iloc[i]['role'] == role:
                        total_periods += 1
                        if user_all_data.iloc[i + 1]['role'] == role:
                            retention_count += 1
            
            # 计算指标
            retention_rate = retention_count / total_periods if total_periods > 0 else 0
            avg_tenure = np.mean(tenure_lengths) if tenure_lengths else 0
            transition_rate = 1 - retention_rate
            
            stability_metrics[role] = {
                'retention_rate': retention_rate,
                'avg_tenure': avg_tenure,
                'transition_rate': transition_rate,
                'total_observations': total_periods
            }
        
        return stability_metrics
    
    def _identify_transition_triggers(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """识别角色转换的触发因素"""
        logger.info("识别转换触发因素...")
        
        triggers = {}
        
        if self.activity_data is not None:
            # 分析活动数据与角色转换的关系
            activity_triggers = self._analyze_activity_based_triggers(temporal_data)
            triggers['activity_based'] = activity_triggers
        
        # 分析时间基础的触发模式
        temporal_triggers = self._analyze_temporal_triggers(temporal_data)
        triggers['temporal_patterns'] = temporal_triggers
        
        # 分析角色序列模式
        sequence_triggers = self._analyze_sequence_triggers(temporal_data)
        triggers['sequence_patterns'] = sequence_triggers
        
        return triggers
    
    def _analyze_activity_based_triggers(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """基于活动数据分析转换触发因素"""
        # 这里需要更复杂的活动数据分析
        # 简化版本：分析活动量变化与角色转换的关系
        
        activity_triggers = {}
        
        # 合并时间数据和活动数据
        if 'timestamp' in self.activity_data.columns and 'user_id' in self.activity_data.columns:
            # 按月聚合活动数据
            self.activity_data['month'] = pd.to_datetime(self.activity_data['timestamp']).dt.to_period('M')
            monthly_activity = self.activity_data.groupby(['user_id', 'month']).size().reset_index(name='activity_count')
            
            # 分析活动量变化与角色转换的相关性
            transition_users = []
            for user_id, user_data in temporal_data.groupby('user_id'):
                user_data = user_data.sort_values('time_period')
                for i in range(len(user_data) - 1):
                    if user_data.iloc[i]['role'] != user_data.iloc[i + 1]['role']:
                        transition_users.append({
                            'user_id': user_id,
                            'transition_period': user_data.iloc[i + 1]['time_period'],
                            'from_role': user_data.iloc[i]['role'],
                            'to_role': user_data.iloc[i + 1]['role']
                        })
            
            activity_triggers['transition_count'] = len(transition_users)
            activity_triggers['users_with_transitions'] = len(set(u['user_id'] for u in transition_users))
        
        return activity_triggers
    
    def _analyze_temporal_triggers(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """分析时间基础的触发模式"""
        temporal_triggers = {}
        
        # 分析转换发生的时间模式
        transition_times = []
        
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            for i in range(len(user_data) - 1):
                if user_data.iloc[i]['role'] != user_data.iloc[i + 1]['role']:
                    transition_times.append(user_data.iloc[i + 1]['time_period'])
        
        if transition_times:
            # 按月份分析转换频率
            transition_months = [t.month for t in transition_times]
            month_distribution = dict(Counter(transition_months))
            
            # 按季度分析
            transition_quarters = [f"Q{(t.month-1)//3 + 1}" for t in transition_times]
            quarter_distribution = dict(Counter(transition_quarters))
            
            temporal_triggers['monthly_distribution'] = month_distribution
            temporal_triggers['quarterly_distribution'] = quarter_distribution
            temporal_triggers['total_transitions'] = len(transition_times)
        
        return temporal_triggers
    
    def _analyze_sequence_triggers(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """分析序列模式触发因素"""
        sequence_triggers = {}
        
        # 分析导致转换的前置角色模式
        pre_transition_patterns = defaultdict(int)
        
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            roles = user_data['role'].tolist()
            
            for i in range(2, len(roles)):
                if roles[i-2] != roles[i-1] or roles[i-1] != roles[i]:
                    # 发生了转换，记录前置模式
                    pattern = f"{roles[i-2]} -> {roles[i-1]} -> {roles[i]}"
                    pre_transition_patterns[pattern] += 1
        
        # 找到最常见的转换模式
        common_patterns = sorted(pre_transition_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        sequence_triggers['common_transition_patterns'] = [
            {'pattern': pattern, 'count': count} for pattern, count in common_patterns
        ]
        
        return sequence_triggers
    
    def _predict_role_transitions(self, temporal_data: pd.DataFrame) -> Dict[str, Any]:
        """预测角色转换"""
        logger.info("构建角色转换预测模型...")
        
        prediction_results = {}
        
        try:
            # 准备特征和标签
            features, labels = self._prepare_prediction_data(temporal_data)
            
            if len(features) > 10:  # 需要足够的数据
                # 划分训练测试集
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.3, random_state=42, stratify=labels
                )
                
                # 训练随机森林模型
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                # 预测和评估
                y_pred = model.predict(X_test)
                
                # 生成分类报告
                class_report = classification_report(y_test, y_pred, output_dict=True)
                
                # 特征重要性
                feature_importance = dict(zip(
                    [f'feature_{i}' for i in range(len(features[0]))],
                    model.feature_importances_
                ))
                
                prediction_results = {
                    'model_accuracy': model.score(X_test, y_test),
                    'classification_report': class_report,
                    'feature_importance': feature_importance,
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                # 存储模型
                self.prediction_model = model
                
            else:
                prediction_results['error'] = 'Insufficient data for prediction model'
        
        except Exception as e:
            logger.error(f"预测模型构建失败: {e}")
            prediction_results['error'] = str(e)
        
        return prediction_results
    
    def _prepare_prediction_data(self, temporal_data: pd.DataFrame) -> Tuple[List, List]:
        """准备预测模型的特征和标签数据"""
        features = []
        labels = []
        
        for user_id, user_data in temporal_data.groupby('user_id'):
            user_data = user_data.sort_values('time_period')
            
            for i in range(len(user_data) - 1):
                current_row = user_data.iloc[i]
                next_row = user_data.iloc[i + 1]
                
                # 特征：当前角色、时期、用户统计信息等
                feature_vector = [
                    list(self.role_mapping.values()).index(current_row['role']),  # 当前角色编码
                    i,  # 时期索引
                    len(user_data),  # 用户总活跃时期
                ]
                
                # 添加用户特征（如果可用）
                if user_id in self.user_roles_df['user_id'].values:
                    user_features = self.user_roles_df[self.user_roles_df['user_id'] == user_id].iloc[0]
                    feature_vector.extend([
                        user_features.get('pr_count', 0),
                        user_features.get('issue_count', 0),
                        user_features.get('star_count', 0),
                        user_features.get('code_focus_ratio', 0),
                        user_features.get('interaction_diversity', 0)
                    ])
                else:
                    feature_vector.extend([0, 0, 0, 0, 0])
                
                features.append(feature_vector)
                labels.append(list(self.role_mapping.values()).index(next_row['role']))
        
        return features, labels
    
    def _calculate_transition_summary(self, 
                                    transition_matrix: pd.DataFrame,
                                    stability_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """计算转换分析摘要统计"""
        summary = {}
        
        if not transition_matrix.empty:
            # 转换矩阵统计
            summary['total_possible_transitions'] = len(transition_matrix) ** 2
            summary['observed_transitions'] = (transition_matrix > 0).sum().sum()
            summary['transition_diversity'] = summary['observed_transitions'] / summary['total_possible_transitions']
            
            # 最稳定的角色（对角线元素最大）
            diagonal_values = np.diag(transition_matrix.values)
            most_stable_idx = np.argmax(diagonal_values)
            summary['most_stable_role'] = transition_matrix.index[most_stable_idx]
            summary['highest_stability_score'] = diagonal_values[most_stable_idx]
            
            # 最不稳定的角色
            least_stable_idx = np.argmin(diagonal_values)
            summary['least_stable_role'] = transition_matrix.index[least_stable_idx]
            summary['lowest_stability_score'] = diagonal_values[least_stable_idx]
        
        # 稳定性分析统计
        if stability_analysis:
            retention_rates = [metrics['retention_rate'] for metrics in stability_analysis.values()]
            if retention_rates:
                summary['avg_retention_rate'] = np.mean(retention_rates)
                summary['std_retention_rate'] = np.std(retention_rates)
        
        return summary
    
    def _save_transition_results(self, results: Dict[str, Any]):
        """保存转换分析结果"""
        # 保存转换矩阵
        if 'transition_matrix' in results and not results['transition_matrix'].empty:
            matrix_path = ANALYSIS_OUTPUT_DIR / "role_transition_matrix.csv"
            results['transition_matrix'].to_csv(matrix_path, encoding='utf-8-sig')
            logger.info(f"转换矩阵已保存至: {matrix_path}")
        
        # 保存详细结果
        import json
        
        # 处理不能JSON序列化的对象
        json_results = {}
        for key, value in results.items():
            if key == 'transition_matrix':
                json_results[key] = value.to_dict() if hasattr(value, 'to_dict') else str(value)
            elif isinstance(value, dict):
                json_results[key] = value
            else:
                json_results[key] = str(value)
        
        results_path = ANALYSIS_OUTPUT_DIR / "role_transition_analysis.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"角色转换分析结果已保存至: {results_path}")
    
    def visualize_transition_patterns(self, results: Dict[str, Any]):
        """可视化角色转换模式"""
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 转换矩阵热力图
        if 'transition_matrix' in results and not results['transition_matrix'].empty:
            transition_matrix = results['transition_matrix']
            sns.heatmap(transition_matrix, annot=True, cmap='Blues', ax=axes[0,0], fmt='.3f')
            axes[0,0].set_title('Role Transition Matrix', fontsize=VISUALIZATION_CONFIG["title_font_size"])
            axes[0,0].set_xlabel('To Role')
            axes[0,0].set_ylabel('From Role')
        
        # 2. 角色稳定性
        if 'stability_analysis' in results:
            stability = results['stability_analysis']
            roles = list(stability.keys())
            retention_rates = [stability[role]['retention_rate'] for role in roles]
            
            bars = axes[0,1].bar(range(len(roles)), retention_rates, color='lightcoral', alpha=0.7)
            axes[0,1].set_title('Role Stability (Retention Rates)', fontsize=VISUALIZATION_CONFIG["title_font_size"])
            axes[0,1].set_xlabel('Roles')
            axes[0,1].set_ylabel('Retention Rate')
            axes[0,1].set_xticks(range(len(roles)))
            axes[0,1].set_xticklabels(roles, rotation=45, ha='right')
            
            # 添加数值标签
            for bar, rate in zip(bars, retention_rates):
                axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                              f'{rate:.2f}', ha='center', va='bottom')
        
        # 3. 路径长度分布
        if 'pathway_analysis' in results and 'path_length_distribution' in results['pathway_analysis']:
            path_dist = results['pathway_analysis']['path_length_distribution']
            if 'length_counts' in path_dist and path_dist['length_counts']:
                lengths = list(path_dist['length_counts'].keys())
                counts = list(path_dist['length_counts'].values())
                
                axes[1,0].bar(lengths, counts, color='lightgreen', alpha=0.7)
                axes[1,0].set_title('Path Length Distribution', fontsize=VISUALIZATION_CONFIG["title_font_size"])
                axes[1,0].set_xlabel('Number of Different Roles')
                axes[1,0].set_ylabel('Number of Users')
        
        # 4. 常见转换序列
        if 'pathway_analysis' in results and 'common_sequences' in results['pathway_analysis']:
            common_seqs = results['pathway_analysis']['common_sequences'][:10]  # 前10个
            if common_seqs:
                seq_labels = [' -> '.join(seq['sequence']) for seq in common_seqs]
                seq_counts = [seq['count'] for seq in common_seqs]
                
                y_pos = np.arange(len(seq_labels))
                axes[1,1].barh(y_pos, seq_counts, color='gold', alpha=0.7)
                axes[1,1].set_title('Most Common Transition Sequences', fontsize=VISUALIZATION_CONFIG["title_font_size"])
                axes[1,1].set_xlabel('Frequency')
                axes[1,1].set_yticks(y_pos)
                axes[1,1].set_yticklabels(seq_labels, fontsize=8)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "role_transition_visualization.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"角色转换可视化图表已保存至: {save_path}")
        
        plt.show()
    
    def generate_transition_report(self, results: Dict[str, Any]) -> str:
        """生成角色转换分析报告"""
        report_lines = [
            "=" * 60,
            "角色转换路径分析报告",
            "Role Transition Analysis Report", 
            "=" * 60,
            ""
        ]
        
        # 摘要统计
        if 'summary_statistics' in results:
            summary = results['summary_statistics']
            report_lines.extend([
                "📊 摘要统计 (Summary Statistics):",
                f"  - 观察到的转换类型: {summary.get('observed_transitions', 'N/A')}/{summary.get('total_possible_transitions', 'N/A')}",
                f"  - 转换多样性: {summary.get('transition_diversity', 'N/A'):.3f}",
                f"  - 最稳定角色: {summary.get('most_stable_role', 'N/A')} (稳定性: {summary.get('highest_stability_score', 'N/A'):.3f})",
                f"  - 最不稳定角色: {summary.get('least_stable_role', 'N/A')} (稳定性: {summary.get('lowest_stability_score', 'N/A'):.3f})",
                f"  - 平均保留率: {summary.get('avg_retention_rate', 'N/A'):.3f}",
                ""
            ])
        
        # 路径分析结果
        if 'pathway_analysis' in results:
            pathway = results['pathway_analysis']
            report_lines.extend([
                "🛤️  转换路径分析 (Pathway Analysis):",
                ""
            ])
            
            if 'novice_expert_paths' in pathway:
                novice_expert = pathway['novice_expert_paths']
                report_lines.extend([
                    f"新手到专家路径:",
                    f"  - 发现路径数: {novice_expert.get('total_pathways_found', 0)}",
                    f"  - 平均路径长度: {novice_expert.get('average_pathway_length', 0):.1f}",
                    f"  - 平均转换时长: {novice_expert.get('average_transition_duration', 0):.1f} 个时期",
                    ""
                ])
                
                if 'most_common_pathways' in novice_expert and novice_expert['most_common_pathways']:
                    report_lines.append("最常见的新手→专家路径:")
                    for path in novice_expert['most_common_pathways'][:5]:
                        report_lines.append(f"  - {' -> '.join(path['pathway'])} ({path['count']} 次)")
                    report_lines.append("")
            
            if 'bridge_roles' in pathway:
                bridge_roles = pathway['bridge_roles']
                if bridge_roles:
                    sorted_bridges = sorted(bridge_roles.items(), key=lambda x: x[1], reverse=True)
                    report_lines.extend([
                        "桥梁角色 (Bridge Roles):",
                        *(f"  - {role}: {score:.3f}" for role, score in sorted_bridges[:5]),
                        ""
                    ])
        
        # 预测结果
        if 'prediction_results' in results and 'model_accuracy' in results['prediction_results']:
            pred_results = results['prediction_results']
            report_lines.extend([
                "🔮 转换预测模型 (Prediction Model):",
                f"  - 模型准确率: {pred_results.get('model_accuracy', 'N/A'):.3f}",
                f"  - 训练样本数: {pred_results.get('training_samples', 'N/A')}",
                f"  - 测试样本数: {pred_results.get('test_samples', 'N/A')}",
                ""
            ])
            
            if 'feature_importance' in pred_results:
                importance = pred_results['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                report_lines.extend([
                    "重要特征 (Top Features):",
                    *(f"  - {feature}: {importance:.3f}" for feature, importance in sorted_features[:5]),
                    ""
                ])
        
        report_lines.extend([
            "=" * 60,
            f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        # 保存报告
        report_path = ANALYSIS_OUTPUT_DIR / "role_transition_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"角色转换分析报告已保存至: {report_path}")
        
        return report


def main():
    """主函数入口"""
    logger.info("角色转换路径分析模块测试")
    
    print("角色转换路径分析器已实现以下功能:")
    print("1. 角色转换矩阵计算")
    print("2. 转换路径模式分析")
    print("3. 新手到专家路径识别")
    print("4. 角色稳定性分析")
    print("5. 转换触发因素识别")
    print("6. 转换预测模型")
    print("7. 可视化和报告生成")
    print("\n使用方法:")
    print("analyzer = RoleTransitionAnalyzer(user_roles_df, activity_data)")
    print("results = analyzer.analyze_role_transitions(temporal_data)")
    print("analyzer.visualize_transition_patterns(results)")
    print("report = analyzer.generate_transition_report(results)")


if __name__ == "__main__":
    main()


