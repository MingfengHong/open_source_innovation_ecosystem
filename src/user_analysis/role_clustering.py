"""
用户角色聚类模块 (增强版)
支持多种聚类算法，包括K-Means、GMM、层次聚类等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 聚类算法
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA

# 评估工具
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as sch

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, VISUALIZATION_CONFIG, MODEL_CONFIG

# 设置日志
logger = setup_logger(__name__)


class MultiAlgorithmClustering:
    """多算法聚类器"""
    
    def __init__(self, 
                 features_df: pd.DataFrame,
                 feature_columns: List[str] = None,
                 random_seed: int = 42):
        """
        初始化多算法聚类器
        
        Args:
            features_df: 特征数据框
            feature_columns: 特征列名列表
            random_seed: 随机种子
        """
        self.features_df = features_df.copy()
        self.random_seed = random_seed
        
        # 确定特征列
        if feature_columns is None:
            self.feature_columns = [col for col in features_df.columns 
                                  if col not in ['user_id', 'login']]
        else:
            self.feature_columns = feature_columns
        
        # 提取特征矩阵
        self.X = features_df[self.feature_columns].values
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        # 存储聚类结果
        self.clustering_results = {}
        self.evaluation_results = {}
        
        logger.info(f"初始化聚类器: {len(features_df)} 个样本, {len(self.feature_columns)} 个特征")
    
    def determine_optimal_k(self, 
                          k_range: range = None,
                          methods: List[str] = None) -> Dict[str, int]:
        """
        使用多种方法确定最优K值
        
        Args:
            k_range: K值范围
            methods: 评估方法列表
            
        Returns:
            Dict[str, int]: 每种方法推荐的最优K值
        """
        if k_range is None:
            k_range = range(2, 16)
        
        if methods is None:
            methods = ['elbow', 'silhouette', 'calinski_harabasz']
        
        logger.info(f"使用多种方法确定最优K值: {methods}")
        
        # 存储评估结果
        evaluation_scores = {
            'k_values': list(k_range),
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        # 测试不同的K值
        for k in tqdm(k_range, desc="评估K值"):
            # K-Means聚类
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_scaled)
            
            # 计算评估指标
            evaluation_scores['inertia'].append(kmeans.inertia_)
            
            if k > 1:  # 轮廓系数需要至少2个聚类
                evaluation_scores['silhouette'].append(silhouette_score(self.X_scaled, cluster_labels))
                evaluation_scores['calinski_harabasz'].append(calinski_harabasz_score(self.X_scaled, cluster_labels))
                evaluation_scores['davies_bouldin'].append(davies_bouldin_score(self.X_scaled, cluster_labels))
            else:
                evaluation_scores['silhouette'].append(0)
                evaluation_scores['calinski_harabasz'].append(0)
                evaluation_scores['davies_bouldin'].append(float('inf'))
        
        # 可视化评估结果
        self._visualize_k_evaluation(evaluation_scores)
        
        # 确定最优K值
        optimal_k = {}
        
        if 'elbow' in methods:
            optimal_k['elbow'] = self._find_elbow_point(evaluation_scores['k_values'], 
                                                       evaluation_scores['inertia'])
        
        if 'silhouette' in methods:
            max_silhouette_idx = np.argmax(evaluation_scores['silhouette'])
            optimal_k['silhouette'] = evaluation_scores['k_values'][max_silhouette_idx]
        
        if 'calinski_harabasz' in methods:
            max_ch_idx = np.argmax(evaluation_scores['calinski_harabasz'])
            optimal_k['calinski_harabasz'] = evaluation_scores['k_values'][max_ch_idx]
        
        logger.info(f"最优K值推荐: {optimal_k}")
        return optimal_k
    
    def perform_kmeans_clustering(self, n_clusters: int) -> np.ndarray:
        """
        执行K-Means聚类
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            np.ndarray: 聚类标签
        """
        logger.info(f"执行K-Means聚类 (K={n_clusters})...")
        
        kmeans_params = MODEL_CONFIG["clustering"]["kmeans_params"]
        kmeans = KMeans(n_clusters=n_clusters, **kmeans_params)
        cluster_labels = kmeans.fit_predict(self.X_scaled)
        
        # 存储结果
        self.clustering_results['kmeans'] = {
            'labels': cluster_labels,
            'model': kmeans,
            'n_clusters': n_clusters
        }
        
        return cluster_labels
    
    def perform_gmm_clustering(self, n_components: int) -> np.ndarray:
        """
        执行高斯混合模型聚类
        
        Args:
            n_components: 组件数量
            
        Returns:
            np.ndarray: 聚类标签
        """
        logger.info(f"执行GMM聚类 (n_components={n_components})...")
        
        gmm_params = MODEL_CONFIG["clustering"]["gmm_params"]
        gmm = GaussianMixture(n_components=n_components, **gmm_params)
        cluster_labels = gmm.fit_predict(self.X_scaled)
        
        # 存储结果
        self.clustering_results['gmm'] = {
            'labels': cluster_labels,
            'model': gmm,
            'n_components': n_components,
            'bic': gmm.bic(self.X_scaled),
            'aic': gmm.aic(self.X_scaled)
        }
        
        return cluster_labels
    
    def perform_hierarchical_clustering(self, 
                                      n_clusters: int,
                                      linkage_method: str = 'ward') -> np.ndarray:
        """
        执行层次聚类
        
        Args:
            n_clusters: 聚类数量
            linkage_method: 连接方法
            
        Returns:
            np.ndarray: 聚类标签
        """
        logger.info(f"执行层次聚类 (n_clusters={n_clusters}, linkage={linkage_method})...")
        
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        cluster_labels = hierarchical.fit_predict(self.X_scaled)
        
        # 存储结果
        self.clustering_results['hierarchical'] = {
            'labels': cluster_labels,
            'model': hierarchical,
            'n_clusters': n_clusters,
            'linkage_method': linkage_method
        }
        
        return cluster_labels
    
    def compare_all_algorithms(self, n_clusters: int) -> pd.DataFrame:
        """
        比较所有聚类算法
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            pd.DataFrame: 算法比较结果
        """
        logger.info(f"比较所有聚类算法 (n_clusters={n_clusters})...")
        
        # 执行所有算法
        algorithms = {
            'K-Means': self.perform_kmeans_clustering(n_clusters),
            'GMM': self.perform_gmm_clustering(n_clusters),
            'Hierarchical': self.perform_hierarchical_clustering(n_clusters)
        }
        
        # 评估结果
        comparison_data = []
        
        for algo_name, labels in algorithms.items():
            # 计算评估指标
            silhouette = silhouette_score(self.X_scaled, labels)
            calinski_harabasz = calinski_harabasz_score(self.X_scaled, labels)
            davies_bouldin = davies_bouldin_score(self.X_scaled, labels)
            
            # 计算聚类大小统计
            unique_labels, counts = np.unique(labels, return_counts=True)
            cluster_sizes = counts
            
            comparison_data.append({
                'algorithm': algo_name,
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin,
                'num_clusters': len(unique_labels),
                'largest_cluster_size': np.max(cluster_sizes),
                'smallest_cluster_size': np.min(cluster_sizes),
                'avg_cluster_size': np.mean(cluster_sizes),
                'cluster_size_std': np.std(cluster_sizes)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存比较结果
        output_path = ANALYSIS_OUTPUT_DIR / "clustering_algorithms_comparison.csv"
        comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"算法比较结果已保存至: {output_path}")
        
        # 可视化比较结果
        self._visualize_algorithm_comparison(comparison_df)
        
        return comparison_df
    
    def analyze_cluster_profiles(self, 
                               algorithm: str = 'kmeans',
                               include_visualization: bool = True) -> pd.DataFrame:
        """
        分析聚类结果的特征画像
        
        Args:
            algorithm: 使用的算法
            include_visualization: 是否生成可视化
            
        Returns:
            pd.DataFrame: 聚类画像数据框
        """
        if algorithm not in self.clustering_results:
            logger.error(f"算法 {algorithm} 的聚类结果不存在")
            return pd.DataFrame()
        
        labels = self.clustering_results[algorithm]['labels']
        
        # 添加聚类标签到数据框
        profile_df = self.features_df.copy()
        profile_df['cluster'] = labels
        
        # 计算每个聚类的特征均值
        cluster_profiles = profile_df.groupby('cluster')[self.feature_columns].agg(['mean', 'std'])
        
        # 计算聚类大小
        cluster_sizes = profile_df['cluster'].value_counts().sort_index()
        
        # 展平多级列名
        profile_summary = []
        for cluster_id in cluster_profiles.index:
            cluster_data = {'cluster_id': cluster_id, 'cluster_size': cluster_sizes[cluster_id]}
            
            for feature in self.feature_columns:
                cluster_data[f'{feature}_mean'] = cluster_profiles.loc[cluster_id, (feature, 'mean')]
                cluster_data[f'{feature}_std'] = cluster_profiles.loc[cluster_id, (feature, 'std')]
            
            profile_summary.append(cluster_data)
        
        profile_summary_df = pd.DataFrame(profile_summary)
        
        # 保存聚类画像
        output_path = ANALYSIS_OUTPUT_DIR / f"cluster_profiles_{algorithm}.csv"
        profile_summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"聚类画像已保存至: {output_path}")
        
        # 可视化聚类画像
        if include_visualization:
            self._visualize_cluster_profiles(cluster_profiles, algorithm)
            self._visualize_cluster_scatter(profile_df, algorithm)
        
        return profile_summary_df
    
    def export_clustering_results(self, algorithm: str = 'kmeans') -> pd.DataFrame:
        """
        导出聚类结果
        
        Args:
            algorithm: 使用的算法
            
        Returns:
            pd.DataFrame: 包含聚类标签的完整数据框
        """
        if algorithm not in self.clustering_results:
            logger.error(f"算法 {algorithm} 的聚类结果不存在")
            return pd.DataFrame()
        
        labels = self.clustering_results[algorithm]['labels']
        
        # 创建结果数据框
        result_df = self.features_df.copy()
        result_df['cluster'] = labels
        
        # 保存结果
        output_path = ANALYSIS_OUTPUT_DIR / f"user_roles_{algorithm}.csv"
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"聚类结果已保存至: {output_path}")
        
        return result_df
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """使用拐点法找到最优K值"""
        # 计算二阶差分
        diffs = np.diff(inertias)
        diff2 = np.diff(diffs)
        
        # 找到最大的二阶差分点
        elbow_idx = np.argmax(diff2) + 2  # +2是因为二阶差分会减少2个点
        return k_values[elbow_idx] if elbow_idx < len(k_values) else k_values[-1]
    
    def _visualize_k_evaluation(self, evaluation_scores: Dict):
        """可视化K值评估结果"""
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 肘部法则图
        axes[0,0].plot(evaluation_scores['k_values'], evaluation_scores['inertia'], 
                      marker='o', linestyle='-')
        axes[0,0].set_title('Elbow Method', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,0].set_xlabel('Number of Clusters (K)')
        axes[0,0].set_ylabel('Inertia (WCSS)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 轮廓系数
        axes[0,1].plot(evaluation_scores['k_values'], evaluation_scores['silhouette'], 
                      marker='s', linestyle='-', color='orange')
        axes[0,1].set_title('Silhouette Score', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,1].set_xlabel('Number of Clusters (K)')
        axes[0,1].set_ylabel('Silhouette Score')
        axes[0,1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz指数
        axes[1,0].plot(evaluation_scores['k_values'], evaluation_scores['calinski_harabasz'], 
                      marker='^', linestyle='-', color='green')
        axes[1,0].set_title('Calinski-Harabasz Index', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,0].set_xlabel('Number of Clusters (K)')
        axes[1,0].set_ylabel('Calinski-Harabasz Index')
        axes[1,0].grid(True, alpha=0.3)
        
        # Davies-Bouldin指数
        axes[1,1].plot(evaluation_scores['k_values'], evaluation_scores['davies_bouldin'], 
                      marker='d', linestyle='-', color='red')
        axes[1,1].set_title('Davies-Bouldin Index', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,1].set_xlabel('Number of Clusters (K)')
        axes[1,1].set_ylabel('Davies-Bouldin Index')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "clustering_k_evaluation.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"K值评估图表已保存至: {save_path}")
        
        plt.show()
    
    def _visualize_algorithm_comparison(self, comparison_df: pd.DataFrame):
        """可视化算法比较结果"""
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 轮廓系数比较
        axes[0,0].bar(comparison_df['algorithm'], comparison_df['silhouette_score'])
        axes[0,0].set_title('Silhouette Score Comparison', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,0].set_ylabel('Silhouette Score')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz指数比较
        axes[0,1].bar(comparison_df['algorithm'], comparison_df['calinski_harabasz_score'])
        axes[0,1].set_title('Calinski-Harabasz Score Comparison', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[0,1].set_ylabel('Calinski-Harabasz Score')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Davies-Bouldin指数比较
        axes[1,0].bar(comparison_df['algorithm'], comparison_df['davies_bouldin_score'])
        axes[1,0].set_title('Davies-Bouldin Score Comparison', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,0].set_ylabel('Davies-Bouldin Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 聚类大小标准差比较
        axes[1,1].bar(comparison_df['algorithm'], comparison_df['cluster_size_std'])
        axes[1,1].set_title('Cluster Size Standard Deviation', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        axes[1,1].set_ylabel('Cluster Size Std')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "clustering_algorithms_comparison.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"算法比较图表已保存至: {save_path}")
        
        plt.show()
    
    def _visualize_cluster_profiles(self, cluster_profiles: pd.DataFrame, algorithm: str):
        """可视化聚类画像热力图"""
        plt.figure(figsize=(16, 8))
        
        # 提取均值数据用于热力图
        mean_data = cluster_profiles.xs('mean', level=1, axis=1)
        
        # 标准化数据用于更好的可视化
        mean_data_scaled = (mean_data - mean_data.mean()) / mean_data.std()
        
        sns.heatmap(mean_data_scaled, 
                   cmap=VISUALIZATION_CONFIG["color_palette"], 
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': 'Standardized Mean Value'})
        
        plt.title(f'Cluster Profiles Heatmap ({algorithm.upper()})', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.xlabel('Features')
        plt.ylabel('Cluster ID')
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / f"cluster_profiles_heatmap_{algorithm}.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"聚类画像热力图已保存至: {save_path}")
        
        plt.show()
    
    def _visualize_cluster_scatter(self, profile_df: pd.DataFrame, algorithm: str):
        """可视化聚类散点图（使用PCA降维）"""
        # PCA降维到2D
        pca = PCA(n_components=2, random_state=self.random_seed)
        X_pca = pca.fit_transform(self.X_scaled)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图
        unique_clusters = sorted(profile_df['cluster'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = profile_df['cluster'] == cluster_id
            plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', 
                       alpha=0.7, s=50)
        
        plt.title(f'Cluster Visualization using PCA ({algorithm.upper()})', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / f"cluster_scatter_pca_{algorithm}.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"聚类散点图已保存至: {save_path}")
        
        plt.show()


def main():
    """主函数入口"""
    # 加载用户行为特征数据
    features_path = ANALYSIS_OUTPUT_DIR / "user_behavior_features.csv"
    
    try:
        features_df = pd.read_csv(features_path)
        logger.info(f"成功加载用户特征数据: {len(features_df)} 个用户")
    except FileNotFoundError:
        logger.error(f"未找到用户特征文件: {features_path}")
        return
    
    # 创建多算法聚类器
    clusterer = MultiAlgorithmClustering(features_df)
    
    # 确定最优K值
    optimal_k = clusterer.determine_optimal_k()
    recommended_k = optimal_k.get('silhouette', 6)  # 使用轮廓系数推荐的K值，默认6
    
    logger.info(f"使用推荐的K值: {recommended_k}")
    
    # 比较所有算法
    comparison_df = clusterer.compare_all_algorithms(recommended_k)
    print("\n算法比较结果:")
    print(comparison_df)
    
    # 分析最佳算法的聚类画像
    best_algorithm = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'algorithm'].lower()
    logger.info(f"最佳算法: {best_algorithm}")
    
    profile_df = clusterer.analyze_cluster_profiles(best_algorithm)
    result_df = clusterer.export_clustering_results(best_algorithm)
    
    logger.info("多算法聚类分析完成！")


if __name__ == "__main__":
    main()
