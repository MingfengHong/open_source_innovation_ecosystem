"""
中心性度量模块
提供多种网络中心性计算功能，包括度中心性、介数中心性、特征向量中心性等
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CentralityCalculator:
    """网络中心性计算器"""
    
    def __init__(self, graph: nx.Graph):
        """
        初始化中心性计算器
        
        Args:
            graph: NetworkX图对象
        """
        self.graph = graph
        self.centrality_cache = {}
    
    def calculate_degree_centrality(self) -> Dict[str, float]:
        """
        计算度中心性
        度中心性衡量节点的直接连接数量
        
        Returns:
            Dict[str, float]: 节点ID到度中心性的映射
        """
        if 'degree' not in self.centrality_cache:
            logger.info("正在计算度中心性...")
            self.centrality_cache['degree'] = nx.degree_centrality(self.graph)
        return self.centrality_cache['degree']
    
    def calculate_betweenness_centrality(self, 
                                       k: Optional[int] = None, 
                                       normalized: bool = True) -> Dict[str, float]:
        """
        计算介数中心性
        介数中心性衡量节点作为其他节点间最短路径桥梁的重要性
        
        Args:
            k: 使用k个节点进行近似计算，None表示使用所有节点
            normalized: 是否标准化结果
            
        Returns:
            Dict[str, float]: 节点ID到介数中心性的映射
        """
        cache_key = f'betweenness_k{k}_norm{normalized}'
        if cache_key not in self.centrality_cache:
            logger.info(f"正在计算介数中心性 (k={k}, normalized={normalized})...")
            self.centrality_cache[cache_key] = nx.betweenness_centrality(
                self.graph, k=k, normalized=normalized
            )
        return self.centrality_cache[cache_key]
    
    def calculate_eigenvector_centrality(self, 
                                       max_iter: int = 100,
                                       tol: float = 1e-6) -> Dict[str, float]:
        """
        计算特征向量中心性
        特征向量中心性衡量节点连接到其他重要节点的程度
        
        Args:
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            
        Returns:
            Dict[str, float]: 节点ID到特征向量中心性的映射
        """
        cache_key = f'eigenvector_iter{max_iter}_tol{tol}'
        if cache_key not in self.centrality_cache:
            logger.info("正在计算特征向量中心性...")
            try:
                self.centrality_cache[cache_key] = nx.eigenvector_centrality(
                    self.graph, max_iter=max_iter, tol=tol
                )
            except nx.PowerIterationFailedConvergence:
                logger.warning("特征向量中心性计算未收敛，使用较低精度结果")
                try:
                    self.centrality_cache[cache_key] = nx.eigenvector_centrality(
                        self.graph, max_iter=max_iter*2, tol=tol*10
                    )
                except:
                    logger.error("特征向量中心性计算失败，返回零值")
                    self.centrality_cache[cache_key] = {node: 0.0 for node in self.graph.nodes()}
        return self.centrality_cache[cache_key]
    
    def calculate_closeness_centrality(self) -> Dict[str, float]:
        """
        计算接近中心性
        接近中心性衡量节点到其他所有节点的平均最短距离的倒数
        
        Returns:
            Dict[str, float]: 节点ID到接近中心性的映射
        """
        if 'closeness' not in self.centrality_cache:
            logger.info("正在计算接近中心性...")
            self.centrality_cache['closeness'] = nx.closeness_centrality(self.graph)
        return self.centrality_cache['closeness']
    
    def calculate_pagerank(self, 
                          alpha: float = 0.85, 
                          max_iter: int = 100,
                          tol: float = 1e-6) -> Dict[str, float]:
        """
        计算PageRank中心性
        PageRank算法衡量节点的重要性，考虑了连接节点的质量
        
        Args:
            alpha: 阻尼参数
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            
        Returns:
            Dict[str, float]: 节点ID到PageRank的映射
        """
        cache_key = f'pagerank_alpha{alpha}_iter{max_iter}_tol{tol}'
        if cache_key not in self.centrality_cache:
            logger.info("正在计算PageRank...")
            self.centrality_cache[cache_key] = nx.pagerank(
                self.graph, alpha=alpha, max_iter=max_iter, tol=tol
            )
        return self.centrality_cache[cache_key]
    
    def calculate_all_centralities(self, 
                                 include_betweenness: bool = True,
                                 betweenness_k: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        计算所有中心性度量
        
        Args:
            include_betweenness: 是否计算介数中心性（计算量较大）
            betweenness_k: 介数中心性的近似参数
            
        Returns:
            Dict[str, Dict[str, float]]: 中心性类型到节点中心性映射的字典
        """
        results = {}
        
        # 计算度中心性
        results['degree'] = self.calculate_degree_centrality()
        
        # 计算特征向量中心性
        results['eigenvector'] = self.calculate_eigenvector_centrality()
        
        # 计算接近中心性
        results['closeness'] = self.calculate_closeness_centrality()
        
        # 计算PageRank
        results['pagerank'] = self.calculate_pagerank()
        
        # 可选计算介数中心性
        if include_betweenness:
            results['betweenness'] = self.calculate_betweenness_centrality(k=betweenness_k)
        
        return results
    
    def get_top_nodes(self, 
                     centrality_type: str, 
                     top_k: int = 10) -> List[Tuple[str, float]]:
        """
        获取指定中心性度量的前K个节点
        
        Args:
            centrality_type: 中心性类型
            top_k: 返回前K个节点
            
        Returns:
            List[Tuple[str, float]]: 节点ID和中心性值的元组列表
        """
        centrality_map = {
            'degree': self.calculate_degree_centrality,
            'betweenness': self.calculate_betweenness_centrality,
            'eigenvector': self.calculate_eigenvector_centrality,
            'closeness': self.calculate_closeness_centrality,
            'pagerank': self.calculate_pagerank
        }
        
        if centrality_type not in centrality_map:
            raise ValueError(f"不支持的中心性类型: {centrality_type}")
        
        centrality_scores = centrality_map[centrality_type]()
        sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes[:top_k]
    
    def clear_cache(self):
        """清空缓存"""
        self.centrality_cache.clear()


def calculate_centrality_for_core_team(graph: nx.Graph, 
                                     core_team_nodes: List[str],
                                     centrality_types: List[str] = None) -> Dict[str, float]:
    """
    计算核心团队的平均中心性
    
    Args:
        graph: NetworkX图对象
        core_team_nodes: 核心团队节点ID列表
        centrality_types: 要计算的中心性类型列表
        
    Returns:
        Dict[str, float]: 中心性类型到平均值的映射
    """
    if centrality_types is None:
        centrality_types = ['degree', 'betweenness', 'eigenvector', 'closeness']
    
    calculator = CentralityCalculator(graph)
    results = {}
    
    for centrality_type in centrality_types:
        try:
            if centrality_type == 'degree':
                centrality = calculator.calculate_degree_centrality()
            elif centrality_type == 'betweenness':
                centrality = calculator.calculate_betweenness_centrality()
            elif centrality_type == 'eigenvector':
                centrality = calculator.calculate_eigenvector_centrality()
            elif centrality_type == 'closeness':
                centrality = calculator.calculate_closeness_centrality()
            else:
                logger.warning(f"未知的中心性类型: {centrality_type}")
                continue
            
            # 计算核心团队的平均中心性
            core_centralities = [centrality.get(node, 0) for node in core_team_nodes if node in graph.nodes()]
            avg_centrality = np.mean(core_centralities) if core_centralities else 0
            results[f'avg_core_{centrality_type}_centrality'] = avg_centrality
            
        except Exception as e:
            logger.error(f"计算{centrality_type}中心性时出错: {e}")
            results[f'avg_core_{centrality_type}_centrality'] = 0
    
    return results


def analyze_centrality_distribution(centralities: Dict[str, float], 
                                  node_types: Dict[str, str] = None) -> Dict[str, float]:
    """
    分析中心性分布的统计特征
    
    Args:
        centralities: 节点中心性字典
        node_types: 节点类型字典（可选）
        
    Returns:
        Dict[str, float]: 统计特征字典
    """
    values = list(centralities.values())
    
    stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'median': np.median(values),
        'min': np.min(values),
        'max': np.max(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
        'gini': calculate_gini_coefficient(values)
    }
    
    # 如果提供了节点类型信息，计算不同类型节点的平均中心性
    if node_types:
        type_centralities = {}
        for node, centrality in centralities.items():
            node_type = node_types.get(node, 'unknown')
            if node_type not in type_centralities:
                type_centralities[node_type] = []
            type_centralities[node_type].append(centrality)
        
        for node_type, type_values in type_centralities.items():
            stats[f'mean_{node_type}'] = np.mean(type_values)
    
    return stats


def calculate_gini_coefficient(values: List[float]) -> float:
    """
    计算基尼系数，衡量中心性分布的不平等程度
    
    Args:
        values: 中心性值列表
        
    Returns:
        float: 基尼系数
    """
    if not values:
        return 0
    
    values = np.array(sorted(values))
    n = len(values)
    cumsum = np.cumsum(values)
    
    if cumsum[-1] == 0:
        return 0
    
    return (2 * np.sum((np.arange(1, n + 1) * values))) / (n * cumsum[-1]) - (n + 1) / n
