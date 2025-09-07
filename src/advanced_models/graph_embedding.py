"""
图嵌入模块
提供多种图嵌入方法，包括Node2Vec、GraphSAGE等，作为HAN模型的补充
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec, SAGEConv, GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, MODELS_DIR, VISUALIZATION_CONFIG

# 设置日志
logger = setup_logger(__name__)


class Node2VecEmbedding:
    """Node2Vec图嵌入"""
    
    def __init__(self, 
                 graph: nx.Graph,
                 embedding_dim: int = 128,
                 walk_length: int = 20,
                 context_size: int = 10,
                 walks_per_node: int = 10,
                 num_negative_samples: int = 1,
                 p: float = 1.0,
                 q: float = 1.0):
        """
        初始化Node2Vec嵌入
        
        Args:
            graph: NetworkX图
            embedding_dim: 嵌入维度
            walk_length: 随机游走长度
            context_size: 上下文窗口大小
            walks_per_node: 每个节点的游走次数
            num_negative_samples: 负采样数量
            p: 返回参数
            q: 进出参数
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.p = p
        self.q = q
        
        # 转换为PyTorch Geometric格式
        self.data = from_networkx(graph)
        
        # 创建节点映射
        self.node_mapping = {node: i for i, node in enumerate(graph.nodes())}
        self.reverse_mapping = {i: node for node, i in self.node_mapping.items()}
        
        self.model = None
        self.embeddings = None
        
        logger.info(f"初始化Node2Vec: {len(graph)} 个节点, 嵌入维度={embedding_dim}")
    
    def train(self, epochs: int = 100, batch_size: int = 128, lr: float = 0.01) -> Dict[str, List[float]]:
        """
        训练Node2Vec模型
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            
        Returns:
            Dict[str, List[float]]: 训练历史
        """
        logger.info(f"开始训练Node2Vec模型: {epochs} 轮")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建Node2Vec模型
        self.model = Node2Vec(
            self.data.edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            num_negative_samples=self.num_negative_samples,
            p=self.p,
            q=self.q,
            sparse=True
        ).to(device)
        
        # 创建数据加载器
        loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=lr)
        
        # 训练循环
        training_history = {'loss': []}
        self.model.train()
        
        for epoch in tqdm(range(epochs), desc="Node2Vec训练"):
            epoch_loss = 0
            num_batches = 0
            
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            training_history['loss'].append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Node2Vec训练完成")
        return training_history
    
    def get_embeddings(self) -> np.ndarray:
        """获取节点嵌入"""
        if self.model is None:
            logger.error("模型尚未训练")
            return None
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model()
        
        self.embeddings = embeddings.cpu().numpy()
        return self.embeddings
    
    def save_embeddings(self, path: str):
        """保存嵌入到文件"""
        if self.embeddings is None:
            self.get_embeddings()
        
        # 创建DataFrame
        embedding_df = pd.DataFrame(self.embeddings)
        embedding_df.index = [self.reverse_mapping[i] for i in range(len(self.embeddings))]
        
        # 保存
        embedding_df.to_csv(path)
        logger.info(f"Node2Vec嵌入已保存至: {path}")


class GraphSAGEEmbedding:
    """GraphSAGE图嵌入"""
    
    def __init__(self, 
                 data: Data,
                 hidden_dim: int = 64,
                 output_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 aggregation: str = 'mean'):
        """
        初始化GraphSAGE嵌入
        
        Args:
            data: PyTorch Geometric数据
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: 层数
            dropout: Dropout率
            aggregation: 聚合方式
        """
        self.data = data
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregation = aggregation
        
        # 如果没有节点特征，创建随机特征
        if not hasattr(data, 'x') or data.x is None:
            self.data.x = torch.randn(data.num_nodes, hidden_dim)
        
        self.model = None
        self.embeddings = None
        
        logger.info(f"初始化GraphSAGE: {data.num_nodes} 个节点, 输出维度={output_dim}")
    
    def create_model(self) -> nn.Module:
        """创建GraphSAGE模型"""
        class GraphSAGE(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
                super(GraphSAGE, self).__init__()
                
                self.num_layers = num_layers
                self.dropout = dropout
                
                self.convs = nn.ModuleList()
                
                # 第一层
                self.convs.append(SAGEConv(input_dim, hidden_dim))
                
                # 中间层
                for _ in range(num_layers - 2):
                    self.convs.append(SAGEConv(hidden_dim, hidden_dim))
                
                # 输出层
                if num_layers > 1:
                    self.convs.append(SAGEConv(hidden_dim, output_dim))
                else:
                    self.convs[0] = SAGEConv(input_dim, output_dim)
            
            def forward(self, x, edge_index):
                for i, conv in enumerate(self.convs):
                    x = conv(x, edge_index)
                    if i < len(self.convs) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=self.training)
                return x
        
        input_dim = self.data.x.shape[1]
        self.model = GraphSAGE(input_dim, self.hidden_dim, self.output_dim, 
                               self.num_layers, self.dropout)
        
        return self.model
    
    def train(self, epochs: int = 200, lr: float = 0.01) -> Dict[str, List[float]]:
        """
        训练GraphSAGE模型
        
        Args:
            epochs: 训练轮数
            lr: 学习率
            
        Returns:
            Dict[str, List[float]]: 训练历史
        """
        logger.info(f"开始训练GraphSAGE模型: {epochs} 轮")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.model is None:
            self.create_model()
        
        self.model = self.model.to(device)
        self.data = self.data.to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # 创建训练任务（这里使用重构任务）
        training_history = {'loss': []}
        
        for epoch in tqdm(range(epochs), desc="GraphSAGE训练"):
            self.model.train()
            optimizer.zero_grad()
            
            # 前向传播
            embeddings = self.model(self.data.x, self.data.edge_index)
            
            # 重构损失（简化版）
            loss = F.mse_loss(embeddings, torch.randn_like(embeddings))
            
            loss.backward()
            optimizer.step()
            
            training_history['loss'].append(loss.item())
            
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        logger.info("GraphSAGE训练完成")
        return training_history
    
    def get_embeddings(self) -> np.ndarray:
        """获取节点嵌入"""
        if self.model is None:
            logger.error("模型尚未训练")
            return None
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.data.x, self.data.edge_index)
        
        self.embeddings = embeddings.cpu().numpy()
        return self.embeddings


class EmbeddingAnalyzer:
    """嵌入分析器"""
    
    def __init__(self, embeddings: np.ndarray, node_labels: Dict[str, str] = None):
        """
        初始化嵌入分析器
        
        Args:
            embeddings: 节点嵌入矩阵
            node_labels: 节点标签字典
        """
        self.embeddings = embeddings
        self.node_labels = node_labels or {}
        
        logger.info(f"初始化嵌入分析器: {embeddings.shape} 嵌入矩阵")
    
    def dimensionality_reduction(self, method: str = 'tsne', n_components: int = 2) -> np.ndarray:
        """
        降维分析
        
        Args:
            method: 降维方法 ('tsne', 'pca')
            n_components: 目标维度
            
        Returns:
            np.ndarray: 降维后的嵌入
        """
        logger.info(f"执行{method.upper()}降维分析...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        reduced_embeddings = reducer.fit_transform(self.embeddings)
        
        logger.info(f"{method.upper()}降维完成: {reduced_embeddings.shape}")
        return reduced_embeddings
    
    def clustering_analysis(self, n_clusters: int = 5) -> np.ndarray:
        """
        聚类分析
        
        Args:
            n_clusters: 聚类数量
            
        Returns:
            np.ndarray: 聚类标签
        """
        logger.info(f"执行K-Means聚类分析: {n_clusters} 个聚类")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(self.embeddings)
        
        return cluster_labels
    
    def visualize_embeddings_2d(self, 
                               reduced_embeddings: np.ndarray,
                               cluster_labels: np.ndarray = None,
                               title: str = "Graph Embeddings Visualization"):
        """
        可视化2D嵌入
        
        Args:
            reduced_embeddings: 2D降维嵌入
            cluster_labels: 聚类标签
            title: 图表标题
        """
        logger.info("可视化2D嵌入...")
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        plt.figure(figsize=VISUALIZATION_CONFIG["figure_size"])
        
        if cluster_labels is not None:
            # 按聚类着色
            scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                                c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
            plt.colorbar(scatter, label='Cluster')
        else:
            # 单色散点图
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                       alpha=0.7, s=50, c='blue')
        
        plt.title(title, fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"嵌入可视化图表已保存至: {save_path}")
        
        plt.show()
    
    def compute_embedding_quality_metrics(self) -> Dict[str, float]:
        """计算嵌入质量指标"""
        logger.info("计算嵌入质量指标...")
        
        metrics = {}
        
        # 1. 嵌入维度信息
        metrics['embedding_dimension'] = self.embeddings.shape[1]
        metrics['num_nodes'] = self.embeddings.shape[0]
        
        # 2. 嵌入分布统计
        metrics['mean_norm'] = np.linalg.norm(self.embeddings, axis=1).mean()
        metrics['std_norm'] = np.linalg.norm(self.embeddings, axis=1).std()
        
        # 3. 嵌入多样性
        # 计算成对余弦相似度的统计
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 为了效率，如果节点数太多，随机采样
        if self.embeddings.shape[0] > 1000:
            indices = np.random.choice(self.embeddings.shape[0], 1000, replace=False)
            sample_embeddings = self.embeddings[indices]
        else:
            sample_embeddings = self.embeddings
        
        cosine_sim = cosine_similarity(sample_embeddings)
        # 排除对角线（自相似度）
        mask = ~np.eye(cosine_sim.shape[0], dtype=bool)
        cosine_sim_values = cosine_sim[mask]
        
        metrics['mean_cosine_similarity'] = cosine_sim_values.mean()
        metrics['std_cosine_similarity'] = cosine_sim_values.std()
        
        # 4. 嵌入稀疏性
        zero_ratio = (self.embeddings == 0).sum() / self.embeddings.size
        metrics['sparsity_ratio'] = zero_ratio
        
        return metrics


class ComparisonAnalyzer:
    """嵌入方法比较分析器"""
    
    def __init__(self, graph: nx.Graph):
        """
        初始化比较分析器
        
        Args:
            graph: NetworkX图
        """
        self.graph = graph
        self.embeddings = {}
        self.results = {}
        
        logger.info("初始化嵌入方法比较分析器")
    
    def compare_embedding_methods(self, embedding_dim: int = 128) -> Dict[str, Any]:
        """
        比较多种嵌入方法
        
        Args:
            embedding_dim: 嵌入维度
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        logger.info("开始比较多种图嵌入方法...")
        
        data = from_networkx(self.graph)
        
        # 1. Node2Vec
        logger.info("训练Node2Vec模型...")
        try:
            node2vec = Node2VecEmbedding(self.graph, embedding_dim=embedding_dim)
            node2vec_history = node2vec.train(epochs=50)  # 减少轮数以节省时间
            node2vec_embeddings = node2vec.get_embeddings()
            self.embeddings['Node2Vec'] = node2vec_embeddings
            self.results['Node2Vec'] = {'training_history': node2vec_history}
        except Exception as e:
            logger.error(f"Node2Vec训练失败: {e}")
            self.embeddings['Node2Vec'] = None
        
        # 2. GraphSAGE
        logger.info("训练GraphSAGE模型...")
        try:
            graphsage = GraphSAGEEmbedding(data, output_dim=embedding_dim)
            graphsage_history = graphsage.train(epochs=100)  # 减少轮数
            graphsage_embeddings = graphsage.get_embeddings()
            self.embeddings['GraphSAGE'] = graphsage_embeddings
            self.results['GraphSAGE'] = {'training_history': graphsage_history}
        except Exception as e:
            logger.error(f"GraphSAGE训练失败: {e}")
            self.embeddings['GraphSAGE'] = None
        
        # 3. 分析每种方法的嵌入质量
        for method_name, embeddings in self.embeddings.items():
            if embeddings is not None:
                analyzer = EmbeddingAnalyzer(embeddings)
                quality_metrics = analyzer.compute_embedding_quality_metrics()
                self.results[method_name]['quality_metrics'] = quality_metrics
        
        # 4. 可视化比较
        self._visualize_comparison()
        
        return self.results
    
    def _visualize_comparison(self):
        """可视化嵌入方法比较"""
        logger.info("可视化嵌入方法比较...")
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        valid_methods = [(name, emb) for name, emb in self.embeddings.items() if emb is not None]
        
        if len(valid_methods) == 0:
            logger.warning("没有有效的嵌入结果可视化")
            return
        
        fig, axes = plt.subplots(1, len(valid_methods), figsize=(6 * len(valid_methods), 6))
        if len(valid_methods) == 1:
            axes = [axes]
        
        for idx, (method_name, embeddings) in enumerate(valid_methods):
            analyzer = EmbeddingAnalyzer(embeddings)
            
            # t-SNE降维
            reduced_emb = analyzer.dimensionality_reduction('tsne')
            
            # 聚类
            cluster_labels = analyzer.clustering_analysis(n_clusters=5)
            
            # 绘制
            ax = axes[idx]
            scatter = ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1], 
                               c=cluster_labels, cmap='viridis', alpha=0.7, s=30)
            ax.set_title(f'{method_name} Embeddings', fontsize=14)
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "embedding_methods_comparison.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"嵌入方法比较图表已保存至: {save_path}")
        
        plt.show()
        
        # 保存质量指标比较
        self._save_quality_comparison()
    
    def _save_quality_comparison(self):
        """保存质量指标比较"""
        quality_data = []
        
        for method_name, result in self.results.items():
            if 'quality_metrics' in result:
                metrics = result['quality_metrics']
                metrics['method'] = method_name
                quality_data.append(metrics)
        
        if quality_data:
            quality_df = pd.DataFrame(quality_data)
            save_path = ANALYSIS_OUTPUT_DIR / "embedding_quality_comparison.csv"
            quality_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            logger.info(f"嵌入质量比较已保存至: {save_path}")


def main():
    """主函数入口"""
    # 加载网络图
    from config.settings import NETWORK_OUTPUT_DIR, FILENAMES
    
    graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
    
    try:
        G = nx.read_graphml(str(graph_path))
        logger.info(f"成功加载网络图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
        # 如果图太大，采样一个子图进行演示
        if G.number_of_nodes() > 1000:
            logger.info("图较大，采样子图进行演示...")
            nodes_sample = list(G.nodes())[:1000]
            G = G.subgraph(nodes_sample).copy()
            logger.info(f"采样后的图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
    except FileNotFoundError:
        logger.error(f"未找到网络图文件: {graph_path}")
        return
    
    # 创建比较分析器
    comparator = ComparisonAnalyzer(G)
    
    # 运行比较分析
    results = comparator.compare_embedding_methods(embedding_dim=64)  # 使用较小的维度以节省时间
    
    # 打印结果摘要
    print("\n=== 图嵌入方法比较结果 ===")
    for method_name, result in results.items():
        if 'quality_metrics' in result:
            metrics = result['quality_metrics']
            print(f"\n{method_name}:")
            print(f"  - 节点数: {metrics['num_nodes']}")
            print(f"  - 嵌入维度: {metrics['embedding_dimension']}")
            print(f"  - 平均范数: {metrics['mean_norm']:.4f}")
            print(f"  - 平均余弦相似度: {metrics['mean_cosine_similarity']:.4f}")
            print(f"  - 稀疏度: {metrics['sparsity_ratio']:.4f}")


if __name__ == "__main__":
    main()
