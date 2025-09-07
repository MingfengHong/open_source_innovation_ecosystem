"""
异构注意力网络(HAN)模型实现
使用PyTorch Geometric实现开源生态系统的异构图神经网络分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear, to_hetero
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, MODELS_DIR, MODEL_CONFIG, VISUALIZATION_CONFIG

# 设置日志
logger = setup_logger(__name__)


class HeteroGraphBuilder:
    """异构图构建器"""
    
    def __init__(self, networkx_graph: nx.Graph):
        """
        初始化异构图构建器
        
        Args:
            networkx_graph: NetworkX异构图
        """
        self.nx_graph = networkx_graph
        self.hetero_data = None
        self.node_type_mapping = {}
        self.edge_type_mapping = {}
        
        logger.info(f"初始化异构图构建器: {self.nx_graph.number_of_nodes()} 个节点, {self.nx_graph.number_of_edges()} 条边")
    
    def build_hetero_data(self) -> HeteroData:
        """
        将NetworkX图转换为PyTorch Geometric异构数据格式
        
        Returns:
            HeteroData: PyTorch Geometric异构图数据
        """
        logger.info("构建异构图数据...")
        
        # 1. 分析节点类型和边类型
        self._analyze_graph_structure()
        
        # 2. 创建异构数据对象
        hetero_data = HeteroData()
        
        # 3. 添加节点数据
        self._add_node_data(hetero_data)
        
        # 4. 添加边数据
        self._add_edge_data(hetero_data)
        
        # 5. 添加节点特征
        self._add_node_features(hetero_data)
        
        self.hetero_data = hetero_data
        
        logger.info("异构图数据构建完成")
        self._print_hetero_data_info()
        
        return hetero_data
    
    def _analyze_graph_structure(self):
        """分析图结构，识别节点类型和边类型"""
        logger.info("分析图结构...")
        
        # 分析节点类型
        node_types = set()
        for node, data in self.nx_graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types.add(node_type)
        
        logger.info(f"发现节点类型: {node_types}")
        
        # 分析边类型
        edge_types = set()
        for u, v, data in self.nx_graph.edges(data=True):
            edge_type = data.get('type', 'unknown')
            u_type = self.nx_graph.nodes[u].get('type', 'unknown')
            v_type = self.nx_graph.nodes[v].get('type', 'unknown')
            
            # 创建边类型标识符
            edge_type_id = (u_type, edge_type, v_type)
            edge_types.add(edge_type_id)
        
        logger.info(f"发现边类型: {len(edge_types)} 种")
        
        # 存储类型信息
        self.node_types = sorted(list(node_types))
        self.edge_types = sorted(list(edge_types))
    
    def _add_node_data(self, hetero_data: HeteroData):
        """添加节点数据到异构图"""
        logger.info("添加节点数据...")
        
        # 为每种节点类型创建节点映射
        for node_type in self.node_types:
            nodes_of_type = [node for node, data in self.nx_graph.nodes(data=True) 
                           if data.get('type', 'unknown') == node_type]
            
            # 创建节点ID映射
            node_mapping = {node: idx for idx, node in enumerate(nodes_of_type)}
            self.node_type_mapping[node_type] = {
                'nodes': nodes_of_type,
                'mapping': node_mapping,
                'reverse_mapping': {idx: node for node, idx in node_mapping.items()}
            }
            
            # 设置节点数量
            hetero_data[node_type].num_nodes = len(nodes_of_type)
            
            logger.info(f"节点类型 '{node_type}': {len(nodes_of_type)} 个节点")
    
    def _add_edge_data(self, hetero_data: HeteroData):
        """添加边数据到异构图"""
        logger.info("添加边数据...")
        
        # 为每种边类型创建边索引
        for src_type, edge_type, dst_type in self.edge_types:
            edge_list = []
            
            for u, v, data in self.nx_graph.edges(data=True):
                u_type = self.nx_graph.nodes[u].get('type', 'unknown')
                v_type = self.nx_graph.nodes[v].get('type', 'unknown')
                e_type = data.get('type', 'unknown')
                
                if u_type == src_type and v_type == dst_type and e_type == edge_type:
                    # 获取节点在各自类型中的索引
                    u_idx = self.node_type_mapping[src_type]['mapping'][u]
                    v_idx = self.node_type_mapping[dst_type]['mapping'][v]
                    edge_list.append([u_idx, v_idx])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                hetero_data[src_type, edge_type, dst_type].edge_index = edge_index
                
                logger.info(f"边类型 '{src_type}' --{edge_type}--> '{dst_type}': {len(edge_list)} 条边")
    
    def _add_node_features(self, hetero_data: HeteroData):
        """添加节点特征"""
        logger.info("添加节点特征...")
        
        for node_type in self.node_types:
            nodes_info = self.node_type_mapping[node_type]
            feature_matrix = []
            
            for node in nodes_info['nodes']:
                features = self._extract_node_features(node, node_type)
                feature_matrix.append(features)
            
            if feature_matrix:
                feature_tensor = torch.tensor(feature_matrix, dtype=torch.float)
                hetero_data[node_type].x = feature_tensor
                
                logger.info(f"节点类型 '{node_type}': 特征维度 {feature_tensor.shape}")
    
    def _extract_node_features(self, node: Any, node_type: str) -> List[float]:
        """提取节点特征"""
        node_data = self.nx_graph.nodes[node]
        features = []
        
        if node_type == 'user':
            # 用户节点特征
            features.extend([
                1.0,  # 用户类型标识
                0.0,  # 非仓库
                0.0,  # 非PR
                0.0,  # 非Issue
                self.nx_graph.degree(node),  # 度
                len([n for n in self.nx_graph.neighbors(node) 
                    if self.nx_graph.nodes[n].get('type') == 'repo']),  # 连接的仓库数
            ])
            
        elif node_type == 'repo':
            # 仓库节点特征
            features.extend([
                0.0,  # 非用户
                1.0,  # 仓库类型标识
                0.0,  # 非PR
                0.0,  # 非Issue
                self.nx_graph.degree(node),  # 度
                node_data.get('stars', 0),  # 星标数
                node_data.get('forks', 0),  # 分叉数
            ])
            
        elif node_type == 'pr':
            # PR节点特征
            features.extend([
                0.0,  # 非用户
                0.0,  # 非仓库
                1.0,  # PR类型标识
                0.0,  # 非Issue
                self.nx_graph.degree(node),  # 度
                1.0 if node_data.get('timestamp') else 0.0,  # 是否有时间戳
            ])
            
        elif node_type == 'issue':
            # Issue节点特征
            features.extend([
                0.0,  # 非用户
                0.0,  # 非仓库
                0.0,  # 非PR
                1.0,  # Issue类型标识
                self.nx_graph.degree(node),  # 度
                1.0 if node_data.get('timestamp') else 0.0,  # 是否有时间戳
            ])
            
        else:
            # 未知类型节点
            features.extend([0.0] * 6)
        
        # 归一化特征
        features = [float(f) for f in features]
        
        return features
    
    def _print_hetero_data_info(self):
        """打印异构图数据信息"""
        logger.info("\n=== 异构图数据信息 ===")
        
        # 节点信息
        for node_type in self.hetero_data.node_types:
            num_nodes = self.hetero_data[node_type].num_nodes
            if hasattr(self.hetero_data[node_type], 'x'):
                feature_dim = self.hetero_data[node_type].x.shape[1]
                logger.info(f"节点类型 '{node_type}': {num_nodes} 个节点, {feature_dim} 维特征")
            else:
                logger.info(f"节点类型 '{node_type}': {num_nodes} 个节点, 无特征")
        
        # 边信息
        for edge_type in self.hetero_data.edge_types:
            num_edges = self.hetero_data[edge_type].edge_index.shape[1]
            logger.info(f"边类型 '{edge_type}': {num_edges} 条边")


class HANModel(nn.Module):
    """异构注意力网络模型"""
    
    def __init__(self, 
                 metadata: Tuple[List[str], List[Tuple[str, str, str]]],
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 num_classes: int = None):
        """
        初始化HAN模型
        
        Args:
            metadata: 异构图元数据(节点类型列表, 边类型列表)
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 层数
            dropout: Dropout率
            num_classes: 分类类别数(用于节点分类任务)
        """
        super(HANModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes
        
        # 构建HAN层
        self.han_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim if i > 0 else -1  # 第一层自动推断输入维度
            self.han_layers.append(
                HANConv(
                    in_channels=in_dim,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    metadata=metadata
                )
            )
        
        # 如果指定了分类任务，添加分类器
        if num_classes is not None:
            self.classifier = nn.ModuleDict()
            for node_type in metadata[0]:
                self.classifier[node_type] = Linear(hidden_dim, num_classes)
        
        # 注意力权重存储
        self.attention_weights = {}
        
        logger.info(f"初始化HAN模型: {num_layers}层, {hidden_dim}维, {num_heads}头")
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                return_attention_weights: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_dict: 节点特征字典
            edge_index_dict: 边索引字典
            return_attention_weights: 是否返回注意力权重
            
        Returns:
            Dict[str, torch.Tensor]: 节点嵌入或分类结果
        """
        # 通过HAN层
        for i, han_layer in enumerate(self.han_layers):
            x_dict = han_layer(x_dict, edge_index_dict, return_attention_weights=return_attention_weights)
            
            # 如果返回注意力权重，提取并存储
            if return_attention_weights and isinstance(x_dict, tuple):
                x_dict, attention_weights = x_dict
                self.attention_weights[f'layer_{i}'] = attention_weights
            
            # 应用激活函数和dropout（除了最后一层）
            if i < len(self.han_layers) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) 
                         for key, x in x_dict.items()}
        
        # 如果有分类任务，应用分类器
        if self.num_classes is not None and hasattr(self, 'classifier'):
            classification_results = {}
            for node_type, embeddings in x_dict.items():
                if node_type in self.classifier:
                    classification_results[node_type] = self.classifier[node_type](embeddings)
            return classification_results
        
        return x_dict
    
    def get_attention_weights(self) -> Dict[str, Any]:
        """获取注意力权重"""
        return self.attention_weights


class HANTrainer:
    """HAN模型训练器"""
    
    def __init__(self, 
                 model: HANModel,
                 hetero_data: HeteroData,
                 task_type: str = 'node_classification',
                 target_node_type: str = 'repo',
                 learning_rate: float = 0.001,
                 device: str = None):
        """
        初始化训练器
        
        Args:
            model: HAN模型
            hetero_data: 异构图数据
            task_type: 任务类型 ('node_classification', 'link_prediction', 'embedding')
            target_node_type: 目标节点类型
            learning_rate: 学习率
            device: 设备
        """
        self.model = model
        self.hetero_data = hetero_data
        self.task_type = task_type
        self.target_node_type = target_node_type
        self.learning_rate = learning_rate
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 移动模型和数据到设备
        self.model = self.model.to(self.device)
        self.hetero_data = self.hetero_data.to(self.device)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练历史
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        logger.info(f"初始化HAN训练器: 任务={task_type}, 目标节点={target_node_type}, 设备={self.device}")
    
    def prepare_node_classification_task(self, 
                                       classification_attribute: str = 'primary_role') -> Dict[str, torch.Tensor]:
        """
        准备节点分类任务
        
        Args:
            classification_attribute: 分类属性名
            
        Returns:
            Dict[str, torch.Tensor]: 训练数据
        """
        logger.info(f"准备节点分类任务: 属性={classification_attribute}")
        
        # 这里需要根据实际的节点属性创建标签
        # 由于我们使用的是从NetworkX转换的数据，需要从原始图中提取标签信息
        
        # 创建虚拟标签作为示例
        target_nodes = self.hetero_data[self.target_node_type]
        num_nodes = target_nodes.num_nodes
        
        # 创建随机标签作为示例（实际应用中应该从真实数据中提取）
        # 这里我们创建3个类别：应用仓库、工具仓库、库仓库
        labels = torch.randint(0, 3, (num_nodes,), device=self.device)
        
        # 划分训练、验证、测试集
        num_train = int(0.6 * num_nodes)
        num_val = int(0.2 * num_nodes)
        
        indices = torch.randperm(num_nodes, device=self.device)
        train_idx = indices[:num_train]
        val_idx = indices[num_train:num_train + num_val]
        test_idx = indices[num_train + num_val:]
        
        # 创建掩码
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        # 将标签和掩码添加到数据中
        self.hetero_data[self.target_node_type].y = labels
        self.hetero_data[self.target_node_type].train_mask = train_mask
        self.hetero_data[self.target_node_type].val_mask = val_mask
        self.hetero_data[self.target_node_type].test_mask = test_mask
        
        logger.info(f"数据划分: 训练={num_train}, 验证={num_val}, 测试={len(test_idx)}")
        
        return {
            'labels': labels,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        
        # 前向传播
        out = self.model(self.hetero_data.x_dict, self.hetero_data.edge_index_dict)
        
        if self.task_type == 'node_classification':
            # 获取目标节点的输出和标签
            target_out = out[self.target_node_type]
            target_labels = self.hetero_data[self.target_node_type].y
            train_mask = self.hetero_data[self.target_node_type].train_mask
            
            # 计算损失
            loss = F.cross_entropy(target_out[train_mask], target_labels[train_mask])
            
            # 计算准确率
            pred = target_out[train_mask].argmax(dim=1)
            train_acc = (pred == target_labels[train_mask]).float().mean()
            
        else:
            # 其他任务类型的损失计算
            loss = torch.tensor(0.0, device=self.device)
            train_acc = torch.tensor(0.0, device=self.device)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), train_acc.item()
    
    def validate(self) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(self.hetero_data.x_dict, self.hetero_data.edge_index_dict)
            
            if self.task_type == 'node_classification':
                target_out = out[self.target_node_type]
                target_labels = self.hetero_data[self.target_node_type].y
                val_mask = self.hetero_data[self.target_node_type].val_mask
                
                val_loss = F.cross_entropy(target_out[val_mask], target_labels[val_mask])
                pred = target_out[val_mask].argmax(dim=1)
                val_acc = (pred == target_labels[val_mask]).float().mean()
                
            else:
                val_loss = torch.tensor(0.0, device=self.device)
                val_acc = torch.tensor(0.0, device=self.device)
        
        return val_loss.item(), val_acc.item()
    
    def train(self, epochs: int = 100, patience: int = 10, verbose: bool = True) -> Dict[str, List[float]]:
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            patience: 早停耐心值
            verbose: 是否打印训练过程
            
        Returns:
            Dict[str, List[float]]: 训练历史
        """
        logger.info(f"开始训练HAN模型: {epochs} 轮")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="训练进度"):
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 记录历史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model(MODELS_DIR / "han_best_model.pth")
            else:
                patience_counter += 1
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("训练完成")
        return self.training_history
    
    def test(self) -> Dict[str, float]:
        """测试模型"""
        logger.info("测试模型...")
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.hetero_data.x_dict, self.hetero_data.edge_index_dict)
            
            if self.task_type == 'node_classification':
                target_out = out[self.target_node_type]
                target_labels = self.hetero_data[self.target_node_type].y
                test_mask = self.hetero_data[self.target_node_type].test_mask
                
                test_loss = F.cross_entropy(target_out[test_mask], target_labels[test_mask])
                pred = target_out[test_mask].argmax(dim=1)
                test_acc = (pred == target_labels[test_mask]).float().mean()
                
                results = {
                    'test_loss': test_loss.item(),
                    'test_accuracy': test_acc.item()
                }
            else:
                results = {'test_loss': 0.0, 'test_accuracy': 0.0}
        
        logger.info(f"测试结果: {results}")
        return results
    
    def extract_attention_weights(self) -> Dict[str, Any]:
        """提取注意力权重"""
        logger.info("提取注意力权重...")
        
        self.model.eval()
        with torch.no_grad():
            # 前向传播并获取注意力权重
            _ = self.model(self.hetero_data.x_dict, self.hetero_data.edge_index_dict, 
                          return_attention_weights=True)
            
            attention_weights = self.model.get_attention_weights()
        
        return attention_weights
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'model_config': {
                'hidden_dim': self.model.hidden_dim,
                'num_heads': self.model.num_heads,
                'num_layers': self.model.num_layers,
                'dropout': self.model.dropout,
                'num_classes': self.model.num_classes
            }
        }, path)
        
        logger.info(f"模型已保存至: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
        
        logger.info(f"模型已从 {path} 加载")


class HANAnalyzer:
    """HAN分析器"""
    
    def __init__(self, networkx_graph: nx.Graph):
        """
        初始化HAN分析器
        
        Args:
            networkx_graph: NetworkX异构图
        """
        self.nx_graph = networkx_graph
        self.hetero_builder = None
        self.hetero_data = None
        self.model = None
        self.trainer = None
        
        logger.info("初始化HAN分析器")
    
    def prepare_data(self) -> HeteroData:
        """准备异构图数据"""
        logger.info("准备异构图数据...")
        
        # 构建异构图
        self.hetero_builder = HeteroGraphBuilder(self.nx_graph)
        self.hetero_data = self.hetero_builder.build_hetero_data()
        
        return self.hetero_data
    
    def create_model(self, 
                    hidden_dim: int = None,
                    num_heads: int = None,
                    num_layers: int = None,
                    dropout: float = None,
                    num_classes: int = 3) -> HANModel:
        """创建HAN模型"""
        logger.info("创建HAN模型...")
        
        # 使用配置文件中的默认值
        config = MODEL_CONFIG["han_model"]
        hidden_dim = hidden_dim or config["hidden_dim"]
        num_heads = num_heads or config["num_heads"]
        num_layers = num_layers or config["num_layers"]
        dropout = dropout or config["dropout"]
        
        # 获取元数据
        metadata = (
            self.hetero_data.node_types,
            self.hetero_data.edge_types
        )
        
        # 创建模型
        self.model = HANModel(
            metadata=metadata,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes
        )
        
        return self.model
    
    def train_model(self, 
                   task_type: str = 'node_classification',
                   target_node_type: str = 'repo',
                   epochs: int = None,
                   learning_rate: float = None) -> Dict[str, List[float]]:
        """训练模型"""
        logger.info("训练HAN模型...")
        
        # 使用配置文件中的默认值
        config = MODEL_CONFIG["han_model"]
        epochs = epochs or config["epochs"]
        learning_rate = learning_rate or config["learning_rate"]
        
        # 创建训练器
        self.trainer = HANTrainer(
            model=self.model,
            hetero_data=self.hetero_data,
            task_type=task_type,
            target_node_type=target_node_type,
            learning_rate=learning_rate
        )
        
        # 准备任务
        if task_type == 'node_classification':
            self.trainer.prepare_node_classification_task()
        
        # 训练
        training_history = self.trainer.train(epochs=epochs)
        
        return training_history
    
    def analyze_attention_weights(self) -> Dict[str, Any]:
        """分析注意力权重"""
        logger.info("分析注意力权重...")
        
        if self.trainer is None:
            logger.error("模型尚未训练，无法分析注意力权重")
            return {}
        
        # 提取注意力权重
        attention_weights = self.trainer.extract_attention_weights()
        
        # 分析注意力权重
        analysis_results = {}
        
        for layer_name, weights in attention_weights.items():
            if isinstance(weights, dict):
                layer_analysis = {}
                for edge_type, edge_weights in weights.items():
                    if isinstance(edge_weights, torch.Tensor):
                        # 计算注意力权重统计
                        layer_analysis[edge_type] = {
                            'mean_attention': edge_weights.mean().item(),
                            'std_attention': edge_weights.std().item(),
                            'max_attention': edge_weights.max().item(),
                            'min_attention': edge_weights.min().item()
                        }
                
                analysis_results[layer_name] = layer_analysis
        
        return analysis_results
    
    def visualize_training_history(self, training_history: Dict[str, List[float]]):
        """可视化训练历史"""
        logger.info("可视化训练历史...")
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 损失曲线
        epochs = range(1, len(training_history['train_loss']) + 1)
        ax1.plot(epochs, training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(epochs, training_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, training_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy', fontsize=VISUALIZATION_CONFIG["title_font_size"])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "han_training_history.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"训练历史图表已保存至: {save_path}")
        
        plt.show()
    
    def visualize_attention_weights(self, attention_analysis: Dict[str, Any]):
        """可视化注意力权重"""
        logger.info("可视化注意力权重...")
        
        if not attention_analysis:
            logger.warning("没有注意力权重数据可视化")
            return
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        # 准备数据
        edge_types = []
        attention_means = []
        
        for layer_name, layer_data in attention_analysis.items():
            for edge_type, stats in layer_data.items():
                edge_types.append(f"{layer_name}_{edge_type}")
                attention_means.append(stats['mean_attention'])
        
        if not edge_types:
            logger.warning("没有有效的注意力权重数据")
            return
        
        # 创建条形图
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(edge_types)), attention_means)
        
        # 美化图表
        plt.title('Attention Weights by Edge Type and Layer', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.xlabel('Edge Type (Layer_EdgeType)')
        plt.ylabel('Mean Attention Weight')
        plt.xticks(range(len(edge_types)), edge_types, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, attention_means)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        save_path = ANALYSIS_OUTPUT_DIR / "han_attention_weights.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"注意力权重图表已保存至: {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """运行完整的HAN分析"""
        logger.info("开始完整的HAN分析...")
        
        try:
            # 1. 准备数据
            hetero_data = self.prepare_data()
            
            # 2. 创建模型
            model = self.create_model()
            
            # 3. 训练模型
            training_history = self.train_model()
            
            # 4. 测试模型
            test_results = self.trainer.test()
            
            # 5. 分析注意力权重
            attention_analysis = self.analyze_attention_weights()
            
            # 6. 可视化结果
            self.visualize_training_history(training_history)
            self.visualize_attention_weights(attention_analysis)
            
            # 7. 保存分析结果
            analysis_results = {
                'hetero_data_info': {
                    'node_types': len(hetero_data.node_types),
                    'edge_types': len(hetero_data.edge_types),
                    'total_nodes': sum(hetero_data[nt].num_nodes for nt in hetero_data.node_types),
                    'total_edges': sum(hetero_data[et].edge_index.shape[1] for et in hetero_data.edge_types)
                },
                'model_info': {
                    'hidden_dim': model.hidden_dim,
                    'num_heads': model.num_heads,
                    'num_layers': model.num_layers,
                    'num_classes': model.num_classes
                },
                'training_results': {
                    'final_train_loss': training_history['train_loss'][-1],
                    'final_val_loss': training_history['val_loss'][-1],
                    'final_train_acc': training_history['train_acc'][-1],
                    'final_val_acc': training_history['val_acc'][-1],
                    'epochs_trained': len(training_history['train_loss'])
                },
                'test_results': test_results,
                'attention_analysis': attention_analysis
            }
            
            # 保存结果
            import json
            results_path = ANALYSIS_OUTPUT_DIR / "han_analysis_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"HAN分析结果已保存至: {results_path}")
            logger.info("HAN完整分析完成！")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"HAN分析失败: {e}")
            return {}


def main():
    """主函数入口"""
    # 加载网络图
    from config.settings import NETWORK_OUTPUT_DIR, FILENAMES
    import networkx as nx
    
    graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
    
    try:
        G = nx.read_graphml(str(graph_path))
        logger.info(f"成功加载网络图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    except FileNotFoundError:
        logger.error(f"未找到网络图文件: {graph_path}")
        return
    
    # 创建HAN分析器
    analyzer = HANAnalyzer(G)
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n=== HAN分析完成 ===")
        print(f"数据: {results['hetero_data_info']['total_nodes']} 个节点, {results['hetero_data_info']['total_edges']} 条边")
        print(f"模型: {results['model_info']['num_layers']} 层, {results['model_info']['num_heads']} 头")
        print(f"测试准确率: {results['test_results']['test_accuracy']:.4f}")
        print(f"注意力权重分析: {len(results['attention_analysis'])} 层")
    else:
        print("❌ HAN分析失败")


if __name__ == "__main__":
    main()
