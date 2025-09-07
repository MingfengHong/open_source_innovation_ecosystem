#!/usr/bin/env python
"""
异构注意力网络(HAN)演示脚本
展示如何使用HAN模型进行开源生态系统分析
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import networkx as nx
import torch
from datetime import datetime

# 导入HAN模块
from src.advanced_models.han_model import HANAnalyzer, HeteroGraphBuilder, HANModel, HANTrainer
from src.advanced_models.graph_embedding import ComparisonAnalyzer
from src.utils.logging_config import setup_logger
from config.settings import NETWORK_OUTPUT_DIR, FILENAMES, ANALYSIS_OUTPUT_DIR

# 设置日志
logger = setup_logger(__name__)


def create_sample_heterogeneous_graph() -> nx.Graph:
    """创建示例异构图"""
    print("🔧 创建示例异构图...")
    
    G = nx.Graph()
    
    # 添加不同类型的节点
    # 用户节点
    users = [f"user_{i}" for i in range(50)]
    for user in users:
        G.add_node(user, type='user', user_id=user)
    
    # 仓库节点
    repos = [f"repo_{i}" for i in range(20)]
    for repo in repos:
        G.add_node(repo, type='repo', 
                  stars=np.random.randint(0, 1000),
                  forks=np.random.randint(0, 100),
                  primary_role=np.random.choice(['application', 'library', 'tool']))
    
    # PR节点
    prs = [f"pr_{i}" for i in range(30)]
    for pr in prs:
        G.add_node(pr, type='pr', timestamp='2023-01-01')
    
    # Issue节点
    issues = [f"issue_{i}" for i in range(40)]
    for issue in issues:
        G.add_node(issue, type='issue', timestamp='2023-01-01')
    
    # 添加边
    # 用户-仓库（Star关系）
    for user in users[:30]:  # 前30个用户
        for repo in np.random.choice(repos, size=np.random.randint(1, 5), replace=False):
            G.add_edge(user, repo, type='star')
    
    # 用户-PR（创建关系）
    for pr in prs:
        user = np.random.choice(users)
        repo = np.random.choice(repos)
        G.add_edge(user, pr, type='create')
        G.add_edge(pr, repo, type='belongs_to')
    
    # 用户-Issue（创建关系）
    for issue in issues:
        user = np.random.choice(users)
        repo = np.random.choice(repos)
        G.add_edge(user, issue, type='create')
        G.add_edge(issue, repo, type='belongs_to')
    
    print(f"✅ 示例图创建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    
    # 打印节点类型统计
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("📊 节点类型分布:")
    for node_type, count in node_types.items():
        print(f"   - {node_type}: {count} 个")
    
    return G


def demonstrate_hetero_graph_building(graph: nx.Graph):
    """演示异构图构建"""
    print("\n" + "=" * 60)
    print("1. 异构图构建演示")
    print("=" * 60)
    
    # 创建异构图构建器
    builder = HeteroGraphBuilder(graph)
    
    # 构建异构数据
    hetero_data = builder.build_hetero_data()
    
    print(f"✅ 异构图构建完成!")
    print(f"   - 节点类型数: {len(hetero_data.node_types)}")
    print(f"   - 边类型数: {len(hetero_data.edge_types)}")
    
    # 详细信息
    print(f"\n📋 详细信息:")
    for node_type in hetero_data.node_types:
        num_nodes = hetero_data[node_type].num_nodes
        feature_dim = hetero_data[node_type].x.shape[1] if hasattr(hetero_data[node_type], 'x') else 0
        print(f"   - {node_type}: {num_nodes} 个节点, {feature_dim} 维特征")
    
    for edge_type in hetero_data.edge_types:
        num_edges = hetero_data[edge_type].edge_index.shape[1]
        print(f"   - {edge_type}: {num_edges} 条边")
    
    return hetero_data


def demonstrate_han_model_creation(hetero_data):
    """演示HAN模型创建"""
    print("\n" + "=" * 60)
    print("2. HAN模型创建演示")
    print("=" * 60)
    
    # 获取元数据
    metadata = (hetero_data.node_types, hetero_data.edge_types)
    
    # 创建HAN模型
    model = HANModel(
        metadata=metadata,
        hidden_dim=64,  # 使用较小的维度以节省计算资源
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        num_classes=3  # 仓库分类任务
    )
    
    print(f"✅ HAN模型创建完成!")
    print(f"   - 隐藏维度: {model.hidden_dim}")
    print(f"   - 注意力头数: {model.num_heads}")
    print(f"   - 层数: {model.num_layers}")
    print(f"   - 分类类别数: {model.num_classes}")
    
    # 打印模型结构
    print(f"\n🏗️ 模型结构:")
    print(model)
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 模型参数:")
    print(f"   - 总参数数: {total_params:,}")
    print(f"   - 可训练参数数: {trainable_params:,}")
    
    return model


def demonstrate_han_training(model, hetero_data):
    """演示HAN模型训练"""
    print("\n" + "=" * 60)
    print("3. HAN模型训练演示")
    print("=" * 60)
    
    # 创建训练器
    trainer = HANTrainer(
        model=model,
        hetero_data=hetero_data,
        task_type='node_classification',
        target_node_type='repo',  # 对仓库节点进行分类
        learning_rate=0.01
    )
    
    print(f"🎯 训练设置:")
    print(f"   - 任务类型: {trainer.task_type}")
    print(f"   - 目标节点类型: {trainer.target_node_type}")
    print(f"   - 学习率: {trainer.learning_rate}")
    print(f"   - 设备: {trainer.device}")
    
    # 准备节点分类任务
    print(f"\n📋 准备节点分类任务...")
    task_data = trainer.prepare_node_classification_task()
    
    train_count = task_data['train_mask'].sum().item()
    val_count = task_data['val_mask'].sum().item()
    test_count = task_data['test_mask'].sum().item()
    
    print(f"   - 训练集: {train_count} 个节点")
    print(f"   - 验证集: {val_count} 个节点")
    print(f"   - 测试集: {test_count} 个节点")
    
    # 开始训练
    print(f"\n🚀 开始训练...")
    training_history = trainer.train(epochs=50, patience=10, verbose=True)
    
    print(f"✅ 训练完成!")
    print(f"   - 训练轮数: {len(training_history['train_loss'])}")
    print(f"   - 最终训练损失: {training_history['train_loss'][-1]:.4f}")
    print(f"   - 最终验证损失: {training_history['val_loss'][-1]:.4f}")
    print(f"   - 最终训练准确率: {training_history['train_acc'][-1]:.4f}")
    print(f"   - 最终验证准确率: {training_history['val_acc'][-1]:.4f}")
    
    return trainer, training_history


def demonstrate_model_evaluation(trainer):
    """演示模型评估"""
    print("\n" + "=" * 60)
    print("4. 模型评估演示")
    print("=" * 60)
    
    # 测试模型
    print("🧪 测试模型性能...")
    test_results = trainer.test()
    
    print(f"✅ 测试完成!")
    print(f"   - 测试损失: {test_results['test_loss']:.4f}")
    print(f"   - 测试准确率: {test_results['test_accuracy']:.4f}")
    
    # 提取注意力权重
    print(f"\n🔍 提取注意力权重...")
    attention_weights = trainer.extract_attention_weights()
    
    print(f"   - 捕获的注意力层数: {len(attention_weights)}")
    
    for layer_name, weights in attention_weights.items():
        if isinstance(weights, dict):
            print(f"   - {layer_name}: {len(weights)} 个边类型")
            for edge_type, edge_attention in weights.items():
                if hasattr(edge_attention, 'shape'):
                    print(f"     * {edge_type}: {edge_attention.shape}")
    
    return test_results, attention_weights


def demonstrate_attention_analysis(attention_weights):
    """演示注意力权重分析"""
    print("\n" + "=" * 60)
    print("5. 注意力权重分析演示")
    print("=" * 60)
    
    if not attention_weights:
        print("⚠️  没有注意力权重数据可分析")
        return {}
    
    analysis_results = {}
    
    print("📊 分析注意力权重统计...")
    
    for layer_name, layer_weights in attention_weights.items():
        print(f"\n🔍 {layer_name}:")
        layer_analysis = {}
        
        if isinstance(layer_weights, dict):
            for edge_type, edge_weights in layer_weights.items():
                if isinstance(edge_weights, torch.Tensor):
                    # 移到CPU并转换为numpy
                    weights_np = edge_weights.detach().cpu().numpy()
                    
                    stats = {
                        'mean': float(np.mean(weights_np)),
                        'std': float(np.std(weights_np)),
                        'min': float(np.min(weights_np)),
                        'max': float(np.max(weights_np)),
                        'shape': weights_np.shape
                    }
                    
                    layer_analysis[str(edge_type)] = stats
                    
                    print(f"   - {edge_type}:")
                    print(f"     * 形状: {stats['shape']}")
                    print(f"     * 均值: {stats['mean']:.4f}")
                    print(f"     * 标准差: {stats['std']:.4f}")
                    print(f"     * 范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
        
        analysis_results[layer_name] = layer_analysis
    
    print(f"\n✅ 注意力权重分析完成!")
    return analysis_results


def demonstrate_graph_embedding_comparison(graph):
    """演示图嵌入方法比较"""
    print("\n" + "=" * 60)
    print("6. 图嵌入方法比较演示")
    print("=" * 60)
    
    # 创建比较分析器
    comparator = ComparisonAnalyzer(graph)
    
    print("🔄 开始比较不同的图嵌入方法...")
    print("   (使用较小的参数以节省时间)")
    
    # 运行比较分析
    comparison_results = comparator.compare_embedding_methods(embedding_dim=32)
    
    print(f"\n✅ 图嵌入比较完成!")
    
    # 打印比较结果
    for method_name, results in comparison_results.items():
        if 'quality_metrics' in results:
            metrics = results['quality_metrics']
            print(f"\n📊 {method_name}:")
            print(f"   - 嵌入维度: {metrics['embedding_dimension']}")
            print(f"   - 节点数: {metrics['num_nodes']}")
            print(f"   - 平均范数: {metrics['mean_norm']:.4f}")
            print(f"   - 平均余弦相似度: {metrics['mean_cosine_similarity']:.4f}")
            print(f"   - 稀疏度: {metrics['sparsity_ratio']:.4f}")
    
    return comparison_results


def demonstrate_complete_han_pipeline(use_sample_graph: bool = True):
    """演示完整的HAN分析流水线"""
    print("🚀 异构注意力网络(HAN)完整演示")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # 1. 准备图数据
        if use_sample_graph:
            graph = create_sample_heterogeneous_graph()
        else:
            # 尝试加载真实图数据
            graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
            if graph_path.exists():
                graph = nx.read_graphml(str(graph_path))
                print(f"✅ 加载真实图数据: {graph.number_of_nodes()} 个节点")
                
                # 如果图太大，采样子图
                if graph.number_of_nodes() > 200:
                    print("📏 图较大，采样子图进行演示...")
                    nodes_sample = list(graph.nodes())[:200]
                    graph = graph.subgraph(nodes_sample).copy()
                    print(f"   采样后: {graph.number_of_nodes()} 个节点")
            else:
                print("⚠️  未找到真实图数据，使用示例图")
                graph = create_sample_heterogeneous_graph()
        
        # 2. 异构图构建
        hetero_data = demonstrate_hetero_graph_building(graph)
        
        # 3. HAN模型创建
        model = demonstrate_han_model_creation(hetero_data)
        
        # 4. 模型训练
        trainer, training_history = demonstrate_han_training(model, hetero_data)
        
        # 5. 模型评估
        test_results, attention_weights = demonstrate_model_evaluation(trainer)
        
        # 6. 注意力权重分析
        attention_analysis = demonstrate_attention_analysis(attention_weights)
        
        # 7. 图嵌入方法比较
        embedding_comparison = demonstrate_graph_embedding_comparison(graph)
        
        # 8. 保存结果
        print("\n" + "=" * 60)
        print("7. 保存分析结果")
        print("=" * 60)
        
        # 保存综合结果
        results_summary = {
            'graph_info': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges(),
                'node_types': len(hetero_data.node_types),
                'edge_types': len(hetero_data.edge_types)
            },
            'model_info': {
                'hidden_dim': model.hidden_dim,
                'num_heads': model.num_heads,
                'num_layers': model.num_layers,
                'total_parameters': sum(p.numel() for p in model.parameters())
            },
            'training_results': {
                'epochs_trained': len(training_history['train_loss']),
                'final_train_loss': training_history['train_loss'][-1],
                'final_val_loss': training_history['val_loss'][-1],
                'final_train_acc': training_history['train_acc'][-1],
                'final_val_acc': training_history['val_acc'][-1]
            },
            'test_results': test_results,
            'attention_analysis': attention_analysis,
            'embedding_comparison': {
                method: {'quality_metrics': results.get('quality_metrics', {})}
                for method, results in embedding_comparison.items()
            }
        }
        
        # 保存到文件
        import json
        results_path = ANALYSIS_OUTPUT_DIR / "han_demo_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📁 分析结果已保存至: {results_path}")
        
        # 9. 总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("✅ HAN演示完成!")
        print(f"   - 总耗时: {duration}")
        print(f"   - 图规模: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
        print(f"   - 异构图: {len(hetero_data.node_types)} 种节点类型, {len(hetero_data.edge_types)} 种边类型")
        print(f"   - 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - 最终测试准确率: {test_results['test_accuracy']:.4f}")
        print(f"   - 结果文件: {results_path}")
        print("=" * 80)
        
        return results_summary
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        logger.error(f"HAN演示失败: {e}")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='HAN模型演示脚本')
    parser.add_argument('--use-sample', action='store_true', 
                       help='使用示例图数据（默认尝试加载真实数据）')
    parser.add_argument('--sample-only', action='store_true',
                       help='仅使用示例图数据')
    
    args = parser.parse_args()
    
    # 运行演示
    results = demonstrate_complete_han_pipeline(use_sample_graph=args.sample_only)
    
    if results:
        print("\n🎉 演示成功完成！")
        print("📊 主要结果:")
        print(f"   - 测试准确率: {results['test_results']['test_accuracy']:.4f}")
        print(f"   - 训练轮数: {results['training_results']['epochs_trained']}")
        print(f"   - 模型参数量: {results['model_info']['total_parameters']:,}")
    else:
        print("\n❌ 演示失败")


if __name__ == "__main__":
    main()
