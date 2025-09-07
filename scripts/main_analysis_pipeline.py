#!/usr/bin/env python
"""
主分析流水线脚本
展示如何使用重构后的开源生态系统分析框架
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import logging
from datetime import datetime

# 导入配置
from config.settings import *
from config.api_config import api_config

# 导入分析模块
from src.network_analysis.dynamic_analysis import DynamicNetworkAnalyzer
from src.network_analysis.community_detection import CommunityDetector, DynamicCommunityAnalyzer
from src.user_analysis.role_clustering import MultiAlgorithmClustering
from src.user_analysis.role_evolution import RoleEvolutionAnalyzer
from src.causal_analysis.granger_causality import GrangerCausalityAnalyzer
from src.utils.logging_config import setup_logger

# 设置日志
logger = setup_logger(__name__, log_file=ANALYSIS_OUTPUT_DIR / "main_pipeline.log")


class AnalysisPipeline:
    """主分析流水线"""
    
    def __init__(self):
        """初始化分析流水线"""
        self.results = {}
        logger.info("初始化分析流水线")
        
        # 确保所有输出目录存在
        ensure_directories()
    
    def run_network_analysis(self):
        """运行网络分析"""
        logger.info("=" * 60)
        logger.info("开始网络分析...")
        logger.info("=" * 60)
        
        try:
            # 1. 动态网络分析（增强版，包含多种中心性）
            logger.info("1. 运行动态网络分析...")
            dynamic_analyzer = DynamicNetworkAnalyzer()
            dynamic_results = dynamic_analyzer.run_complete_analysis(include_betweenness=True)
            self.results['dynamic_network'] = dynamic_results
            logger.info("✓ 动态网络分析完成")
            
            # 2. 社区检测算法对比
            logger.info("2. 运行社区检测算法对比...")
            
            # 加载网络图
            import networkx as nx
            graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
            
            if graph_path.exists():
                G = nx.read_graphml(str(graph_path))
                
                # 静态社区检测对比
                detector = CommunityDetector(G)
                comparison_df = detector.compare_algorithms()
                self.results['community_comparison'] = comparison_df
                
                # 动态社区分析
                dynamic_community_analyzer = DynamicCommunityAnalyzer(
                    G, 
                    ANALYSIS_CONFIG["start_date"], 
                    ANALYSIS_CONFIG["end_date"],
                    algorithms=['louvain', 'leiden']  # 根据可用算法调整
                )
                dynamic_community_results = dynamic_community_analyzer.analyze_temporal_communities()
                self.results['dynamic_community'] = dynamic_community_results
                
                logger.info("✓ 社区检测分析完成")
            else:
                logger.warning(f"网络图文件不存在: {graph_path}")
                
        except Exception as e:
            logger.error(f"网络分析失败: {e}")
    
    def run_user_analysis(self):
        """运行用户分析"""
        logger.info("=" * 60)
        logger.info("开始用户分析...")
        logger.info("=" * 60)
        
        try:
            # 1. 多算法角色聚类
            logger.info("1. 运行多算法角色聚类...")
            
            features_path = ANALYSIS_OUTPUT_DIR / "user_behavior_features.csv"
            if features_path.exists():
                features_df = pd.read_csv(features_path)
                
                # 多算法聚类对比
                clusterer = MultiAlgorithmClustering(features_df)
                optimal_k = clusterer.determine_optimal_k()
                recommended_k = optimal_k.get('silhouette', 6)
                
                comparison_df = clusterer.compare_all_algorithms(recommended_k)
                self.results['clustering_comparison'] = comparison_df
                
                # 选择最佳算法
                best_algorithm = comparison_df.loc[comparison_df['silhouette_score'].idxmax(), 'algorithm'].lower()
                profile_df = clusterer.analyze_cluster_profiles(best_algorithm)
                result_df = clusterer.export_clustering_results(best_algorithm)
                
                self.results['user_clustering'] = {
                    'recommended_k': recommended_k,
                    'best_algorithm': best_algorithm,
                    'profile_df': profile_df,
                    'result_df': result_df
                }
                
                logger.info(f"✓ 角色聚类完成，最佳算法: {best_algorithm}, K={recommended_k}")
            else:
                logger.warning(f"用户特征文件不存在: {features_path}")
            
            # 2. 角色演化分析
            logger.info("2. 运行角色演化分析...")
            
            evolution_analyzer = RoleEvolutionAnalyzer(
                clustering_algorithm='kmeans',
                n_clusters=6
            )
            evolution_results = evolution_analyzer.run_complete_analysis()
            self.results['role_evolution'] = evolution_results
            
            logger.info("✓ 角色演化分析完成")
            
        except Exception as e:
            logger.error(f"用户分析失败: {e}")
    
    def run_causal_analysis(self):
        """运行因果分析"""
        logger.info("=" * 60)
        logger.info("开始因果分析...")
        logger.info("=" * 60)
        
        try:
            # 1. 构建面板数据
            logger.info("1. 构建面板数据...")
            from src.causal_analysis.panel_data_builder import PanelDataBuilder
            
            builder = PanelDataBuilder(frequency='M')
            ecosystem_panel = builder.build_ecosystem_panel()
            
            if not ecosystem_panel.empty:
                builder.save_panel_data(ecosystem_panel, "ecosystem_panel_data.csv", "生态系统层面")
                self.results['panel_data_construction'] = {
                    'n_entities': len(ecosystem_panel.index.get_level_values(0).unique()),
                    'n_time_periods': len(ecosystem_panel.index.get_level_values(1).unique()),
                    'n_observations': len(ecosystem_panel)
                }
                logger.info("✓ 面板数据构建完成")
            
            # 2. 格兰杰因果检验
            logger.info("2. 运行格兰杰因果检验...")
            
            panel_data_path = ANALYSIS_OUTPUT_DIR / "monthly_panel_data.csv"
            if panel_data_path.exists():
                panel_data = pd.read_csv(panel_data_path, index_col=0, parse_dates=True)
                
                granger_analyzer = GrangerCausalityAnalyzer(panel_data, max_lags=3)
                granger_results = granger_analyzer.run_complete_analysis()
                self.results['granger_causality'] = granger_results
                
                logger.info("✓ 格兰杰因果检验完成")
                logger.info(f"发现 {granger_results['summary']['significant_causal_relationships']} 个显著的因果关系")
            else:
                logger.warning(f"面板数据文件不存在: {panel_data_path}")
            
            # 3. 固定效应模型分析
            logger.info("3. 运行固定效应模型分析...")
            
            if not ecosystem_panel.empty:
                from src.causal_analysis.fixed_effects import PanelModelComparison
                
                # 定义分析变量
                dependent_var = 'attract_stars_growth'
                independent_vars = ['mech_app_creation', 'mech_code_contrib', 'mech_problem_solving']
                
                # 创建模型比较分析器
                comparator = PanelModelComparison(ecosystem_panel)
                
                # 运行模型比较
                comparison_results = comparator.compare_all_models(dependent_var, independent_vars)
                self.results['fixed_effects_models'] = comparison_results
                
                # 可视化比较结果
                comparator.visualize_model_comparison()
                
                logger.info("✓ 固定效应模型分析完成")
                logger.info(f"比较了 {len(comparison_results)} 个不同的面板数据模型")
            
            # 4. 工具变量分析
            logger.info("4. 运行工具变量分析...")
            
            if not ecosystem_panel.empty:
                from src.causal_analysis.fixed_effects import InstrumentalVariablesAnalyzer
                
                iv_analyzer = InstrumentalVariablesAnalyzer(ecosystem_panel.reset_index())
                
                # 识别工具变量
                iv_results = iv_analyzer.identify_instruments('mech_app_creation')
                
                if iv_results['valid_instruments']:
                    logger.info(f"找到 {len(iv_results['valid_instruments'])} 个有效工具变量")
                    
                    # 使用工具变量估计
                    instruments = iv_results['valid_instruments'][:2]  # 使用前2个最佳工具变量
                    sls_results = iv_analyzer.estimate_2sls(
                        dependent_var, ['mech_app_creation'], instruments, 
                        ['mech_code_contrib', 'mech_problem_solving']
                    )
                    
                    if sls_results is not None:
                        self.results['instrumental_variables'] = {
                            'valid_instruments': iv_results['valid_instruments'],
                            'model_estimated': True,
                            'r_squared': sls_results.rsquared
                        }
                        logger.info("✓ 工具变量分析完成")
                else:
                    logger.warning("未找到有效的工具变量")
                    self.results['instrumental_variables'] = {
                        'valid_instruments': [],
                        'model_estimated': False
                    }
                
        except Exception as e:
            logger.error(f"因果分析失败: {e}")
    
    def run_advanced_models(self):
        """运行高级模型分析"""
        logger.info("=" * 60)
        logger.info("开始高级模型分析...")
        logger.info("=" * 60)
        
        try:
            # 1. HAN模型分析
            logger.info("1. 运行异构注意力网络(HAN)分析...")
            
            from src.advanced_models.han_model import HANAnalyzer
            
            # 加载网络图
            graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
            if graph_path.exists():
                import networkx as nx
                graph = nx.read_graphml(str(graph_path))
                
                # 如果图太大，采样进行分析
                if graph.number_of_nodes() > 500:
                    logger.info(f"图较大({graph.number_of_nodes()}个节点)，采样进行HAN分析...")
                    nodes_sample = list(graph.nodes())[:500]
                    graph = graph.subgraph(nodes_sample).copy()
                    logger.info(f"采样后图规模: {graph.number_of_nodes()} 个节点")
                
                # 创建HAN分析器
                han_analyzer = HANAnalyzer(graph)
                
                # 运行完整分析
                han_results = han_analyzer.run_complete_analysis()
                self.results['han_analysis'] = han_results
                
                logger.info("✓ HAN分析完成")
                logger.info(f"模型测试准确率: {han_results.get('test_results', {}).get('test_accuracy', 0):.4f}")
                
            else:
                logger.warning(f"未找到网络图文件: {graph_path}")
            
            # 2. 图嵌入比较分析
            logger.info("2. 运行图嵌入方法比较...")
            
            if graph_path.exists():
                from src.advanced_models.graph_embedding import ComparisonAnalyzer
                
                # 创建比较分析器
                comparator = ComparisonAnalyzer(graph)
                
                # 运行比较分析
                embedding_results = comparator.compare_embedding_methods(embedding_dim=64)
                self.results['embedding_comparison'] = embedding_results
                
                logger.info("✓ 图嵌入比较完成")
                logger.info(f"比较了 {len(embedding_results)} 种嵌入方法")
            
        except Exception as e:
            logger.error(f"高级模型分析失败: {e}")
    
    def generate_summary_report(self):
        """生成分析摘要报告"""
        logger.info("=" * 60)
        logger.info("生成分析摘要报告...")
        logger.info("=" * 60)
        
        try:
            report = {
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_period': {
                    'start_date': ANALYSIS_CONFIG["start_date"],
                    'end_date': ANALYSIS_CONFIG["end_date"]
                },
                'modules_executed': list(self.results.keys()),
                'summary': {}
            }
            
            # 网络分析摘要
            if 'dynamic_network' in self.results:
                dynamic_data = self.results['dynamic_network']
                if not dynamic_data.empty:
                    report['summary']['network_analysis'] = {
                        'total_time_points': len(dynamic_data),
                        'avg_nodes_per_month': dynamic_data['nodes_in_month'].mean(),
                        'avg_edges_per_month': dynamic_data['edges_in_month'].mean(),
                        'centrality_metrics_calculated': [col for col in dynamic_data.columns if 'centrality' in col]
                    }
            
            # 社区检测摘要
            if 'community_comparison' in self.results:
                community_data = self.results['community_comparison']
                if not community_data.empty:
                    best_community_algorithm = community_data.loc[community_data['modularity'].idxmax(), 'algorithm']
                    report['summary']['community_detection'] = {
                        'algorithms_compared': community_data['algorithm'].tolist(),
                        'best_algorithm_by_modularity': best_community_algorithm,
                        'best_modularity_score': community_data['modularity'].max()
                    }
            
            # 用户分析摘要
            if 'user_clustering' in self.results:
                clustering_data = self.results['user_clustering']
                report['summary']['user_analysis'] = {
                    'recommended_clusters': clustering_data['recommended_k'],
                    'best_clustering_algorithm': clustering_data['best_algorithm'],
                    'total_users_analyzed': len(clustering_data['result_df']) if 'result_df' in clustering_data else 0
                }
            
            # 角色演化摘要
            if 'role_evolution' in self.results:
                evolution_data = self.results['role_evolution']
                report['summary']['role_evolution'] = {
                    'time_periods_analyzed': evolution_data.get('num_periods', 0),
                    'role_transitions_found': evolution_data.get('num_transitions', 0),
                    'clustering_algorithm_used': evolution_data.get('clustering_algorithm', 'unknown')
                }
            
            # 因果分析摘要
            causal_summary = {}
            
            # 面板数据构建
            if 'panel_data_construction' in self.results:
                panel_data = self.results['panel_data_construction']
                causal_summary['panel_data'] = {
                    'entities': panel_data['n_entities'],
                    'time_periods': panel_data['n_time_periods'],
                    'observations': panel_data['n_observations']
                }
            
            # 格兰杰因果检验
            if 'granger_causality' in self.results:
                granger_data = self.results['granger_causality']
                causal_summary['granger_causality'] = {
                    'variables_tested': granger_data['summary']['total_variables'],
                    'stationary_variables': granger_data['summary']['stationary_variables'],
                    'causal_relationships_found': granger_data['summary']['significant_causal_relationships'],
                    'total_tests_performed': granger_data['summary']['total_tests']
                }
            
            # 固定效应模型
            if 'fixed_effects_models' in self.results:
                fe_data = self.results['fixed_effects_models']
                causal_summary['fixed_effects'] = {
                    'models_compared': len([k for k in fe_data.keys() if k != 'hausman_test']),
                    'hausman_test_performed': 'hausman_test' in fe_data,
                    'recommended_model': fe_data.get('hausman_test', {}).get('recommendation', 'unknown')
                }
            
            # 工具变量分析
            if 'instrumental_variables' in self.results:
                iv_data = self.results['instrumental_variables']
                causal_summary['instrumental_variables'] = {
                    'valid_instruments_found': len(iv_data['valid_instruments']),
                    'model_estimated': iv_data['model_estimated'],
                    'r_squared': iv_data.get('r_squared', None)
                }
            
            if causal_summary:
                report['summary']['causal_analysis'] = causal_summary
            
            # 高级模型分析摘要
            advanced_summary = {}
            
            # HAN模型分析
            if 'han_analysis' in self.results:
                han_data = self.results['han_analysis']
                advanced_summary['han_model'] = {
                    'model_parameters': han_data.get('model_info', {}).get('total_parameters', 0),
                    'test_accuracy': han_data.get('test_results', {}).get('test_accuracy', 0),
                    'node_types': han_data.get('hetero_data_info', {}).get('node_types', 0),
                    'edge_types': han_data.get('hetero_data_info', {}).get('edge_types', 0),
                    'training_epochs': han_data.get('training_results', {}).get('epochs_trained', 0)
                }
            
            # 图嵌入比较
            if 'embedding_comparison' in self.results:
                embedding_data = self.results['embedding_comparison']
                advanced_summary['graph_embedding'] = {
                    'methods_compared': len(embedding_data),
                    'embedding_methods': list(embedding_data.keys())
                }
            
            if advanced_summary:
                report['summary']['advanced_models'] = advanced_summary
            
            # 保存报告
            import json
            report_path = ANALYSIS_OUTPUT_DIR / "analysis_summary_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"✓ 分析摘要报告已保存至: {report_path}")
            
            # 打印摘要到控制台
            self._print_summary(report)
            
        except Exception as e:
            logger.error(f"生成摘要报告失败: {e}")
    
    def _print_summary(self, report):
        """打印分析摘要到控制台"""
        print("\n" + "=" * 80)
        print("                    开源生态系统分析 - 最终报告")
        print("=" * 80)
        
        print(f"分析时间: {report['analysis_timestamp']}")
        print(f"分析期间: {report['analysis_period']['start_date']} 到 {report['analysis_period']['end_date']}")
        print(f"执行模块: {', '.join(report['modules_executed'])}")
        
        summary = report.get('summary', {})
        
        if 'network_analysis' in summary:
            net_sum = summary['network_analysis']
            print(f"\n📊 网络分析:")
            print(f"  - 分析时间点数: {net_sum['total_time_points']}")
            print(f"  - 平均月度节点数: {net_sum['avg_nodes_per_month']:.0f}")
            print(f"  - 平均月度边数: {net_sum['avg_edges_per_month']:.0f}")
            print(f"  - 计算的中心性指标: {len(net_sum['centrality_metrics_calculated'])}")
        
        if 'community_detection' in summary:
            comm_sum = summary['community_detection']
            print(f"\n🏘️  社区检测:")
            print(f"  - 对比算法数: {len(comm_sum['algorithms_compared'])}")
            print(f"  - 最佳算法: {comm_sum['best_algorithm_by_modularity']}")
            print(f"  - 最高模块度: {comm_sum['best_modularity_score']:.3f}")
        
        if 'user_analysis' in summary:
            user_sum = summary['user_analysis']
            print(f"\n👥 用户分析:")
            print(f"  - 推荐聚类数: {user_sum['recommended_clusters']}")
            print(f"  - 最佳聚类算法: {user_sum['best_clustering_algorithm']}")
            print(f"  - 分析用户数: {user_sum['total_users_analyzed']}")
        
        if 'role_evolution' in summary:
            evol_sum = summary['role_evolution']
            print(f"\n🔄 角色演化:")
            print(f"  - 分析时间段数: {evol_sum['time_periods_analyzed']}")
            print(f"  - 发现角色转移数: {evol_sum['role_transitions_found']}")
            print(f"  - 使用算法: {evol_sum['clustering_algorithm_used']}")
        
        if 'causal_analysis' in summary:
            causal_sum = summary['causal_analysis']
            print(f"\n🔗 因果分析:")
            
            # 面板数据
            if 'panel_data' in causal_sum:
                panel_data = causal_sum['panel_data']
                print(f"  📊 面板数据: {panel_data['entities']} 个实体, {panel_data['time_periods']} 个时间段, {panel_data['observations']} 个观测值")
            
            # 格兰杰因果检验
            if 'granger_causality' in causal_sum:
                granger_data = causal_sum['granger_causality']
                print(f"  ⏰ 格兰杰因果: 测试 {granger_data['variables_tested']} 个变量, 发现 {granger_data['causal_relationships_found']} 个因果关系")
            
            # 固定效应模型
            if 'fixed_effects' in causal_sum:
                fe_data = causal_sum['fixed_effects']
                print(f"  🏛️  固定效应: 比较 {fe_data['models_compared']} 个模型, 推荐 {fe_data['recommended_model']}")
            
            # 工具变量分析
            if 'instrumental_variables' in causal_sum:
                iv_data = causal_sum['instrumental_variables']
                print(f"  🔧 工具变量: 找到 {iv_data['valid_instruments_found']} 个有效工具变量, 模型估计{'成功' if iv_data['model_estimated'] else '失败'}")
        
        if 'advanced_models' in summary:
            advanced_sum = summary['advanced_models']
            print(f"\n🧠 高级模型:")
            
            # HAN模型
            if 'han_model' in advanced_sum:
                han_data = advanced_sum['han_model']
                print(f"  🎯 HAN模型: {han_data['model_parameters']:,} 参数, 测试准确率 {han_data['test_accuracy']:.4f}")
                print(f"    - 异构图: {han_data['node_types']} 种节点类型, {han_data['edge_types']} 种边类型")
                print(f"    - 训练轮数: {han_data['training_epochs']}")
            
            # 图嵌入比较
            if 'graph_embedding' in advanced_sum:
                embedding_data = advanced_sum['graph_embedding']
                print(f"  📊 图嵌入: 比较 {embedding_data['methods_compared']} 种方法 ({', '.join(embedding_data['embedding_methods'])})")
        
        print("\n" + "=" * 80)
        print("分析完成！所有结果已保存到 results/analysis_output/ 目录")
        print("=" * 80)
    
    def run_complete_pipeline(self):
        """运行完整的分析流水线"""
        start_time = datetime.now()
        logger.info("🚀 开始运行完整的开源生态系统分析流水线...")
        
        try:
            # 1. 网络分析
            self.run_network_analysis()
            
            # 2. 用户分析
            self.run_user_analysis()
            
            # 3. 因果分析
            self.run_causal_analysis()
            
            # 4. 高级模型分析
            self.run_advanced_models()
            
            # 5. 生成摘要报告
            self.generate_summary_report()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info(f"🎉 分析流水线执行完成！总耗时: {duration}")
            
        except Exception as e:
            logger.error(f"分析流水线执行失败: {e}")
            raise


def main():
    """主函数入口"""
    print("开源生态系统组织结构与创新模式分析")
    print("=" * 50)
    
    # 检查API配置
    if not api_config.validate_github_token():
        print("⚠️  警告: GitHub Token配置无效")
    
    if not api_config.validate_openai_key():
        print("⚠️  警告: OpenAI API Key配置无效")
    
    # 创建并运行分析流水线
    pipeline = AnalysisPipeline()
    
    try:
        pipeline.run_complete_pipeline()
        print("\n✅ 分析流水线成功执行完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️  分析被用户中断")
        logger.info("分析被用户中断")
        
    except Exception as e:
        print(f"\n❌ 分析流水线执行失败: {e}")
        logger.error(f"分析流水线执行失败: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
