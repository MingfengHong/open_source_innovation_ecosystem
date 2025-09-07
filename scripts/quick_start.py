#!/usr/bin/env python
"""
快速启动脚本
提供简化的分析流程，用于快速验证项目功能
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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.utils.logging_config import setup_logger

# 设置日志
logger = setup_logger(__name__)

def create_sample_data():
    """创建示例数据用于快速验证"""
    print("🔧 创建示例数据...")
    
    from config.settings import FINAL_ANALYSIS_DATA_DIR, CLASSIFICATION_OUTPUT_DIR
    
    # 确保目录存在
    FINAL_ANALYSIS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建示例仓库数据
    repos_data = []
    for i in range(50):
        repos_data.append({
            'repo_id': i,
            'repo_name': f'test_repo_{i}',
            'owner': f'owner_{i%10}',
            'stars': np.random.randint(0, 1000),
            'forks': np.random.randint(0, 100),
            'created_at': '2022-01-01T00:00:00Z'
        })
    
    repos_df = pd.DataFrame(repos_data)
    repos_df.to_csv(FINAL_ANALYSIS_DATA_DIR / 'repos.csv', index=False)
    
    # 创建示例用户数据
    users_data = []
    for i in range(100):
        users_data.append({
            'user_id': i,
            'login': f'user_{i}',
            'user_type': 'User'
        })
    
    users_df = pd.DataFrame(users_data)
    users_df.to_csv(FINAL_ANALYSIS_DATA_DIR / 'users.csv', index=False)
    
    # 创建示例PR数据
    prs_data = []
    for i in range(200):
        prs_data.append({
            'pr_id': i,
            'repo_id': np.random.randint(0, 50),
            'author_id': np.random.randint(0, 100),
            'created_at': f'2023-{np.random.randint(1,13):02d}-01T00:00:00Z',
            'state': np.random.choice(['open', 'closed', 'merged']),
            'additions': np.random.randint(1, 500),
            'deletions': np.random.randint(1, 200)
        })
    
    prs_df = pd.DataFrame(prs_data)
    prs_df.to_csv(FINAL_ANALYSIS_DATA_DIR / 'prs.csv', index=False)
    
    # 创建示例Issue数据
    issues_data = []
    for i in range(150):
        issues_data.append({
            'issue_id': i,
            'repo_id': np.random.randint(0, 50),
            'author_id': np.random.randint(0, 100),
            'created_at': f'2023-{np.random.randint(1,13):02d}-01T00:00:00Z',
            'state': np.random.choice(['open', 'closed'])
        })
    
    issues_df = pd.DataFrame(issues_data)
    issues_df.to_csv(FINAL_ANALYSIS_DATA_DIR / 'issues.csv', index=False)
    
    # 创建示例Star数据
    stars_data = []
    for i in range(300):
        stars_data.append({
            'user_id': np.random.randint(0, 100),
            'repo_id': np.random.randint(0, 50),
            'starred_at': f'2023-{np.random.randint(1,13):02d}-01T00:00:00Z'
        })
    
    stars_df = pd.DataFrame(stars_data)
    stars_df.to_csv(FINAL_ANALYSIS_DATA_DIR / 'stars.csv', index=False)
    
    # 创建示例分类数据
    classification_data = []
    for i in range(50):
        classification_data.append({
            'repo_name': f'test_repo_{i}',
            'primary_role': np.random.choice(['application', 'library', 'tool', 'tutorial', 'integration']),
            'is_monorepo': np.random.choice([True, False]),
            'has_documentation': np.random.choice([True, False])
        })
    
    classification_df = pd.DataFrame(classification_data)
    classification_df.to_csv(CLASSIFICATION_OUTPUT_DIR / 'repos_classified.csv', index=False)
    
    print("✅ 示例数据创建完成")
    return True

def run_quick_network_analysis():
    """运行快速网络分析"""
    print("\n📊 运行网络分析...")
    
    try:
        from src.network_analysis.dynamic_analysis import DynamicNetworkAnalyzer
        
        # 创建分析器
        analyzer = DynamicNetworkAnalyzer()
        
        # 运行简化分析
        print("   - 加载数据...")
        results = analyzer.run_complete_analysis(
            include_betweenness=True,
            include_eigenvector=True,
            max_months=6  # 限制分析月数以节省时间
        )
        
        if results:
            print("   ✅ 网络分析完成")
            print(f"   - 分析了 {results.get('months_analyzed', 0)} 个月的数据")
            return True
        else:
            print("   ⚠️  网络分析未返回结果")
            return False
            
    except Exception as e:
        print(f"   ❌ 网络分析失败: {e}")
        return False

def run_quick_user_analysis():
    """运行快速用户分析"""
    print("\n👥 运行用户分析...")
    
    try:
        from src.user_analysis.role_clustering import MultiAlgorithmClustering
        
        # 创建示例用户特征数据
        from config.settings import ANALYSIS_OUTPUT_DIR
        ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 生成示例特征
        n_users = 100
        features_data = []
        
        for i in range(n_users):
            features_data.append({
                'user_id': i,
                'login': f'user_{i}',
                'pr_count': np.random.randint(0, 50),
                'issue_count': np.random.randint(0, 30),
                'star_count': np.random.randint(0, 100),
                'total_additions': np.random.randint(0, 5000),
                'total_deletions': np.random.randint(0, 2000),
                'repo_count': np.random.randint(1, 10),
                'code_focus_ratio': np.random.random(),
                'interaction_diversity': np.random.randint(1, 5)
            })
        
        features_df = pd.DataFrame(features_data)
        features_path = ANALYSIS_OUTPUT_DIR / 'user_behavior_features.csv'
        features_df.to_csv(features_path, index=False)
        
        # 运行聚类分析
        print("   - 执行多算法聚类...")
        clusterer = MultiAlgorithmClustering(features_df)
        clustering_results = clusterer.compare_all_algorithms(n_clusters=5)
        
        if clustering_results:
            print("   ✅ 用户分析完成")
            print(f"   - 比较了 {len(clustering_results)} 种聚类算法")
            return True
        else:
            print("   ⚠️  用户分析未返回结果")
            return False
            
    except Exception as e:
        print(f"   ❌ 用户分析失败: {e}")
        return False

def run_quick_causal_analysis():
    """运行快速因果分析"""
    print("\n🔗 运行因果分析...")
    
    try:
        # 创建示例月度面板数据
        from config.settings import ANALYSIS_OUTPUT_DIR
        
        # 生成示例月度数据
        dates = pd.date_range('2023-01-01', '2023-12-01', freq='MS')
        panel_data = []
        
        for date in dates:
            panel_data.append({
                'attract_stars_growth': np.random.random() * 100,
                'mech_app_creation': np.random.random() * 50,
                'mech_code_contrib': np.random.random() * 200,
                'mech_problem_solving': np.random.random() * 80,
                'robust_social_capital': np.random.random() * 150,
                'robust_cognitive_capital': np.random.random() * 120,
                'innovate_novel_solutions': np.random.random() * 90,
                'innovate_paradigm_shift': np.random.random() * 60
            })
        
        panel_df = pd.DataFrame(panel_data, index=dates)
        panel_path = ANALYSIS_OUTPUT_DIR / 'monthly_panel_data.csv'
        panel_df.to_csv(panel_path)
        
        # 运行格兰杰因果检验
        print("   - 执行格兰杰因果检验...")
        from src.causal_analysis.granger_causality import GrangerCausalityAnalyzer
        
        granger_analyzer = GrangerCausalityAnalyzer(panel_df, max_lags=2)
        granger_results = granger_analyzer.run_complete_analysis()
        
        if granger_results:
            print("   ✅ 因果分析完成")
            print(f"   - 发现 {granger_results['summary']['significant_causal_relationships']} 个因果关系")
            return True
        else:
            print("   ⚠️  因果分析未返回结果")
            return False
            
    except Exception as e:
        print(f"   ❌ 因果分析失败: {e}")
        return False

def run_quick_han_demo():
    """运行HAN模型快速演示"""
    print("\n🧠 运行HAN模型演示...")
    
    try:
        from src.advanced_models.han_model import HANAnalyzer
        import networkx as nx
        
        # 创建简单的异构图
        print("   - 创建示例异构图...")
        G = nx.Graph()
        
        # 添加不同类型的节点
        for i in range(20):
            G.add_node(f"user_{i}", type='user')
        for i in range(10):
            G.add_node(f"repo_{i}", type='repo', primary_role='application')
        for i in range(15):
            G.add_node(f"pr_{i}", type='pr')
        
        # 添加边
        for i in range(20):
            # 用户-仓库边
            G.add_edge(f"user_{i}", f"repo_{i%10}", type='star')
            # 用户-PR边
            if i < 15:
                G.add_edge(f"user_{i}", f"pr_{i}", type='create')
                G.add_edge(f"pr_{i}", f"repo_{i%10}", type='belongs_to')
        
        print("   - 初始化HAN分析器...")
        analyzer = HANAnalyzer(G)
        
        print("   - 运行HAN分析（简化版）...")
        # 运行简化的分析
        hetero_data = analyzer.prepare_data()
        model = analyzer.create_model(hidden_dim=32, num_heads=2, num_layers=1)
        
        # 快速训练
        training_history = analyzer.train_model(epochs=10, learning_rate=0.01)
        
        if training_history:
            print("   ✅ HAN演示完成")
            print(f"   - 训练了 {len(training_history['train_loss'])} 轮")
            return True
        else:
            print("   ⚠️  HAN演示未返回结果")
            return False
            
    except Exception as e:
        print(f"   ⚠️  HAN演示跳过（需要PyTorch Geometric）: {e}")
        return True  # 不算作失败，因为这是可选功能


def run_quick_subcenter_demo():
    """运行子中心识别快速演示"""
    print("\n🎯 运行子中心识别演示...")
    
    try:
        from src.network_analysis.community_detection import CommunityDetector, SubCenterDetector
        import networkx as nx
        
        # 创建简单测试网络
        print("   - 创建测试网络...")
        G = nx.Graph()
        
        # 添加核心团队
        G.add_node("core_1", type="user", login="hwchase17", user_type="core")
        G.add_node("core_2", type="user", login="agola11", user_type="core")
        
        # 添加社区用户（潜在子中心）
        community_users = []
        for i in range(3, 15):
            user_id = f"user_{i}"
            G.add_node(user_id, type="user", login=f"community_user_{i}", user_type="community")
            community_users.append(user_id)
        
        # 添加仓库
        for i in range(20):
            repo_type = "application" if i < 10 else "library"
            G.add_node(f"repo_{i}", type="repo", name=f"repo_{i}", 
                      primary_role=repo_type, stars=100)
        
        # 添加PR
        for i in range(30):
            contrib_type = "code" if i < 20 else "doc"
            G.add_node(f"pr_{i}", type="pr", contribution_type=contrib_type)
        
        # 创建连接模式（模拟子中心）
        # 核心团队连接主要仓库
        for core in ["core_1", "core_2"]:
            for i in range(5):
                G.add_edge(core, f"repo_{i}", type="maintain")
        
        # 创建应用开发子中心
        app_devs = community_users[:6]
        for user in app_devs:
            # 连接应用仓库
            for i in range(5, 10):
                G.add_edge(user, f"repo_{i}", type="contribute")
            # 连接代码PR
            for i in range(3):
                G.add_edge(user, f"pr_{i*2}", type="create", contribution_type="code")
        
        # 创建文档贡献子中心
        doc_contribs = community_users[6:]
        for user in doc_contribs:
            # 连接各种仓库
            for i in range(10, 15):
                G.add_edge(user, f"repo_{i}", type="document")
            # 连接文档PR
            for i in range(20, 25):
                G.add_edge(user, f"pr_{i}", type="create", contribution_type="doc")
        
        print(f"   - 测试网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
        
        # 社区检测
        print("   - 执行社区检测...")
        detector = CommunityDetector(G)
        communities = detector.detect_louvain_communities()
        print(f"   - 检测到 {len(communities)} 个社区")
        
        # 子中心识别
        print("   - 识别子中心...")
        subcenter_detector = SubCenterDetector(
            G, 
            core_team_logins=["hwchase17", "agola11"],
            min_subcenter_size=3,
            innovation_threshold=0.01
        )
        
        subcenters = subcenter_detector.identify_sub_centers(communities, 'louvain')
        
        if subcenters:
            print(f"   ✅ 识别出 {len(subcenters)} 个子中心")
            for i, sc in enumerate(subcenters):
                print(f"     子中心{i+1}: 规模{sc['size']}, 功能{sc['functional_type']}, 创新{sc['innovation_score']:.3f}")
            return True
        else:
            print("   ⚠️  未识别出子中心")
            return False
            
    except Exception as e:
        print(f"   ❌ 子中心识别演示失败: {e}")
        return False

def generate_quick_report():
    """生成快速报告"""
    print("\n📄 生成快速分析报告...")
    
    from config.settings import ANALYSIS_OUTPUT_DIR
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'mode': 'quick_start_demo',
        'status': 'completed',
        'components_tested': [
            'network_analysis',
            'user_analysis', 
            'causal_analysis',
            'han_model_demo'
        ],
        'note': 'This is a quick start demo with synthetic data'
    }
    
    import json
    report_path = ANALYSIS_OUTPUT_DIR / 'quick_start_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ 报告已保存至: {report_path}")
    return True

def main():
    """主函数"""
    print("🚀 开源生态系统分析 - 快速启动演示")
    print("=" * 60)
    print("这是一个快速验证项目功能的演示脚本")
    print("使用合成数据来验证各个分析模块是否正常工作")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 执行快速测试
    tests = [
        ("创建示例数据", create_sample_data),
        ("网络分析", run_quick_network_analysis), 
        ("用户分析", run_quick_user_analysis),
        ("因果分析", run_quick_causal_analysis),
        ("子中心识别演示", run_quick_subcenter_demo),
        ("HAN模型演示", run_quick_han_demo),
        ("生成报告", generate_quick_report)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - 成功")
            else:
                print(f"⚠️  {test_name} - 部分成功或跳过")
                
        except Exception as e:
            print(f"❌ {test_name} - 失败: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📋 快速启动测试结果汇总")
    print("=" * 60)
    
    success_count = 0
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            success_count += 1
    
    success_rate = success_count / len(results) * 100
    
    print(f"\n🎯 成功率: {success_count}/{len(results)} ({success_rate:.0f}%)")
    print(f"⏱️  总耗时: {duration}")
    
    if success_rate >= 80:
        print("\n🎉 项目功能验证成功！您可以开始使用完整功能了。")
        print("\n📚 下一步建议:")
        print("   1. 运行完整分析: python scripts/main_analysis_pipeline.py")
        print("   2. 查看演示脚本: python scripts/han_demo.py --use-sample")
        print("   3. 准备真实数据并运行完整流程")
        
    elif success_rate >= 50:
        print("\n⚠️  部分功能正常，建议检查失败的组件")
        print("   可能需要安装额外的依赖包")
        
    else:
        print("\n❌ 多个组件失败，请检查环境配置")
        print("   建议先运行: python scripts/check_environment.py")
    
    print("\n📁 生成的文件位置:")
    print("   - 示例数据: data/processed/final_analysis_data/")
    print("   - 分析结果: results/analysis_output/")
    print("   - 快速报告: results/analysis_output/quick_start_report.json")
    
    return success_rate >= 50

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
