#!/usr/bin/env python
"""
子中心识别和生命周期追踪演示脚本
专门用于演示RQ1相关的子中心分析功能
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
from src.network_analysis.community_detection import (
    CommunityDetector, 
    SubCenterDetector, 
    SubCenterLifecycleTracker,
    analyze_subcenters_and_lifecycle
)

# 设置日志
logger = setup_logger(__name__)


def create_sample_network_for_demo():
    """创建示例网络用于演示子中心识别功能"""
    print("🔧 创建示例异构网络...")
    
    from config.settings import NETWORK_OUTPUT_DIR, FILENAMES
    
    # 确保目录存在
    NETWORK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建异构图
    G = nx.Graph()
    
    # === 添加节点 ===
    
    # 1. 核心团队成员（官方）
    core_team = [
        ("user_0", {"type": "user", "login": "hwchase17", "user_type": "core"}),
        ("user_1", {"type": "user", "login": "agola11", "user_type": "core"}),
        ("user_2", {"type": "user", "login": "baskaryan", "user_type": "core"}),
        ("user_3", {"type": "user", "login": "ccurme", "user_type": "core"}),
    ]
    
    # 2. 社区用户（潜在子中心成员）
    community_users = []
    for i in range(4, 50):
        community_users.append((
            f"user_{i}", 
            {"type": "user", "login": f"community_user_{i}", "user_type": "community"}
        ))
    
    # 3. 仓库节点
    repos = []
    repo_types = ['application', 'library', 'tool', 'tutorial']
    for i in range(100):
        repo_type = np.random.choice(repo_types)
        repos.append((
            f"repo_{i}",
            {
                "type": "repo", 
                "name": f"repo_{i}", 
                "primary_role": repo_type,
                "stars": np.random.randint(5, 1000),
                "forks": np.random.randint(1, 100)
            }
        ))
    
    # 4. PR节点
    prs = []
    contribution_types = ['code', 'doc']
    for i in range(200):
        contrib_type = np.random.choice(contribution_types, p=[0.7, 0.3])
        prs.append((
            f"pr_{i}",
            {
                "type": "pr",
                "contribution_type": contrib_type,
                "timestamp": f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}T10:00:00Z"
            }
        ))
    
    # 5. Issue节点
    issues = []
    for i in range(150):
        issues.append((
            f"issue_{i}",
            {
                "type": "issue",
                "timestamp": f"2023-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}T10:00:00Z"
            }
        ))
    
    # 添加所有节点
    G.add_nodes_from(core_team + community_users + repos + prs + issues)
    
    # === 添加边（模拟不同的交互模式） ===
    
    # 1. 核心团队与主要仓库的连接
    for core_user, _ in core_team:
        for i in range(0, 20):  # 连接前20个仓库
            G.add_edge(core_user, f"repo_{i}", type="maintain")
    
    # 2. 创建几个功能性子中心
    
    # 子中心1：应用开发者群体 (users 4-15)
    app_developers = [f"user_{i}" for i in range(4, 16)]
    app_repos = [f"repo_{i}" for i in range(20, 40) if repos[i-20][1]["primary_role"] == "application"]
    
    for user in app_developers:
        # 连接到应用仓库
        for repo in app_repos[:5]:  # 每人连接5个应用仓库
            G.add_edge(user, repo, type="contribute")
        
        # 创建代码PR
        for i in range(3):
            pr_id = f"pr_{len(app_developers)*3 + i}"
            if pr_id in [node for node, _ in prs]:
                G.add_edge(user, pr_id, type="create", contribution_type="code")
    
    # 子中心2：文档贡献者群体 (users 16-25)
    doc_contributors = [f"user_{i}" for i in range(16, 26)]
    
    for user in doc_contributors:
        # 连接到各种仓库（文档贡献者更多样化）
        for i in range(5):
            repo_id = f"repo_{np.random.randint(0, 50)}"
            G.add_edge(user, repo_id, type="document")
        
        # 创建文档PR
        for i in range(2):
            pr_id = f"pr_{50 + len(doc_contributors)*2 + i}"
            if pr_id in [node for node, _ in prs]:
                G.add_edge(user, pr_id, type="create", contribution_type="doc")
    
    # 子中心3：工具开发者群体 (users 26-35)
    tool_developers = [f"user_{i}" for i in range(26, 36)]
    tool_repos = [f"repo_{i}" for i in range(60, 80) if repos[i-60][1]["primary_role"] == "tool"]
    
    for user in tool_developers:
        # 连接到工具仓库
        for repo in tool_repos[:3]:
            G.add_edge(user, repo, type="develop")
        
        # 创建代码PR
        for i in range(4):  # 工具开发者更活跃
            pr_id = f"pr_{100 + len(tool_developers)*4 + i}"
            if pr_id in [node for node, _ in prs]:
                G.add_edge(user, pr_id, type="create", contribution_type="code")
    
    # 3. 添加Issue交互
    all_users = [f"user_{i}" for i in range(50)]
    for i, (issue_id, _) in enumerate(issues):
        # 随机选择用户与Issue交互
        interacting_users = np.random.choice(all_users, size=np.random.randint(1, 4), replace=False)
        for user in interacting_users:
            G.add_edge(user, issue_id, type="interact")
    
    # 4. 添加时间戳到边
    for u, v, data in G.edges(data=True):
        if 'timestamp' not in data:
            # 为边添加随机时间戳
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            data['timestamp'] = f"2023-{month:02d}-{day:02d}T{np.random.randint(9,18):02d}:00:00Z"
    
    # 保存网络图
    graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
    nx.write_graphml(G, str(graph_path))
    
    print(f"✅ 示例网络创建完成:")
    print(f"   - 节点数: {G.number_of_nodes()}")
    print(f"   - 边数: {G.number_of_edges()}")
    print(f"   - 用户节点: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'user'])}")
    print(f"   - 仓库节点: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'repo'])}")
    print(f"   - PR节点: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'pr'])}")
    print(f"   - Issue节点: {len([n for n, d in G.nodes(data=True) if d.get('type') == 'issue'])}")
    print(f"   - 网络图已保存至: {graph_path}")
    
    return True


def run_subcenter_identification_demo():
    """运行子中心识别演示"""
    print("\n🎯 运行子中心识别演示...")
    
    try:
        from config.settings import NETWORK_OUTPUT_DIR, FILENAMES, ANALYSIS_CONFIG
        
        # 加载网络图
        graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
        G = nx.read_graphml(str(graph_path))
        
        print(f"   - 已加载网络图: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
        
        # 社区检测
        print("   - 执行社区检测...")
        detector = CommunityDetector(G)
        communities = detector.detect_louvain_communities()
        print(f"   - 检测到 {len(communities)} 个社区")
        
        # 子中心识别
        print("   - 识别子中心...")
        subcenter_detector = SubCenterDetector(
            G, 
            core_team_logins=ANALYSIS_CONFIG["core_team_logins"],
            min_subcenter_size=5,  # 降低阈值以便演示
            innovation_threshold=0.01  # 降低阈值以便演示
        )
        
        subcenters = subcenter_detector.identify_sub_centers(communities, 'louvain')
        
        if subcenters:
            print(f"   ✅ 成功识别出 {len(subcenters)} 个子中心")
            
            # 显示子中心详细信息
            for i, sc in enumerate(subcenters):
                print(f"\n   子中心 {i+1}:")
                print(f"     - ID: {sc['subcenter_id']}")
                print(f"     - 规模: {sc['size']} 个节点")
                print(f"     - 功能类型: {sc['functional_type']}")
                print(f"     - 创新得分: {sc['innovation_score']:.3f}")
                print(f"     - 应用创建率: {sc['app_creation_rate']:.3f}")
                print(f"     - 代码贡献率: {sc['code_contribution_rate']:.3f}")
                print(f"     - 知识分享率: {sc['knowledge_sharing_rate']:.3f}")
                print(f"     - 专业化指数: {sc['specialization_index']:.3f}")
                print(f"     - 外部连接度: {sc['external_connectivity']:.3f}")
                print(f"     - 内部密度: {sc['internal_density']:.3f}")
                print(f"     - 关键节点数: {len(sc['key_nodes'])}")
            
            return True
        else:
            print("   ⚠️  未识别出符合条件的子中心")
            return False
            
    except Exception as e:
        print(f"   ❌ 子中心识别演示失败: {e}")
        return False


def run_lifecycle_tracking_demo():
    """运行生命周期追踪演示"""
    print("\n📈 运行生命周期追踪演示...")
    
    try:
        from config.settings import NETWORK_OUTPUT_DIR, FILENAMES
        
        # 加载网络图
        graph_path = NETWORK_OUTPUT_DIR / FILENAMES["graph_file"]
        G = nx.read_graphml(str(graph_path))
        
        print("   - 创建生命周期追踪器...")
        
        # 创建生命周期追踪器（缩短时间范围用于演示）
        lifecycle_tracker = SubCenterLifecycleTracker(
            start_date="2023-01-01",
            end_date="2023-06-30",  # 6个月的演示数据
            similarity_threshold=0.2,  # 降低阈值以便演示
            stability_threshold=2  # 降低稳定性阈值
        )
        
        print("   - 追踪子中心生命周期演化...")
        lifecycle_df = lifecycle_tracker.track_subcenter_evolution(G, algorithm='louvain')
        
        if not lifecycle_df.empty:
            print(f"   ✅ 成功追踪到 {len(lifecycle_df)} 个子中心家族谱系")
            
            # 生成摘要统计
            summary = lifecycle_tracker.generate_lifecycle_summary(lifecycle_df)
            
            print(f"\n   生命周期追踪摘要:")
            print(f"     - 总家族数: {summary['total_subcenters']}")
            print(f"     - 当前活跃: {summary['active_subcenters']}")
            print(f"     - 已消失: {summary['extinct_subcenters']}")
            print(f"     - 稳定子中心: {summary['stable_subcenters']}")
            print(f"     - 平均寿命: {summary['avg_lifespan']:.1f} 个月")
            print(f"     - 最长寿命: {summary['max_lifespan']} 个月")
            print(f"     - 平均峰值规模: {summary['avg_peak_size']:.1f}")
            
            print(f"\n   生命周期类型分布:")
            for lc_type, count in summary['lifecycle_type_distribution'].items():
                print(f"     - {lc_type}: {count}")
            
            print(f"\n   功能类型分布:")
            for func_type, count in summary['functional_type_distribution'].items():
                print(f"     - {func_type}: {count}")
            
            # 显示几个典型的生命周期案例
            print(f"\n   典型生命周期案例:")
            for i, row in lifecycle_df.head(3).iterrows():
                print(f"     案例 {i+1}:")
                print(f"       - 家族ID: {row['lineage_id']}")
                print(f"       - 生命周期类型: {row['lifecycle_type']}")
                print(f"       - 诞生月份: {row['birth_month']}")
                print(f"       - 死亡月份: {row['death_month'] if row['death_month'] else '仍活跃'}")
                print(f"       - 总寿命: {row['total_lifespan_months']} 个月")
                print(f"       - 峰值规模: {row['peak_size']}")
                print(f"       - 主导功能: {row['dominant_functional_type']}")
                print(f"       - 最终状态: {row['final_status']}")
            
            # 可视化生命周期模式
            print("   - 生成生命周期可视化...")
            lifecycle_tracker.visualize_lifecycle_patterns(lifecycle_df)
            
            return True
        else:
            print("   ⚠️  未能追踪到生命周期数据")
            return False
            
    except Exception as e:
        print(f"   ❌ 生命周期追踪演示失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 子中心识别和生命周期追踪 - 功能演示")
    print("=" * 60)
    print("这个演示脚本将展示如何识别网络中的创新子中心")
    print("以及追踪它们的生命周期演化过程")
    print("=" * 60)
    
    start_time = datetime.now()
    
    # 执行演示步骤
    steps = [
        ("创建示例网络", create_sample_network_for_demo),
        ("子中心识别演示", run_subcenter_identification_demo),
        ("生命周期追踪演示", run_lifecycle_tracking_demo)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            result = step_func()
            results.append((step_name, result))
            
            if result:
                print(f"✅ {step_name} - 成功")
            else:
                print(f"⚠️  {step_name} - 部分成功")
                
        except Exception as e:
            print(f"❌ {step_name} - 失败: {e}")
            results.append((step_name, False))
    
    # 汇总结果
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("📋 演示结果汇总")
    print("=" * 60)
    
    success_count = 0
    for step_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {step_name}: {status}")
        if success:
            success_count += 1
    
    success_rate = success_count / len(results) * 100
    
    print(f"\n🎯 成功率: {success_count}/{len(results)} ({success_rate:.0f}%)")
    print(f"⏱️  总耗时: {duration}")
    
    if success_rate >= 80:
        print("\n🎉 子中心识别和生命周期追踪功能演示成功！")
        print("\n📚 关键功能:")
        print("   ✓ 基于社区检测识别非官方创新子中心")
        print("   ✓ 计算子中心的创新指标和功能特征")
        print("   ✓ 追踪子中心的时间演化和生命周期")
        print("   ✓ 分析子中心的涌现、发展、衰退模式")
        print("   ✓ 可视化生命周期分布和演化趋势")
        
        print("\n🔬 研究价值:")
        print("   • 支持RQ1: 网络结构从中心化向多中心演化的实证分析")
        print("   • 识别社区驱动的创新子中心及其功能定位")
        print("   • 量化子中心的生命周期模式和稳定性")
        print("   • 为理解开源生态系统治理演化提供微观机制洞察")
        
    else:
        print("\n⚠️  部分功能存在问题，建议检查:")
        print("   - 网络图数据的完整性")
        print("   - 社区检测算法的可用性")
        print("   - 中心性计算的依赖包")
    
    print(f"\n📁 输出文件位置:")
    print("   - 示例网络: data/network/network_output/full_ecosystem_graph.graphml")
    print("   - 静态子中心: results/analysis_output/static_subcenters_analysis.csv")
    print("   - 生命周期数据: results/analysis_output/subcenter_lifecycle_louvain.csv")
    print("   - 生命周期摘要: results/analysis_output/subcenter_lifecycle_summary.json")
    print("   - 可视化图表: results/analysis_output/subcenter_lifecycle_patterns.png")
    
    return success_rate >= 50


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
