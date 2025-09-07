#!/usr/bin/env python
"""
子中心识别功能的简单测试脚本
用于验证新增的子中心识别和生命周期追踪功能
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


def create_simple_test_network():
    """创建一个简单的测试网络"""
    print("🔧 创建简单测试网络...")
    
    G = nx.Graph()
    
    # 添加节点
    # 核心团队
    G.add_node("core_1", type="user", login="hwchase17", user_type="core")
    G.add_node("core_2", type="user", login="agola11", user_type="core")
    
    # 社区用户 - 子中心1（应用开发者）
    app_users = []
    for i in range(3, 8):
        user_id = f"app_user_{i}"
        G.add_node(user_id, type="user", login=f"app_dev_{i}", user_type="community")
        app_users.append(user_id)
    
    # 社区用户 - 子中心2（文档贡献者）
    doc_users = []
    for i in range(8, 12):
        user_id = f"doc_user_{i}"
        G.add_node(user_id, type="user", login=f"doc_contrib_{i}", user_type="community")
        doc_users.append(user_id)
    
    # 仓库节点
    repos = []
    for i in range(20):
        repo_id = f"repo_{i}"
        if i < 10:
            primary_role = "application"
        elif i < 15:
            primary_role = "library"
        else:
            primary_role = "tool"
        
        G.add_node(repo_id, type="repo", name=f"test_repo_{i}", 
                  primary_role=primary_role, stars=100, forks=10)
        repos.append(repo_id)
    
    # PR节点
    for i in range(30):
        pr_id = f"pr_{i}"
        contrib_type = "code" if i < 20 else "doc"
        G.add_node(pr_id, type="pr", contribution_type=contrib_type,
                  timestamp="2023-06-01T10:00:00Z")
    
    # 添加边
    # 核心团队连接主要仓库
    for core in ["core_1", "core_2"]:
        for i in range(5):
            G.add_edge(core, f"repo_{i}", type="maintain")
    
    # 应用开发者子中心
    for user in app_users:
        # 连接应用仓库
        for i in range(5, 10):
            G.add_edge(user, f"repo_{i}", type="contribute")
        # 连接代码PR
        for i in range(5):
            G.add_edge(user, f"pr_{i}", type="create", contribution_type="code")
    
    # 文档贡献者子中心
    for user in doc_users:
        # 连接各种仓库
        for i in range(10, 15):
            G.add_edge(user, f"repo_{i}", type="document")
        # 连接文档PR
        for i in range(20, 25):
            G.add_edge(user, f"pr_{i}", type="create", contribution_type="doc")
    
    # 添加子中心内部连接（通过共同的PR）
    for i, user1 in enumerate(app_users):
        for user2 in app_users[i+1:]:
            # 通过共同的PR连接
            G.add_edge(user1, f"pr_{i+10}", type="collaborate")
            G.add_edge(user2, f"pr_{i+10}", type="collaborate")
    
    print(f"✅ 测试网络创建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    return G


def test_subcenter_detection():
    """测试子中心检测功能"""
    print("\n🎯 测试子中心检测功能...")
    
    try:
        # 创建测试网络
        G = create_simple_test_network()
        
        # 导入子中心检测器
        from src.network_analysis.community_detection import CommunityDetector, SubCenterDetector
        
        # 社区检测
        print("   - 执行社区检测...")
        detector = CommunityDetector(G)
        communities = detector.detect_louvain_communities()
        print(f"   - 检测到 {len(communities)} 个社区")
        
        # 打印社区详情
        for i, community in enumerate(communities):
            print(f"     社区 {i}: {len(community)} 个节点")
            user_nodes = [n for n in community if G.nodes[n].get('type') == 'user']
            print(f"       用户节点: {user_nodes}")
        
        # 子中心识别
        print("   - 识别子中心...")
        subcenter_detector = SubCenterDetector(
            G, 
            core_team_logins=["hwchase17", "agola11"],
            min_subcenter_size=3,  # 降低阈值用于测试
            innovation_threshold=0.01  # 降低阈值用于测试
        )
        
        subcenters = subcenter_detector.identify_sub_centers(communities, 'louvain')
        
        if subcenters:
            print(f"   ✅ 成功识别出 {len(subcenters)} 个子中心")
            
            for i, sc in enumerate(subcenters):
                print(f"\n   子中心 {i+1}:")
                print(f"     - ID: {sc['subcenter_id']}")
                print(f"     - 规模: {sc['size']} 个节点")
                print(f"     - 功能类型: {sc['functional_type']}")
                print(f"     - 创新得分: {sc['innovation_score']:.3f}")
                print(f"     - 关键节点: {sc['key_nodes']}")
                print(f"     - 是否官方主导: {sc['is_official_dominated']}")
            
            return True
        else:
            print("   ⚠️  未识别出符合条件的子中心")
            return False
            
    except Exception as e:
        print(f"   ❌ 子中心检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_community_detection():
    """测试基础社区检测功能"""
    print("\n🔍 测试基础社区检测功能...")
    
    try:
        # 创建测试网络
        G = create_simple_test_network()
        
        # 导入社区检测器
        from src.network_analysis.community_detection import CommunityDetector
        
        # 创建检测器
        detector = CommunityDetector(G)
        
        # 测试Louvain算法
        print("   - 测试Louvain算法...")
        louvain_communities = detector.detect_louvain_communities()
        print(f"     Louvain: {len(louvain_communities)} 个社区")
        
        # 计算模块度
        modularity = detector.calculate_modularity(louvain_communities)
        print(f"     模块度: {modularity:.3f}")
        
        # 测试算法比较
        print("   - 测试算法比较...")
        comparison_df = detector.compare_algorithms()
        
        if not comparison_df.empty:
            print("   ✅ 社区检测功能正常")
            print(f"     比较了 {len(comparison_df)} 种算法")
            return True
        else:
            print("   ⚠️  算法比较返回空结果")
            return False
            
    except Exception as e:
        print(f"   ❌ 社区检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🚀 子中心识别功能 - 简单测试")
    print("=" * 50)
    
    start_time = datetime.now()
    
    # 测试步骤
    tests = [
        ("基础社区检测", test_basic_community_detection),
        ("子中心识别", test_subcenter_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - 成功")
            else:
                print(f"⚠️  {test_name} - 失败")
                
        except Exception as e:
            print(f"❌ {test_name} - 异常: {e}")
            results.append((test_name, False))
    
    # 汇总结果
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 50)
    print("📋 测试结果汇总")
    print("=" * 50)
    
    success_count = 0
    for test_name, success in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"   {test_name}: {status}")
        if success:
            success_count += 1
    
    success_rate = success_count / len(results) * 100
    
    print(f"\n🎯 成功率: {success_count}/{len(results)} ({success_rate:.0f}%)")
    print(f"⏱️  总耗时: {duration}")
    
    if success_rate >= 50:
        print("\n🎉 子中心识别功能基本正常！")
        print("   可以进行更复杂的测试和真实数据分析")
    else:
        print("\n⚠️  存在问题，需要检查:")
        print("   - 依赖包是否正确安装")
        print("   - 代码逻辑是否有错误")
    
    return success_rate >= 50


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
