"""
动态网络分析模块 (增强版)
支持多种中心性度量的时间序列分析
"""

import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Optional
import logging

from ..utils.logging_config import setup_logger
from .centrality_measures import CentralityCalculator, calculate_centrality_for_core_team
from config.settings import (
    NETWORK_OUTPUT_DIR, ANALYSIS_OUTPUT_DIR, FILENAMES,
    ANALYSIS_CONFIG, VISUALIZATION_CONFIG
)

# 设置日志
logger = setup_logger(__name__)


class DynamicNetworkAnalyzer:
    """动态网络分析器"""
    
    def __init__(self, 
                 graph_path: Optional[str] = None,
                 core_team_logins: Optional[List[str]] = None,
                 start_date: str = None,
                 end_date: str = None):
        """
        初始化动态网络分析器
        
        Args:
            graph_path: 网络图文件路径
            core_team_logins: 核心团队成员登录名列表
            start_date: 分析开始日期
            end_date: 分析结束日期
        """
        self.graph_path = graph_path or (NETWORK_OUTPUT_DIR / FILENAMES["graph_file"])
        self.core_team_logins = core_team_logins or ANALYSIS_CONFIG["core_team_logins"]
        self.start_date = start_date or ANALYSIS_CONFIG["start_date"]
        self.end_date = end_date or ANALYSIS_CONFIG["end_date"]
        self.centrality_types = ANALYSIS_CONFIG["centrality_types"]
        
        self.graph = None
        self.core_team_node_ids = []
        self.analysis_results = []
    
    def load_graph(self):
        """加载网络图"""
        logger.info(f"正在从 '{self.graph_path}' 加载网络图...")
        try:
            self.graph = nx.read_graphml(str(self.graph_path))
            logger.info("网络图加载成功")
            return True
        except FileNotFoundError:
            logger.error(f"未找到图文件 '{self.graph_path}'")
            return False
        except Exception as e:
            logger.error(f"加载图文件时出错: {e}")
            return False
    
    def identify_core_team_nodes(self):
        """识别核心团队节点ID"""
        if not self.graph:
            logger.error("图未加载，无法识别核心团队节点")
            return False
        
        # 创建从login到node_id的映射
        login_to_node_id = {
            data['login']: node 
            for node, data in self.graph.nodes(data=True) 
            if data.get('type') == 'user' and 'login' in data
        }
        
        self.core_team_node_ids = []
        for login in self.core_team_logins:
            if login in login_to_node_id:
                self.core_team_node_ids.append(login_to_node_id[login])
            else:
                logger.warning(f"核心成员 '{login}' 未在图中找到")
        
        if not self.core_team_node_ids:
            logger.error("核心团队成员ID列表为空，无法继续分析")
            return False
        
        logger.info(f"成功识别 {len(self.core_team_node_ids)} 位核心团队成员的节点ID")
        return True
    
    def create_monthly_snapshot(self, start_of_month: pd.Timestamp, end_of_month: pd.Timestamp) -> nx.Graph:
        """
        创建指定月份的网络快照
        
        Args:
            start_of_month: 月份开始时间
            end_of_month: 月份结束时间
            
        Returns:
            nx.Graph: 月度网络快照
        """
        G_month = nx.Graph()
        G_month.add_nodes_from(self.graph.nodes(data=True))
        
        # 添加时间范围内的边
        edges_in_month = []
        for u, v, data in self.graph.edges(data=True):
            if 'timestamp' in data:
                try:
                    ts = pd.to_datetime(data['timestamp']).tz_localize(None)
                    if start_of_month <= ts <= end_of_month:
                        edges_in_month.append((u, v))
                except (ValueError, TypeError):
                    continue
        
        G_month.add_edges_from(edges_in_month)
        G_month.remove_nodes_from(list(nx.isolates(G_month)))
        
        return G_month
    
    def analyze_monthly_centralities(self, 
                                   include_betweenness: bool = True,
                                   betweenness_sample_k: Optional[int] = 1000) -> List[Dict]:
        """
        分析月度中心性变化
        
        Args:
            include_betweenness: 是否包含介数中心性（计算较慢）
            betweenness_sample_k: 介数中心性计算的采样数量
            
        Returns:
            List[Dict]: 每月分析结果列表
        """
        if not self.graph or not self.core_team_node_ids:
            logger.error("图或核心团队节点未初始化")
            return []
        
        logger.info("开始按月生成网络快照并计算多种中心性...")
        
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='MS')
        results = []
        
        for start_of_month in tqdm(date_range, desc="分析各月份网络"):
            end_of_month = start_of_month + pd.offsets.MonthEnd(0)
            
            # 创建月度快照
            G_month = self.create_monthly_snapshot(start_of_month, end_of_month)
            
            if G_month.number_of_nodes() == 0 or G_month.number_of_edges() == 0:
                continue
            
            # 初始化结果字典
            month_result = {
                "month": start_of_month,
                "nodes_in_month": G_month.number_of_nodes(),
                "edges_in_month": G_month.number_of_edges()
            }
            
            # 计算多种中心性度量
            calculator = CentralityCalculator(G_month)
            
            try:
                # 计算各种中心性的核心团队平均值
                centrality_results = calculate_centrality_for_core_team(
                    G_month, 
                    self.core_team_node_ids,
                    centrality_types=self.centrality_types if include_betweenness else 
                                   [ct for ct in self.centrality_types if ct != 'betweenness']
                )
                month_result.update(centrality_results)
                
                # 计算全网络中心性统计
                for centrality_type in self.centrality_types:
                    if not include_betweenness and centrality_type == 'betweenness':
                        continue
                    
                    try:
                        if centrality_type == 'degree':
                            centrality = calculator.calculate_degree_centrality()
                        elif centrality_type == 'betweenness':
                            centrality = calculator.calculate_betweenness_centrality(k=betweenness_sample_k)
                        elif centrality_type == 'eigenvector':
                            centrality = calculator.calculate_eigenvector_centrality()
                        elif centrality_type == 'closeness':
                            centrality = calculator.calculate_closeness_centrality()
                        else:
                            continue
                        
                        # 计算全网络统计
                        values = list(centrality.values())
                        month_result.update({
                            f'{centrality_type}_mean': np.mean(values),
                            f'{centrality_type}_std': np.std(values),
                            f'{centrality_type}_max': np.max(values)
                        })
                        
                    except Exception as e:
                        logger.warning(f"计算{centrality_type}中心性时出错: {e}")
                        month_result.update({
                            f'avg_core_{centrality_type}_centrality': 0,
                            f'{centrality_type}_mean': 0,
                            f'{centrality_type}_std': 0,
                            f'{centrality_type}_max': 0
                        })
                
            except Exception as e:
                logger.error(f"月份 {start_of_month} 的中心性计算失败: {e}")
                continue
            
            results.append(month_result)
        
        logger.info("所有月份的中心性计算完成")
        return results
    
    def visualize_centrality_trends(self, results: List[Dict], save_path: Optional[str] = None):
        """
        可视化中心性变化趋势
        
        Args:
            results: 分析结果列表
            save_path: 图表保存路径
        """
        if not results:
            logger.warning("没有分析结果可视化")
            return
        
        results_df = pd.DataFrame(results)
        
        # 应用可视化配置
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        # 创建子图
        centrality_cols = [col for col in results_df.columns if col.startswith('avg_core_') and col.endswith('_centrality')]
        n_centralities = len(centrality_cols)
        
        if n_centralities == 0:
            logger.warning("没有找到中心性数据列")
            return
        
        fig, axes = plt.subplots(n_centralities, 1, 
                                figsize=(VISUALIZATION_CONFIG["figure_size"][0], 
                                        VISUALIZATION_CONFIG["figure_size"][1] * n_centralities // 2),
                                sharex=True)
        
        if n_centralities == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_centralities))
        
        for i, (col, color) in enumerate(zip(centrality_cols, colors)):
            ax = axes[i]
            centrality_name = col.replace('avg_core_', '').replace('_centrality', '').title()
            
            ax.plot(results_df['month'], results_df[col], 
                   marker='o', linestyle='-', color=color, linewidth=2, markersize=6)
            ax.set_title(f'{centrality_name} Centrality of Core Team Over Time', 
                        fontsize=VISUALIZATION_CONFIG["title_font_size"])
            ax.set_ylabel(f'Average {centrality_name} Centrality', 
                         fontsize=VISUALIZATION_CONFIG["font_size"])
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(results_df) > 2:
                z = np.polyfit(range(len(results_df)), results_df[col], 1)
                p = np.poly1d(z)
                ax.plot(results_df['month'], p(range(len(results_df))), 
                       "--", alpha=0.8, color=color, linewidth=1)
        
        # 设置X轴
        axes[-1].set_xlabel('Month', fontsize=VISUALIZATION_CONFIG["font_size"])
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = ANALYSIS_OUTPUT_DIR / "enhanced_core_team_centrality_trends.png"
        
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"中心性趋势图已保存至: {save_path}")
        
        plt.show()
    
    def run_complete_analysis(self, include_betweenness: bool = True) -> pd.DataFrame:
        """
        运行完整的动态网络分析
        
        Args:
            include_betweenness: 是否包含介数中心性计算
            
        Returns:
            pd.DataFrame: 分析结果数据框
        """
        logger.info("开始完整的动态网络分析...")
        
        # 加载图
        if not self.load_graph():
            return pd.DataFrame()
        
        # 识别核心团队节点
        if not self.identify_core_team_nodes():
            return pd.DataFrame()
        
        # 分析月度中心性
        self.analysis_results = self.analyze_monthly_centralities(include_betweenness=include_betweenness)
        
        if not self.analysis_results:
            logger.error("没有生成分析结果")
            return pd.DataFrame()
        
        # 转换为DataFrame
        results_df = pd.DataFrame(self.analysis_results)
        
        # 保存结果
        output_path = ANALYSIS_OUTPUT_DIR / "enhanced_dynamic_analysis_results.csv"
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"分析结果已保存至: {output_path}")
        
        # 可视化
        self.visualize_centrality_trends(self.analysis_results)
        
        # 打印统计摘要
        self._print_analysis_summary(results_df)
        
        return results_df
    
    def _print_analysis_summary(self, results_df: pd.DataFrame):
        """打印分析结果摘要"""
        logger.info("\n--- 动态网络分析结果摘要 ---")
        logger.info(f"分析时间范围: {self.start_date} 到 {self.end_date}")
        logger.info(f"分析月份数: {len(results_df)}")
        logger.info(f"核心团队成员数: {len(self.core_team_node_ids)}")
        
        # 中心性趋势分析
        centrality_cols = [col for col in results_df.columns if col.startswith('avg_core_') and col.endswith('_centrality')]
        
        for col in centrality_cols:
            centrality_name = col.replace('avg_core_', '').replace('_centrality', '')
            mean_value = results_df[col].mean()
            trend = "上升" if results_df[col].iloc[-1] > results_df[col].iloc[0] else "下降"
            logger.info(f"{centrality_name.title()} 中心性: 平均值 {mean_value:.4f}, 整体趋势 {trend}")


def main():
    """主函数入口"""
    # 确保输出目录存在
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 创建分析器并运行分析
    analyzer = DynamicNetworkAnalyzer()
    results_df = analyzer.run_complete_analysis(include_betweenness=True)
    
    if not results_df.empty:
        logger.info("动态网络分析完成！")
    else:
        logger.error("动态网络分析失败")


if __name__ == "__main__":
    main()
