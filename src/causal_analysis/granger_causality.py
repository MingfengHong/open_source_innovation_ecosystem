"""
格兰杰因果检验模块
实现时间序列的因果关系分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# 统计分析工具
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, VISUALIZATION_CONFIG

# 设置日志
logger = setup_logger(__name__)


class GrangerCausalityAnalyzer:
    """格兰杰因果检验分析器"""
    
    def __init__(self, 
                 panel_data: pd.DataFrame,
                 date_column: str = None,
                 max_lags: int = 4,
                 alpha: float = 0.05):
        """
        初始化格兰杰因果检验分析器
        
        Args:
            panel_data: 面板数据
            date_column: 日期列名
            max_lags: 最大滞后期数
            alpha: 显著性水平
        """
        self.panel_data = panel_data.copy()
        self.date_column = date_column or panel_data.index.name or 'date'
        self.max_lags = max_lags
        self.alpha = alpha
        
        # 如果日期列不是索引，则设置为索引
        if self.date_column in self.panel_data.columns:
            self.panel_data.set_index(self.date_column, inplace=True)
        
        # 确保索引是日期时间类型
        if not isinstance(self.panel_data.index, pd.DatetimeIndex):
            self.panel_data.index = pd.to_datetime(self.panel_data.index)
        
        # 排序确保时间序列顺序
        self.panel_data.sort_index(inplace=True)
        
        # 存储分析结果
        self.stationarity_results = {}
        self.granger_results = {}
        self.var_model = None
        
        logger.info(f"初始化格兰杰因果检验分析器: {len(self.panel_data)} 个时间点, {len(self.panel_data.columns)} 个变量")
    
    def check_stationarity(self, 
                          variables: List[str] = None,
                          method: str = 'adf') -> Dict[str, Dict]:
        """
        检验时间序列的平稳性
        
        Args:
            variables: 要检验的变量列表，None表示检验所有变量
            method: 检验方法，'adf'表示ADF检验
            
        Returns:
            Dict[str, Dict]: 平稳性检验结果
        """
        if variables is None:
            variables = self.panel_data.columns.tolist()
        
        logger.info(f"检验 {len(variables)} 个变量的平稳性...")
        
        stationarity_results = {}
        
        for var in variables:
            if var not in self.panel_data.columns:
                logger.warning(f"变量 {var} 不存在于数据中")
                continue
            
            series = self.panel_data[var].dropna()
            
            if len(series) < 10:
                logger.warning(f"变量 {var} 的有效观测值太少 ({len(series)})，跳过平稳性检验")
                continue
            
            try:
                if method == 'adf':
                    # ADF检验
                    adf_result = adfuller(series, autolag='AIC')
                    
                    stationarity_results[var] = {
                        'method': 'ADF',
                        'test_statistic': adf_result[0],
                        'p_value': adf_result[1],
                        'critical_values': adf_result[4],
                        'is_stationary': adf_result[1] < self.alpha,
                        'conclusion': 'Stationary' if adf_result[1] < self.alpha else 'Non-stationary'
                    }
                    
            except Exception as e:
                logger.error(f"变量 {var} 的平稳性检验失败: {e}")
                stationarity_results[var] = {
                    'method': method,
                    'error': str(e),
                    'is_stationary': False,
                    'conclusion': 'Test failed'
                }
        
        self.stationarity_results = stationarity_results
        
        # 保存平稳性检验结果
        self._save_stationarity_results()
        
        return stationarity_results
    
    def perform_granger_causality_test(self, 
                                     variable_pairs: List[Tuple[str, str]] = None,
                                     auto_generate_pairs: bool = True) -> Dict[str, Dict]:
        """
        执行格兰杰因果检验
        
        Args:
            variable_pairs: 变量对列表，每个元组为(原因变量, 结果变量)
            auto_generate_pairs: 是否自动生成所有可能的变量对
            
        Returns:
            Dict[str, Dict]: 格兰杰因果检验结果
        """
        # 生成变量对
        if variable_pairs is None and auto_generate_pairs:
            variable_pairs = self._generate_variable_pairs()
        elif variable_pairs is None:
            logger.error("未提供变量对且未启用自动生成")
            return {}
        
        logger.info(f"执行 {len(variable_pairs)} 个变量对的格兰杰因果检验...")
        
        granger_results = {}
        
        for cause_var, effect_var in variable_pairs:
            pair_key = f"{cause_var} -> {effect_var}"
            
            if cause_var not in self.panel_data.columns or effect_var not in self.panel_data.columns:
                logger.warning(f"变量对 {pair_key} 中存在不存在的变量")
                continue
            
            # 准备数据
            data_subset = self.panel_data[[effect_var, cause_var]].dropna()
            
            if len(data_subset) < max(10, self.max_lags + 5):
                logger.warning(f"变量对 {pair_key} 的有效观测值太少 ({len(data_subset)})")
                continue
            
            try:
                # 执行格兰杰因果检验
                gc_result = grangercausalitytests(
                    data_subset, 
                    maxlag=self.max_lags, 
                    verbose=False
                )
                
                # 提取结果
                test_results = {}
                for lag in range(1, self.max_lags + 1):
                    if lag in gc_result:
                        # 获取F检验结果
                        f_test = gc_result[lag][0]['ssr_ftest']
                        lratio_test = gc_result[lag][0]['lratio']
                        
                        test_results[f'lag_{lag}'] = {
                            'f_statistic': f_test[0],
                            'f_pvalue': f_test[1],
                            'lr_statistic': lratio_test[0],
                            'lr_pvalue': lratio_test[1],
                            'significant_f': f_test[1] < self.alpha,
                            'significant_lr': lratio_test[1] < self.alpha
                        }
                
                # 找到最佳滞后期（p值最小的）
                best_lag = min(test_results.keys(), 
                             key=lambda x: test_results[x]['f_pvalue'])
                
                granger_results[pair_key] = {
                    'cause_variable': cause_var,
                    'effect_variable': effect_var,
                    'test_results': test_results,
                    'best_lag': best_lag,
                    'best_f_pvalue': test_results[best_lag]['f_pvalue'],
                    'is_causal': test_results[best_lag]['f_pvalue'] < self.alpha,
                    'conclusion': f"{cause_var} Granger-causes {effect_var}" if test_results[best_lag]['f_pvalue'] < self.alpha else "No Granger causality"
                }
                
            except Exception as e:
                logger.error(f"变量对 {pair_key} 的格兰杰因果检验失败: {e}")
                granger_results[pair_key] = {
                    'cause_variable': cause_var,
                    'effect_variable': effect_var,
                    'error': str(e),
                    'is_causal': False,
                    'conclusion': 'Test failed'
                }
        
        self.granger_results = granger_results
        
        # 保存格兰杰因果检验结果
        self._save_granger_results()
        
        return granger_results
    
    def analyze_var_model(self, 
                         variables: List[str] = None,
                         ic: str = 'aic') -> Dict[str, Any]:
        """
        构建和分析向量自回归(VAR)模型
        
        Args:
            variables: 包含在VAR模型中的变量
            ic: 信息准则，用于选择最优滞后期
            
        Returns:
            Dict[str, Any]: VAR模型分析结果
        """
        if variables is None:
            # 选择数值型变量
            variables = self.panel_data.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"构建包含 {len(variables)} 个变量的VAR模型...")
        
        # 准备数据
        var_data = self.panel_data[variables].dropna()
        
        if len(var_data) < 20:
            logger.error(f"VAR模型数据点太少: {len(var_data)}")
            return {}
        
        try:
            # 创建VAR模型
            var_model = VAR(var_data)
            
            # 选择最优滞后期
            lag_order_results = var_model.select_order(maxlags=self.max_lags)
            optimal_lag = getattr(lag_order_results, ic)
            
            logger.info(f"根据{ic.upper()}准则选择的最优滞后期: {optimal_lag}")
            
            # 拟合VAR模型
            var_fitted = var_model.fit(optimal_lag)
            self.var_model = var_fitted
            
            # 模型诊断
            # 1. 残差的序列相关检验
            ljung_box_results = {}
            for i, var_name in enumerate(variables):
                residuals = var_fitted.resid.iloc[:, i]
                lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
                ljung_box_results[var_name] = {
                    'lb_stat': lb_test['lb_stat'].iloc[-1],
                    'lb_pvalue': lb_test['lb_pvalue'].iloc[-1],
                    'no_autocorr': lb_test['lb_pvalue'].iloc[-1] > self.alpha
                }
            
            # 2. 脉冲响应分析
            irf = var_fitted.irf(periods=10)
            
            var_results = {
                'variables': variables,
                'optimal_lag': optimal_lag,
                'aic': var_fitted.aic,
                'bic': var_fitted.bic,
                'hqic': var_fitted.hqic,
                'model_summary': str(var_fitted.summary()),
                'ljung_box_results': ljung_box_results,
                'irf_available': True,
                'forecast_error_variance_decomp': True
            }
            
            # 保存VAR模型结果
            self._save_var_results(var_results)
            
            # 可视化脉冲响应函数
            self._plot_impulse_response(irf, variables)
            
            return var_results
            
        except Exception as e:
            logger.error(f"VAR模型分析失败: {e}")
            return {'error': str(e)}
    
    def _generate_variable_pairs(self) -> List[Tuple[str, str]]:
        """生成所有可能的变量对"""
        numeric_vars = self.panel_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 定义感兴趣的因果关系假设
        hypotheses = [
            # 机制对吸引力的因果关系
            ('mech_app_creation', 'attract_stars_growth'),
            ('mech_code_contrib', 'attract_stars_growth'),
            ('mech_problem_solving', 'attract_stars_growth'),
            ('mech_knowledge_sharing', 'attract_stars_growth'),
            
            # 机制对新仓库的因果关系
            ('mech_app_creation', 'attract_new_app_repo_count'),
            ('mech_code_contrib', 'attract_new_app_repo_count'),
            ('mech_problem_solving', 'attract_new_app_repo_count'),
            
            # 机制对健壮性的因果关系
            ('mech_code_contrib', 'robust_issue_closure_rate'),
            ('mech_problem_solving', 'robust_issue_closure_rate'),
            
            # 创新对机制的因果关系
            ('innovate_topic_diversity', 'mech_app_creation'),
            ('innovate_topic_diversity', 'mech_code_contrib'),
            
            # 反向因果关系
            ('attract_stars_growth', 'mech_app_creation'),
            ('attract_new_app_repo_count', 'mech_code_contrib'),
        ]
        
        # 只保留数据中存在的变量对
        valid_pairs = []
        for cause, effect in hypotheses:
            if cause in numeric_vars and effect in numeric_vars:
                valid_pairs.append((cause, effect))
        
        logger.info(f"生成了 {len(valid_pairs)} 个理论假设驱动的变量对")
        return valid_pairs
    
    def _save_stationarity_results(self):
        """保存平稳性检验结果"""
        if not self.stationarity_results:
            return
        
        stationarity_df = pd.DataFrame.from_dict(self.stationarity_results, orient='index')
        output_path = ANALYSIS_OUTPUT_DIR / "stationarity_test_results.csv"
        stationarity_df.to_csv(output_path, encoding='utf-8-sig')
        logger.info(f"平稳性检验结果已保存至: {output_path}")
    
    def _save_granger_results(self):
        """保存格兰杰因果检验结果"""
        if not self.granger_results:
            return
        
        # 创建摘要表
        summary_data = []
        for pair_key, result in self.granger_results.items():
            if 'error' not in result:
                summary_data.append({
                    'variable_pair': pair_key,
                    'cause_variable': result['cause_variable'],
                    'effect_variable': result['effect_variable'],
                    'best_lag': result['best_lag'],
                    'best_f_pvalue': result['best_f_pvalue'],
                    'is_causal': result['is_causal'],
                    'conclusion': result['conclusion']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('best_f_pvalue')
        
        output_path = ANALYSIS_OUTPUT_DIR / "granger_causality_results.csv"
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"格兰杰因果检验结果已保存至: {output_path}")
        
        # 保存详细结果
        import json
        detailed_path = ANALYSIS_OUTPUT_DIR / "granger_causality_detailed.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(self.granger_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"详细结果已保存至: {detailed_path}")
    
    def _save_var_results(self, var_results: Dict):
        """保存VAR模型结果"""
        import json
        output_path = ANALYSIS_OUTPUT_DIR / "var_model_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(var_results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"VAR模型结果已保存至: {output_path}")
    
    def visualize_granger_results(self):
        """可视化格兰杰因果检验结果"""
        if not self.granger_results:
            logger.warning("没有格兰杰因果检验结果可视化")
            return
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        # 1. 因果关系网络图
        self._plot_causality_network()
        
        # 2. p值分布图
        self._plot_pvalue_distribution()
        
        # 3. 因果关系矩阵热力图
        self._plot_causality_matrix()
    
    def _plot_causality_network(self):
        """绘制因果关系网络图"""
        import networkx as nx
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加显著的因果关系
        for pair_key, result in self.granger_results.items():
            if 'error' not in result and result['is_causal']:
                cause = result['cause_variable']
                effect = result['effect_variable']
                pvalue = result['best_f_pvalue']
                
                G.add_edge(cause, effect, weight=1-pvalue, pvalue=pvalue)
        
        if G.number_of_edges() == 0:
            logger.warning("没有显著的因果关系可绘制")
            return
        
        plt.figure(figsize=(12, 8))
        
        # 计算布局
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.7)
        
        # 绘制边（按权重设置粗细）
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                              alpha=0.6, edge_color='red', 
                              arrowsize=20, arrowstyle='->')
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # 添加边标签（p值）
        edge_labels = {(u, v): f"p={G[u][v]['pvalue']:.3f}" 
                      for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)
        
        plt.title('Granger Causality Network', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.axis('off')
        
        save_path = ANALYSIS_OUTPUT_DIR / "granger_causality_network.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"因果关系网络图已保存至: {save_path}")
        plt.show()
    
    def _plot_pvalue_distribution(self):
        """绘制p值分布图"""
        pvalues = []
        pair_names = []
        
        for pair_key, result in self.granger_results.items():
            if 'error' not in result:
                pvalues.append(result['best_f_pvalue'])
                pair_names.append(pair_key)
        
        if not pvalues:
            return
        
        plt.figure(figsize=(12, 8))
        
        # 创建条形图
        bars = plt.bar(range(len(pvalues)), pvalues)
        
        # 为显著的结果着色
        for i, pval in enumerate(pvalues):
            if pval < self.alpha:
                bars[i].set_color('red')
            else:
                bars[i].set_color('lightblue')
        
        # 添加显著性水平线
        plt.axhline(y=self.alpha, color='red', linestyle='--', 
                   label=f'Significance level (α={self.alpha})')
        
        plt.xlabel('Variable Pairs')
        plt.ylabel('p-value')
        plt.title('Granger Causality Test p-values', 
                 fontsize=VISUALIZATION_CONFIG["title_font_size"])
        plt.xticks(range(len(pair_names)), pair_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = ANALYSIS_OUTPUT_DIR / "granger_pvalues_distribution.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"p值分布图已保存至: {save_path}")
        plt.show()
    
    def _plot_causality_matrix(self):
        """绘制因果关系矩阵热力图"""
        # 获取所有变量
        all_vars = set()
        for result in self.granger_results.values():
            if 'error' not in result:
                all_vars.add(result['cause_variable'])
                all_vars.add(result['effect_variable'])
        
        all_vars = sorted(list(all_vars))
        
        # 创建因果关系矩阵
        causality_matrix = pd.DataFrame(0, index=all_vars, columns=all_vars)
        pvalue_matrix = pd.DataFrame(1.0, index=all_vars, columns=all_vars)
        
        for result in self.granger_results.values():
            if 'error' not in result:
                cause = result['cause_variable']
                effect = result['effect_variable']
                is_causal = result['is_causal']
                pvalue = result['best_f_pvalue']
                
                causality_matrix.loc[cause, effect] = 1 if is_causal else 0
                pvalue_matrix.loc[cause, effect] = pvalue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 因果关系矩阵
        sns.heatmap(causality_matrix, annot=True, cmap='Reds', 
                   square=True, ax=ax1, cbar_kws={'label': 'Causal (1) / Non-causal (0)'})
        ax1.set_title('Granger Causality Matrix')
        ax1.set_xlabel('Effect Variable')
        ax1.set_ylabel('Cause Variable')
        
        # p值矩阵
        sns.heatmap(pvalue_matrix, annot=True, cmap='Blues_r', 
                   square=True, ax=ax2, cbar_kws={'label': 'p-value'})
        ax2.set_title('p-value Matrix')
        ax2.set_xlabel('Effect Variable')
        ax2.set_ylabel('Cause Variable')
        
        plt.tight_layout()
        
        save_path = ANALYSIS_OUTPUT_DIR / "granger_causality_matrices.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"因果关系矩阵图已保存至: {save_path}")
        plt.show()
    
    def _plot_impulse_response(self, irf, variables: List[str]):
        """绘制脉冲响应函数"""
        try:
            fig, axes = plt.subplots(len(variables), len(variables), 
                                   figsize=(4*len(variables), 4*len(variables)))
            
            if len(variables) == 1:
                axes = [[axes]]
            elif len(variables) == 2:
                axes = axes.reshape(2, 2)
            
            for i, shock_var in enumerate(variables):
                for j, response_var in enumerate(variables):
                    ax = axes[i][j]
                    
                    # 绘制脉冲响应
                    irf_data = irf.irfs[:, j, i]  # [periods, response_var, shock_var]
                    periods = range(len(irf_data))
                    
                    ax.plot(periods, irf_data, 'b-', linewidth=2)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    ax.set_title(f'Response of {response_var} to {shock_var}')
                    ax.set_xlabel('Periods')
                    ax.set_ylabel('Response')
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = ANALYSIS_OUTPUT_DIR / "var_impulse_response.png"
            plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
            logger.info(f"脉冲响应函数图已保存至: {save_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制脉冲响应函数失败: {e}")
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        运行完整的格兰杰因果分析
        
        Returns:
            Dict[str, Any]: 完整分析结果
        """
        logger.info("开始完整的格兰杰因果分析...")
        
        # 1. 平稳性检验
        logger.info("1. 执行平稳性检验...")
        stationarity_results = self.check_stationarity()
        
        # 2. 格兰杰因果检验
        logger.info("2. 执行格兰杰因果检验...")
        granger_results = self.perform_granger_causality_test()
        
        # 3. VAR模型分析
        logger.info("3. 构建VAR模型...")
        # 只使用平稳的变量构建VAR模型
        stationary_vars = [var for var, result in stationarity_results.items() 
                          if result.get('is_stationary', False)]
        
        if len(stationary_vars) >= 2:
            var_results = self.analyze_var_model(stationary_vars)
        else:
            logger.warning(f"平稳变量数量不足 ({len(stationary_vars)})，跳过VAR模型分析")
            var_results = {}
        
        # 4. 可视化结果
        logger.info("4. 生成可视化结果...")
        self.visualize_granger_results()
        
        # 5. 汇总分析结果
        complete_results = {
            'stationarity_results': stationarity_results,
            'granger_results': granger_results,
            'var_results': var_results,
            'summary': {
                'total_variables': len(self.panel_data.columns),
                'stationary_variables': len([v for v, r in stationarity_results.items() 
                                           if r.get('is_stationary', False)]),
                'significant_causal_relationships': len([r for r in granger_results.values() 
                                                       if r.get('is_causal', False)]),
                'total_tests': len(granger_results)
            }
        }
        
        logger.info("格兰杰因果分析完成！")
        logger.info(f"发现 {complete_results['summary']['significant_causal_relationships']} 个显著的因果关系")
        
        return complete_results


def main():
    """主函数入口"""
    # 加载面板数据
    panel_data_path = ANALYSIS_OUTPUT_DIR / "monthly_panel_data.csv"
    
    try:
        panel_data = pd.read_csv(panel_data_path, index_col=0, parse_dates=True)
        logger.info(f"成功加载面板数据: {len(panel_data)} 个时间点")
    except FileNotFoundError:
        logger.error(f"未找到面板数据文件: {panel_data_path}")
        return
    
    # 创建格兰杰因果分析器
    analyzer = GrangerCausalityAnalyzer(panel_data, max_lags=3)
    
    # 运行完整分析
    results = analyzer.run_complete_analysis()
    
    # 打印摘要
    summary = results['summary']
    print(f"\n=== 格兰杰因果分析摘要 ===")
    print(f"总变量数: {summary['total_variables']}")
    print(f"平稳变量数: {summary['stationary_variables']}")
    print(f"测试的变量对数: {summary['total_tests']}")
    print(f"显著的因果关系数: {summary['significant_causal_relationships']}")


if __name__ == "__main__":
    main()
