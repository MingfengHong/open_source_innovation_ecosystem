"""
固定效应模型和工具变量分析模块
实现面板数据的固定效应估计和工具变量方法，用于处理内生性问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# 统计分析工具
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels import PanelOLS, IV2SLS, IVGMM
from linearmodels.iv import _IVGMMResults
import scipy.stats as stats

from ..utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR, VISUALIZATION_CONFIG

# 设置日志
logger = setup_logger(__name__)


class FixedEffectsAnalyzer:
    """固定效应模型分析器"""
    
    def __init__(self, 
                 panel_data: pd.DataFrame,
                 entity_col: str = 'entity_id',
                 time_col: str = 'time_period',
                 dependent_var: str = None):
        """
        初始化固定效应分析器
        
        Args:
            panel_data: 面板数据
            entity_col: 实体标识列（如用户ID、仓库ID）
            time_col: 时间标识列
            dependent_var: 因变量名称
        """
        self.panel_data = panel_data.copy()
        self.entity_col = entity_col
        self.time_col = time_col
        self.dependent_var = dependent_var
        
        # 设置多重索引
        if entity_col in self.panel_data.columns and time_col in self.panel_data.columns:
            self.panel_data = self.panel_data.set_index([entity_col, time_col])
        
        # 存储模型结果
        self.model_results = {}
        self.diagnostic_results = {}
        
        logger.info(f"初始化固定效应分析器: {len(self.panel_data)} 个观测值")
        logger.info(f"面板结构: {len(self.panel_data.index.get_level_values(0).unique())} 个实体, "
                   f"{len(self.panel_data.index.get_level_values(1).unique())} 个时间段")
    
    def prepare_monthly_panel_data(self, monthly_data: pd.DataFrame) -> pd.DataFrame:
        """
        将月度数据转换为面板数据格式
        
        Args:
            monthly_data: 月度时间序列数据
            
        Returns:
            pd.DataFrame: 面板数据格式
        """
        logger.info("将月度数据转换为面板数据格式...")
        
        # 创建虚拟实体（由于我们只有一个生态系统，创建多个观测角度）
        panel_rows = []
        
        for idx, row in monthly_data.iterrows():
            # 为每个月创建多个观测单位
            # 实体1: 整体生态系统
            panel_rows.append({
                'entity_id': 'ecosystem_overall',
                'time_period': idx,
                **row.to_dict()
            })
            
            # 实体2: 核心贡献视角（强调代码贡献）
            core_contrib_data = row.copy()
            # 对代码相关指标给予更高权重
            if 'mech_code_contrib' in core_contrib_data:
                core_contrib_data['weighted_contrib'] = core_contrib_data['mech_code_contrib'] * 1.5
            panel_rows.append({
                'entity_id': 'ecosystem_core_contrib',
                'time_period': idx,
                **core_contrib_data.to_dict()
            })
            
            # 实体3: 应用开发视角（强调应用创建）
            app_dev_data = row.copy()
            if 'mech_app_creation' in app_dev_data:
                app_dev_data['weighted_app'] = app_dev_data['mech_app_creation'] * 1.5
            panel_rows.append({
                'entity_id': 'ecosystem_app_dev',
                'time_period': idx,
                **app_dev_data.to_dict()
            })
            
            # 实体4: 社区维护视角（强调问题解决）
            maintenance_data = row.copy()
            if 'mech_problem_solving' in maintenance_data:
                maintenance_data['weighted_maintenance'] = maintenance_data['mech_problem_solving'] * 1.5
            panel_rows.append({
                'entity_id': 'ecosystem_maintenance',
                'time_period': idx,
                **maintenance_data.to_dict()
            })
        
        panel_df = pd.DataFrame(panel_rows)
        panel_df = panel_df.set_index(['entity_id', 'time_period'])
        
        logger.info(f"面板数据创建完成: {len(panel_df)} 个观测值")
        return panel_df
    
    def estimate_pooled_ols(self, 
                           formula: str,
                           data: pd.DataFrame = None) -> sm.regression.linear_model.RegressionResults:
        """
        估计混合OLS模型（基准模型）
        
        Args:
            formula: 回归公式
            data: 数据，默认使用实例数据
            
        Returns:
            sm.regression.linear_model.RegressionResults: OLS结果
        """
        if data is None:
            data = self.panel_data.reset_index()
        
        logger.info(f"估计混合OLS模型: {formula}")
        
        try:
            model = ols(formula, data=data)
            results = model.fit()
            
            self.model_results['pooled_ols'] = results
            
            # 模型诊断
            self._run_ols_diagnostics(results, data, formula)
            
            return results
            
        except Exception as e:
            logger.error(f"混合OLS估计失败: {e}")
            return None
    
    def estimate_fixed_effects(self, 
                              dependent_var: str,
                              independent_vars: List[str],
                              entity_effects: bool = True,
                              time_effects: bool = True,
                              cluster_entity: bool = True) -> Any:
        """
        估计固定效应模型
        
        Args:
            dependent_var: 因变量
            independent_vars: 自变量列表
            entity_effects: 是否包含实体固定效应
            time_effects: 是否包含时间固定效应
            cluster_entity: 是否在实体层面聚类标准误
            
        Returns:
            PanelOLS结果对象
        """
        logger.info(f"估计固定效应模型: {dependent_var} ~ {' + '.join(independent_vars)}")
        
        try:
            # 准备数据
            model_data = self.panel_data[[dependent_var] + independent_vars].dropna()
            
            if len(model_data) == 0:
                logger.error("模型数据为空")
                return None
            
            # 构建模型
            dependent = model_data[dependent_var]
            exog = sm.add_constant(model_data[independent_vars])
            
            # 创建固定效应模型
            model = PanelOLS(
                dependent=dependent,
                exog=exog,
                entity_effects=entity_effects,
                time_effects=time_effects
            )
            
            # 估计模型
            if cluster_entity:
                results = model.fit(cov_type='clustered', cluster_entity=True)
            else:
                results = model.fit()
            
            self.model_results['fixed_effects'] = results
            
            # 保存模型信息
            model_info = {
                'dependent_var': dependent_var,
                'independent_vars': independent_vars,
                'entity_effects': entity_effects,
                'time_effects': time_effects,
                'cluster_entity': cluster_entity,
                'n_obs': results.nobs,
                'n_entities': len(model_data.index.get_level_values(0).unique()),
                'n_time_periods': len(model_data.index.get_level_values(1).unique()),
                'r_squared': results.rsquared,
                'r_squared_within': results.rsquared_within,
                'r_squared_between': results.rsquared_between,
                'f_statistic': results.f_statistic.stat,
                'f_pvalue': results.f_statistic.pval
            }
            
            self.diagnostic_results['fixed_effects_info'] = model_info
            
            return results
            
        except Exception as e:
            logger.error(f"固定效应模型估计失败: {e}")
            return None
    
    def estimate_random_effects(self, 
                               dependent_var: str,
                               independent_vars: List[str]) -> Any:
        """
        估计随机效应模型
        
        Args:
            dependent_var: 因变量
            independent_vars: 自变量列表
            
        Returns:
            随机效应模型结果
        """
        logger.info(f"估计随机效应模型: {dependent_var} ~ {' + '.join(independent_vars)}")
        
        try:
            from linearmodels import RandomEffects
            
            # 准备数据
            model_data = self.panel_data[[dependent_var] + independent_vars].dropna()
            
            dependent = model_data[dependent_var]
            exog = sm.add_constant(model_data[independent_vars])
            
            # 创建随机效应模型
            model = RandomEffects(dependent=dependent, exog=exog)
            results = model.fit()
            
            self.model_results['random_effects'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"随机效应模型估计失败: {e}")
            return None
    
    def hausman_test(self, 
                    dependent_var: str,
                    independent_vars: List[str]) -> Dict[str, Any]:
        """
        Hausman检验：选择固定效应还是随机效应
        
        Args:
            dependent_var: 因变量
            independent_vars: 自变量列表
            
        Returns:
            Dict[str, Any]: Hausman检验结果
        """
        logger.info("执行Hausman检验...")
        
        try:
            # 估计固定效应和随机效应模型
            fe_results = self.estimate_fixed_effects(dependent_var, independent_vars)
            re_results = self.estimate_random_effects(dependent_var, independent_vars)
            
            if fe_results is None or re_results is None:
                logger.error("无法完成Hausman检验：模型估计失败")
                return {}
            
            # 提取系数
            fe_coef = fe_results.params[independent_vars]
            re_coef = re_results.params[independent_vars]
            
            # 提取协方差矩阵
            fe_cov = fe_results.cov[independent_vars].loc[independent_vars]
            re_cov = re_results.cov[independent_vars].loc[independent_vars]
            
            # 计算Hausman统计量
            coef_diff = fe_coef - re_coef
            cov_diff = fe_cov - re_cov
            
            # 确保协方差矩阵可逆
            try:
                cov_diff_inv = np.linalg.inv(cov_diff)
                hausman_stat = coef_diff.T @ cov_diff_inv @ coef_diff
                
                # 自由度
                df = len(independent_vars)
                
                # p值
                p_value = 1 - stats.chi2.cdf(hausman_stat, df)
                
                hausman_results = {
                    'hausman_statistic': hausman_stat,
                    'degrees_of_freedom': df,
                    'p_value': p_value,
                    'critical_value_5pct': stats.chi2.ppf(0.95, df),
                    'reject_re': p_value < 0.05,
                    'recommendation': 'Fixed Effects' if p_value < 0.05 else 'Random Effects'
                }
                
                self.diagnostic_results['hausman_test'] = hausman_results
                
                logger.info(f"Hausman检验完成: 统计量={hausman_stat:.4f}, p值={p_value:.4f}")
                logger.info(f"推荐模型: {hausman_results['recommendation']}")
                
                return hausman_results
                
            except np.linalg.LinAlgError:
                logger.warning("协方差矩阵不可逆，无法计算Hausman统计量")
                return {'error': 'Covariance matrix is not invertible'}
                
        except Exception as e:
            logger.error(f"Hausman检验失败: {e}")
            return {'error': str(e)}
    
    def _run_ols_diagnostics(self, 
                            results: sm.regression.linear_model.RegressionResults,
                            data: pd.DataFrame,
                            formula: str):
        """运行OLS模型诊断"""
        logger.info("运行OLS模型诊断...")
        
        diagnostics = {}
        
        try:
            # 1. 异方差检验
            # White检验
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(results.resid, results.model.exog)
            diagnostics['white_test'] = {
                'lm_statistic': lm_stat,
                'lm_pvalue': lm_pvalue,
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'heteroskedasticity': lm_pvalue < 0.05
            }
            
            # Breusch-Pagan检验
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(results.resid, results.model.exog)
            diagnostics['breusch_pagan_test'] = {
                'statistic': bp_stat,
                'pvalue': bp_pvalue,
                'heteroskedasticity': bp_pvalue < 0.05
            }
            
            # 2. 自相关检验
            dw_stat = durbin_watson(results.resid)
            diagnostics['durbin_watson'] = {
                'statistic': dw_stat,
                'autocorrelation': dw_stat < 1.5 or dw_stat > 2.5
            }
            
            # 3. 多重共线性检验（VIF）
            if len(results.model.exog_names) > 1:
                vif_data = pd.DataFrame()
                vif_data["Variable"] = results.model.exog_names
                vif_data["VIF"] = [variance_inflation_factor(results.model.exog, i) 
                                 for i in range(len(results.model.exog_names))]
                
                diagnostics['vif'] = {
                    'vif_values': vif_data.to_dict('records'),
                    'high_multicollinearity': (vif_data["VIF"] > 10).any()
                }
            
            # 4. 正态性检验（Shapiro-Wilk）
            shapiro_stat, shapiro_pvalue = stats.shapiro(results.resid)
            diagnostics['shapiro_test'] = {
                'statistic': shapiro_stat,
                'pvalue': shapiro_pvalue,
                'non_normal': shapiro_pvalue < 0.05
            }
            
            self.diagnostic_results['ols_diagnostics'] = diagnostics
            
        except Exception as e:
            logger.error(f"OLS诊断失败: {e}")


class InstrumentalVariablesAnalyzer:
    """工具变量分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化工具变量分析器
        
        Args:
            data: 分析数据
        """
        self.data = data.copy()
        self.iv_results = {}
        
        logger.info(f"初始化工具变量分析器: {len(data)} 个观测值")
    
    def identify_instruments(self, 
                           endogenous_var: str,
                           potential_instruments: List[str] = None) -> Dict[str, Any]:
        """
        识别和评估工具变量
        
        Args:
            endogenous_var: 内生变量
            potential_instruments: 潜在工具变量列表
            
        Returns:
            Dict[str, Any]: 工具变量评估结果
        """
        logger.info(f"为内生变量 {endogenous_var} 识别工具变量...")
        
        if potential_instruments is None:
            # 根据生态系统特点定义潜在工具变量
            potential_instruments = self._generate_potential_instruments()
        
        instrument_evaluation = {}
        
        for instrument in potential_instruments:
            if instrument not in self.data.columns:
                continue
            
            # 评估工具变量质量
            evaluation = self._evaluate_instrument(endogenous_var, instrument)
            instrument_evaluation[instrument] = evaluation
        
        # 选择最佳工具变量
        valid_instruments = [iv for iv, eval_result in instrument_evaluation.items() 
                           if eval_result.get('is_valid', False)]
        
        results = {
            'endogenous_var': endogenous_var,
            'potential_instruments': potential_instruments,
            'instrument_evaluation': instrument_evaluation,
            'valid_instruments': valid_instruments,
            'recommended_instrument': self._select_best_instrument(instrument_evaluation)
        }
        
        logger.info(f"找到 {len(valid_instruments)} 个有效工具变量")
        
        return results
    
    def _generate_potential_instruments(self) -> List[str]:
        """生成潜在工具变量列表"""
        # 根据开源生态系统的特点，定义潜在的外生工具变量
        potential_instruments = []
        
        # 滞后变量作为工具变量
        for col in self.data.columns:
            if any(keyword in col.lower() for keyword in ['mech_', 'attract_', 'robust_', 'innovate_']):
                # 为时间序列数据创建滞后变量
                lag_col = f'{col}_lag1'
                if len(self.data) > 1:
                    self.data[lag_col] = self.data[col].shift(1)
                    potential_instruments.append(lag_col)
                
                # 二阶滞后
                lag2_col = f'{col}_lag2'
                if len(self.data) > 2:
                    self.data[lag2_col] = self.data[col].shift(2)
                    potential_instruments.append(lag2_col)
        
        # 移动平均作为工具变量
        for col in self.data.columns:
            if 'mech_' in col:
                ma_col = f'{col}_ma3'
                self.data[ma_col] = self.data[col].rolling(window=3, min_periods=1).mean()
                potential_instruments.append(ma_col)
        
        # 外生冲击变量（模拟）
        # 在实际应用中，这些应该是真实的外生事件
        self.data['tech_trend_shock'] = np.random.normal(0, 1, len(self.data))  # 技术趋势冲击
        self.data['market_attention_shock'] = np.random.normal(0, 1, len(self.data))  # 市场关注度冲击
        potential_instruments.extend(['tech_trend_shock', 'market_attention_shock'])
        
        return potential_instruments
    
    def _evaluate_instrument(self, endogenous_var: str, instrument: str) -> Dict[str, Any]:
        """评估单个工具变量的有效性"""
        evaluation = {}
        
        try:
            # 准备数据
            eval_data = self.data[[endogenous_var, instrument]].dropna()
            
            if len(eval_data) < 10:
                evaluation['is_valid'] = False
                evaluation['reason'] = 'Insufficient data'
                return evaluation
            
            # 1. 相关性检验（第一阶段）
            correlation = eval_data[endogenous_var].corr(eval_data[instrument])
            evaluation['correlation'] = correlation
            evaluation['strong_correlation'] = abs(correlation) > 0.3  # 通常要求相关性>0.3
            
            # 2. 第一阶段F统计量
            first_stage_ols = sm.OLS(eval_data[endogenous_var], 
                                   sm.add_constant(eval_data[instrument])).fit()
            f_stat = first_stage_ols.fvalue
            evaluation['first_stage_f_stat'] = f_stat
            evaluation['weak_instrument'] = f_stat < 10  # F统计量<10通常被认为是弱工具变量
            
            # 3. 显著性检验
            evaluation['first_stage_pvalue'] = first_stage_ols.pvalues[instrument]
            evaluation['significant'] = evaluation['first_stage_pvalue'] < 0.05
            
            # 4. 综合评估
            evaluation['is_valid'] = (evaluation['strong_correlation'] and 
                                    not evaluation['weak_instrument'] and 
                                    evaluation['significant'])
            
            # 5. 质量评分
            score = 0
            if evaluation['strong_correlation']:
                score += 2
            if not evaluation['weak_instrument']:
                score += 3
            if evaluation['significant']:
                score += 1
            evaluation['quality_score'] = score
            
        except Exception as e:
            evaluation['is_valid'] = False
            evaluation['error'] = str(e)
        
        return evaluation
    
    def _select_best_instrument(self, instrument_evaluation: Dict[str, Dict]) -> Optional[str]:
        """选择最佳工具变量"""
        valid_instruments = {iv: eval_result for iv, eval_result in instrument_evaluation.items() 
                           if eval_result.get('is_valid', False)}
        
        if not valid_instruments:
            return None
        
        # 选择质量评分最高的工具变量
        best_instrument = max(valid_instruments.keys(), 
                            key=lambda x: valid_instruments[x].get('quality_score', 0))
        
        return best_instrument
    
    def estimate_2sls(self, 
                     dependent_var: str,
                     endogenous_vars: List[str],
                     instruments: List[str],
                     exogenous_vars: List[str] = None) -> Any:
        """
        估计两阶段最小二乘法(2SLS)模型
        
        Args:
            dependent_var: 因变量
            endogenous_vars: 内生变量列表
            instruments: 工具变量列表
            exogenous_vars: 外生变量列表
            
        Returns:
            2SLS估计结果
        """
        logger.info(f"估计2SLS模型: {dependent_var} ~ {endogenous_vars} | {instruments}")
        
        try:
            # 准备数据
            all_vars = [dependent_var] + endogenous_vars + instruments
            if exogenous_vars:
                all_vars.extend(exogenous_vars)
            
            model_data = self.data[all_vars].dropna()
            
            if len(model_data) < 10:
                logger.error("2SLS模型数据不足")
                return None
            
            # 构建模型
            dependent = model_data[dependent_var]
            
            # 内生变量
            endog = model_data[endogenous_vars] if len(endogenous_vars) > 1 else model_data[endogenous_vars[0]]
            
            # 工具变量
            instruments_data = model_data[instruments]
            
            # 外生变量
            if exogenous_vars:
                exog = sm.add_constant(model_data[exogenous_vars])
            else:
                exog = None
            
            # 估计2SLS模型
            model = IV2SLS(dependent=dependent, 
                          endog=endog, 
                          instruments=instruments_data,
                          exog=exog)
            
            results = model.fit()
            
            self.iv_results['2sls'] = results
            
            # 运行IV诊断
            self._run_iv_diagnostics(results, model_data, dependent_var, endogenous_vars, instruments)
            
            return results
            
        except Exception as e:
            logger.error(f"2SLS估计失败: {e}")
            return None
    
    def estimate_gmm(self,
                    dependent_var: str,
                    endogenous_vars: List[str],
                    instruments: List[str],
                    exogenous_vars: List[str] = None) -> Any:
        """
        估计广义矩估计(GMM)模型
        
        Args:
            dependent_var: 因变量
            endogenous_vars: 内生变量列表
            instruments: 工具变量列表
            exogenous_vars: 外生变量列表
            
        Returns:
            GMM估计结果
        """
        logger.info(f"估计GMM模型: {dependent_var} ~ {endogenous_vars} | {instruments}")
        
        try:
            # 准备数据
            all_vars = [dependent_var] + endogenous_vars + instruments
            if exogenous_vars:
                all_vars.extend(exogenous_vars)
            
            model_data = self.data[all_vars].dropna()
            
            # 构建模型
            dependent = model_data[dependent_var]
            endog = model_data[endogenous_vars] if len(endogenous_vars) > 1 else model_data[endogenous_vars[0]]
            instruments_data = model_data[instruments]
            
            if exogenous_vars:
                exog = sm.add_constant(model_data[exogenous_vars])
            else:
                exog = None
            
            # 估计GMM模型
            model = IVGMM(dependent=dependent,
                         endog=endog,
                         instruments=instruments_data,
                         exog=exog)
            
            results = model.fit()
            
            self.iv_results['gmm'] = results
            
            return results
            
        except Exception as e:
            logger.error(f"GMM估计失败: {e}")
            return None
    
    def _run_iv_diagnostics(self, 
                           results: Any,
                           data: pd.DataFrame,
                           dependent_var: str,
                           endogenous_vars: List[str],
                           instruments: List[str]):
        """运行工具变量模型诊断"""
        logger.info("运行工具变量模型诊断...")
        
        diagnostics = {}
        
        try:
            # 1. 弱工具变量检验
            first_stage_f_stats = []
            for endog_var in endogenous_vars:
                if endog_var in data.columns:
                    first_stage_ols = sm.OLS(data[endog_var], 
                                           sm.add_constant(data[instruments])).fit()
                    first_stage_f_stats.append(first_stage_ols.fvalue)
            
            diagnostics['weak_instruments_test'] = {
                'first_stage_f_stats': first_stage_f_stats,
                'min_f_stat': min(first_stage_f_stats) if first_stage_f_stats else 0,
                'weak_instruments': min(first_stage_f_stats) < 10 if first_stage_f_stats else True
            }
            
            # 2. 过度识别检验（如果工具变量数量>内生变量数量）
            if len(instruments) > len(endogenous_vars):
                # Sargan检验
                try:
                    sargan_stat = results.sargan.stat
                    sargan_pvalue = results.sargan.pval
                    
                    diagnostics['overidentification_test'] = {
                        'sargan_statistic': sargan_stat,
                        'sargan_pvalue': sargan_pvalue,
                        'overidentified': sargan_pvalue < 0.05
                    }
                except AttributeError:
                    logger.warning("无法获取Sargan检验统计量")
            
            # 3. 内生性检验
            try:
                # 计算Hausman检验统计量（比较OLS和IV结果）
                ols_model = sm.OLS(data[dependent_var], 
                                 sm.add_constant(data[endogenous_vars])).fit()
                
                # 简化的内生性检验
                iv_coef = results.params[endogenous_vars[0]] if len(endogenous_vars) == 1 else results.params[endogenous_vars].iloc[0]
                ols_coef = ols_model.params[endogenous_vars[0]] if len(endogenous_vars) == 1 else ols_model.params[endogenous_vars].iloc[0]
                
                coef_diff = abs(iv_coef - ols_coef)
                
                diagnostics['endogeneity_test'] = {
                    'iv_coefficient': iv_coef,
                    'ols_coefficient': ols_coef,
                    'coefficient_difference': coef_diff,
                    'substantial_difference': coef_diff > 0.1 * abs(ols_coef)
                }
                
            except Exception as e:
                logger.warning(f"内生性检验失败: {e}")
            
            self.iv_results['diagnostics'] = diagnostics
            
        except Exception as e:
            logger.error(f"IV诊断失败: {e}")


class PanelModelComparison:
    """面板模型比较分析器"""
    
    def __init__(self, panel_data: pd.DataFrame):
        """初始化模型比较分析器"""
        self.panel_data = panel_data
        self.comparison_results = {}
        
    def compare_all_models(self, 
                          dependent_var: str,
                          independent_vars: List[str],
                          instruments: List[str] = None) -> Dict[str, Any]:
        """
        比较所有面板数据模型
        
        Args:
            dependent_var: 因变量
            independent_vars: 自变量列表
            instruments: 工具变量列表（用于IV估计）
            
        Returns:
            Dict[str, Any]: 模型比较结果
        """
        logger.info("开始面板数据模型比较分析...")
        
        comparison_results = {}
        
        # 1. 混合OLS
        fe_analyzer = FixedEffectsAnalyzer(self.panel_data)
        
        # 准备公式
        formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
        ols_results = fe_analyzer.estimate_pooled_ols(formula)
        
        if ols_results is not None:
            comparison_results['pooled_ols'] = {
                'model': ols_results,
                'r_squared': ols_results.rsquared,
                'aic': ols_results.aic,
                'bic': ols_results.bic,
                'log_likelihood': ols_results.llf,
                'n_obs': ols_results.nobs
            }
        
        # 2. 固定效应模型
        fe_results = fe_analyzer.estimate_fixed_effects(dependent_var, independent_vars)
        
        if fe_results is not None:
            comparison_results['fixed_effects'] = {
                'model': fe_results,
                'r_squared': fe_results.rsquared,
                'r_squared_within': fe_results.rsquared_within,
                'r_squared_between': fe_results.rsquared_between,
                'n_obs': fe_results.nobs,
                'f_statistic': fe_results.f_statistic.stat,
                'f_pvalue': fe_results.f_statistic.pval
            }
        
        # 3. 随机效应模型
        re_results = fe_analyzer.estimate_random_effects(dependent_var, independent_vars)
        
        if re_results is not None:
            comparison_results['random_effects'] = {
                'model': re_results,
                'r_squared': re_results.rsquared,
                'n_obs': re_results.nobs
            }
        
        # 4. Hausman检验
        hausman_results = fe_analyzer.hausman_test(dependent_var, independent_vars)
        comparison_results['hausman_test'] = hausman_results
        
        # 5. 工具变量估计（如果提供了工具变量）
        if instruments:
            iv_analyzer = InstrumentalVariablesAnalyzer(self.panel_data.reset_index())
            
            # 2SLS估计
            sls_results = iv_analyzer.estimate_2sls(dependent_var, [independent_vars[0]], instruments)
            if sls_results is not None:
                comparison_results['2sls'] = {
                    'model': sls_results,
                    'r_squared': sls_results.rsquared,
                    'n_obs': sls_results.nobs
                }
            
            # GMM估计
            gmm_results = iv_analyzer.estimate_gmm(dependent_var, [independent_vars[0]], instruments)
            if gmm_results is not None:
                comparison_results['gmm'] = {
                    'model': gmm_results,
                    'r_squared': gmm_results.rsquared,
                    'n_obs': gmm_results.nobs
                }
        
        self.comparison_results = comparison_results
        
        # 生成比较摘要
        self._generate_comparison_summary()
        
        return comparison_results
    
    def _generate_comparison_summary(self):
        """生成模型比较摘要"""
        summary_data = []
        
        for model_name, results in self.comparison_results.items():
            if model_name == 'hausman_test':
                continue
            
            if 'model' in results:
                summary_data.append({
                    'model': model_name,
                    'r_squared': results.get('r_squared', np.nan),
                    'n_obs': results.get('n_obs', np.nan),
                    'aic': results.get('aic', np.nan),
                    'bic': results.get('bic', np.nan)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存比较摘要
        output_path = ANALYSIS_OUTPUT_DIR / "panel_models_comparison.csv"
        summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"模型比较摘要已保存至: {output_path}")
        
        self.comparison_summary = summary_df
    
    def visualize_model_comparison(self):
        """可视化模型比较结果"""
        if not hasattr(self, 'comparison_summary'):
            logger.warning("没有比较摘要可可视化")
            return
        
        plt.style.use(VISUALIZATION_CONFIG["style"])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R²比较
        axes[0,0].bar(self.comparison_summary['model'], self.comparison_summary['r_squared'])
        axes[0,0].set_title('R² Comparison Across Models')
        axes[0,0].set_ylabel('R²')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 观测值数量比较
        axes[0,1].bar(self.comparison_summary['model'], self.comparison_summary['n_obs'])
        axes[0,1].set_title('Number of Observations')
        axes[0,1].set_ylabel('N. Obs')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # AIC比较
        aic_data = self.comparison_summary.dropna(subset=['aic'])
        if not aic_data.empty:
            axes[1,0].bar(aic_data['model'], aic_data['aic'])
            axes[1,0].set_title('AIC Comparison (Lower is Better)')
            axes[1,0].set_ylabel('AIC')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # BIC比较
        bic_data = self.comparison_summary.dropna(subset=['bic'])
        if not bic_data.empty:
            axes[1,1].bar(bic_data['model'], bic_data['bic'])
            axes[1,1].set_title('BIC Comparison (Lower is Better)')
            axes[1,1].set_ylabel('BIC')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = ANALYSIS_OUTPUT_DIR / "panel_models_comparison.png"
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches='tight')
        logger.info(f"模型比较图表已保存至: {save_path}")
        plt.show()


def main():
    """主函数入口"""
    # 加载面板数据
    panel_data_path = ANALYSIS_OUTPUT_DIR / "monthly_panel_data.csv"
    
    try:
        monthly_data = pd.read_csv(panel_data_path, index_col=0, parse_dates=True)
        logger.info(f"成功加载月度数据: {len(monthly_data)} 个时间点")
    except FileNotFoundError:
        logger.error(f"未找到面板数据文件: {panel_data_path}")
        return
    
    # 创建固定效应分析器
    fe_analyzer = FixedEffectsAnalyzer(monthly_data)
    
    # 准备面板数据
    panel_data = fe_analyzer.prepare_monthly_panel_data(monthly_data)
    
    # 定义分析变量
    dependent_var = 'attract_stars_growth'
    independent_vars = ['mech_app_creation', 'mech_code_contrib', 'mech_problem_solving']
    
    # 创建模型比较分析器
    comparator = PanelModelComparison(panel_data)
    
    # 运行模型比较
    comparison_results = comparator.compare_all_models(dependent_var, independent_vars)
    
    # 可视化比较结果
    comparator.visualize_model_comparison()
    
    # 创建工具变量分析器
    iv_analyzer = InstrumentalVariablesAnalyzer(panel_data.reset_index())
    
    # 识别工具变量
    iv_results = iv_analyzer.identify_instruments('mech_app_creation')
    
    if iv_results['valid_instruments']:
        logger.info(f"找到有效工具变量: {iv_results['valid_instruments']}")
        
        # 使用工具变量估计
        instruments = iv_results['valid_instruments'][:2]  # 使用前2个最佳工具变量
        sls_results = iv_analyzer.estimate_2sls(dependent_var, ['mech_app_creation'], instruments, 
                                               ['mech_code_contrib', 'mech_problem_solving'])
        
        if sls_results is not None:
            logger.info("2SLS估计完成")
    
    logger.info("固定效应模型和工具变量分析完成！")


if __name__ == "__main__":
    main()
