#!/usr/bin/env python
"""
固定效应模型和工具变量分析演示脚本
展示如何使用新实现的面板数据模型
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

# 导入分析模块
from src.causal_analysis.panel_data_builder import PanelDataBuilder
from src.causal_analysis.fixed_effects import FixedEffectsAnalyzer, InstrumentalVariablesAnalyzer, PanelModelComparison
from src.utils.logging_config import setup_logger
from config.settings import ANALYSIS_OUTPUT_DIR

# 设置日志
logger = setup_logger(__name__)


def demonstrate_panel_data_construction():
    """演示面板数据构建"""
    print("=" * 60)
    print("1. 面板数据构建演示")
    print("=" * 60)
    
    # 创建面板数据构建器
    builder = PanelDataBuilder(frequency='M')
    
    # 构建生态系统层面面板数据
    ecosystem_panel = builder.build_ecosystem_panel()
    
    if ecosystem_panel.empty:
        print("❌ 面板数据构建失败")
        return None
    
    print(f"✅ 面板数据构建成功!")
    print(f"   - 实体数: {len(ecosystem_panel.index.get_level_values(0).unique())}")
    print(f"   - 时间段数: {len(ecosystem_panel.index.get_level_values(1).unique())}")
    print(f"   - 总观测值: {len(ecosystem_panel)}")
    print(f"   - 变量数: {len(ecosystem_panel.columns)}")
    
    # 保存面板数据
    builder.save_panel_data(ecosystem_panel, "demo_ecosystem_panel.csv", "演示用生态系统")
    
    # 显示面板数据结构
    print("\n📊 面板数据结构预览:")
    print(ecosystem_panel.head(10))
    
    return ecosystem_panel


def demonstrate_fixed_effects_analysis(panel_data):
    """演示固定效应分析"""
    print("\n" + "=" * 60)
    print("2. 固定效应模型分析演示")
    print("=" * 60)
    
    # 创建固定效应分析器
    fe_analyzer = FixedEffectsAnalyzer(panel_data)
    
    # 定义分析变量
    dependent_var = 'attract_stars_growth'
    independent_vars = ['mech_app_creation', 'mech_code_contrib', 'mech_problem_solving']
    
    print(f"📈 分析设置:")
    print(f"   - 因变量: {dependent_var}")
    print(f"   - 自变量: {', '.join(independent_vars)}")
    
    # 1. 混合OLS估计
    print(f"\n🔍 1. 混合OLS估计...")
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    ols_results = fe_analyzer.estimate_pooled_ols(formula)
    
    if ols_results is not None:
        print(f"   ✅ OLS估计完成: R² = {ols_results.rsquared:.4f}")
    else:
        print(f"   ❌ OLS估计失败")
    
    # 2. 固定效应估计
    print(f"\n🏛️  2. 固定效应模型估计...")
    fe_results = fe_analyzer.estimate_fixed_effects(
        dependent_var, independent_vars, 
        entity_effects=True, time_effects=True
    )
    
    if fe_results is not None:
        print(f"   ✅ 固定效应估计完成:")
        print(f"      - R² (overall): {fe_results.rsquared:.4f}")
        print(f"      - R² (within): {fe_results.rsquared_within:.4f}")
        print(f"      - R² (between): {fe_results.rsquared_between:.4f}")
        print(f"      - F统计量: {fe_results.f_statistic.stat:.4f} (p={fe_results.f_statistic.pval:.4f})")
    else:
        print(f"   ❌ 固定效应估计失败")
    
    # 3. 随机效应估计
    print(f"\n🎲 3. 随机效应模型估计...")
    re_results = fe_analyzer.estimate_random_effects(dependent_var, independent_vars)
    
    if re_results is not None:
        print(f"   ✅ 随机效应估计完成: R² = {re_results.rsquared:.4f}")
    else:
        print(f"   ❌ 随机效应估计失败")
    
    # 4. Hausman检验
    print(f"\n⚖️  4. Hausman检验...")
    hausman_results = fe_analyzer.hausman_test(dependent_var, independent_vars)
    
    if 'error' not in hausman_results:
        print(f"   ✅ Hausman检验完成:")
        print(f"      - 检验统计量: {hausman_results['hausman_statistic']:.4f}")
        print(f"      - p值: {hausman_results['p_value']:.4f}")
        print(f"      - 推荐模型: {hausman_results['recommendation']}")
        
        if hausman_results['reject_re']:
            print(f"      - 结论: 拒绝随机效应，使用固定效应模型")
        else:
            print(f"      - 结论: 不拒绝随机效应，可使用随机效应模型")
    else:
        print(f"   ❌ Hausman检验失败: {hausman_results['error']}")
    
    return fe_analyzer


def demonstrate_instrumental_variables_analysis(panel_data):
    """演示工具变量分析"""
    print("\n" + "=" * 60)
    print("3. 工具变量分析演示")
    print("=" * 60)
    
    # 创建工具变量分析器
    iv_analyzer = InstrumentalVariablesAnalyzer(panel_data.reset_index())
    
    # 1. 识别工具变量
    print(f"🔍 1. 识别工具变量...")
    endogenous_var = 'mech_app_creation'
    print(f"   - 内生变量: {endogenous_var}")
    
    iv_results = iv_analyzer.identify_instruments(endogenous_var)
    
    print(f"   📋 潜在工具变量数: {len(iv_results['potential_instruments'])}")
    print(f"   ✅ 有效工具变量数: {len(iv_results['valid_instruments'])}")
    
    if iv_results['valid_instruments']:
        print(f"   🎯 推荐工具变量: {iv_results['recommended_instrument']}")
        print(f"   📝 有效工具变量列表:")
        for iv in iv_results['valid_instruments'][:5]:  # 显示前5个
            eval_result = iv_results['instrument_evaluation'][iv]
            print(f"      - {iv}: 质量评分={eval_result.get('quality_score', 0)}, "
                  f"相关性={eval_result.get('correlation', 0):.3f}")
    else:
        print(f"   ⚠️  未找到有效的工具变量")
        return None
    
    # 2. 2SLS估计
    print(f"\n🔄 2. 两阶段最小二乘法(2SLS)估计...")
    
    dependent_var = 'attract_stars_growth'
    instruments = iv_results['valid_instruments'][:2]  # 使用前2个最佳工具变量
    exogenous_vars = ['mech_code_contrib', 'mech_problem_solving']
    
    print(f"   - 因变量: {dependent_var}")
    print(f"   - 内生变量: {endogenous_var}")
    print(f"   - 工具变量: {', '.join(instruments)}")
    print(f"   - 外生变量: {', '.join(exogenous_vars)}")
    
    sls_results = iv_analyzer.estimate_2sls(
        dependent_var, [endogenous_var], instruments, exogenous_vars
    )
    
    if sls_results is not None:
        print(f"   ✅ 2SLS估计完成:")
        print(f"      - R² = {sls_results.rsquared:.4f}")
        print(f"      - 观测值数: {sls_results.nobs}")
        
        # 显示系数
        print(f"   📊 主要系数:")
        for var in [endogenous_var] + exogenous_vars:
            if var in sls_results.params.index:
                coef = sls_results.params[var]
                se = sls_results.std_errors[var] if hasattr(sls_results, 'std_errors') else np.nan
                print(f"      - {var}: {coef:.4f} (SE: {se:.4f})")
    else:
        print(f"   ❌ 2SLS估计失败")
        return None
    
    # 3. GMM估计
    print(f"\n📈 3. 广义矩估计(GMM)...")
    
    gmm_results = iv_analyzer.estimate_gmm(
        dependent_var, [endogenous_var], instruments, exogenous_vars
    )
    
    if gmm_results is not None:
        print(f"   ✅ GMM估计完成:")
        print(f"      - R² = {gmm_results.rsquared:.4f}")
        print(f"      - 观测值数: {gmm_results.nobs}")
    else:
        print(f"   ❌ GMM估计失败")
    
    return iv_analyzer


def demonstrate_model_comparison(panel_data):
    """演示模型比较分析"""
    print("\n" + "=" * 60)
    print("4. 面板数据模型比较演示")
    print("=" * 60)
    
    # 创建模型比较分析器
    comparator = PanelModelComparison(panel_data)
    
    # 定义分析变量
    dependent_var = 'attract_stars_growth'
    independent_vars = ['mech_app_creation', 'mech_code_contrib', 'mech_problem_solving']
    
    print(f"🔍 比较分析设置:")
    print(f"   - 因变量: {dependent_var}")
    print(f"   - 自变量: {', '.join(independent_vars)}")
    
    # 运行模型比较
    comparison_results = comparator.compare_all_models(dependent_var, independent_vars)
    
    print(f"\n📊 模型比较结果:")
    
    # 显示各模型的主要指标
    models_info = []
    for model_name, results in comparison_results.items():
        if model_name == 'hausman_test':
            continue
        
        if 'model' in results:
            models_info.append({
                'model': model_name,
                'r_squared': results.get('r_squared', np.nan),
                'n_obs': results.get('n_obs', np.nan)
            })
    
    if models_info:
        comparison_df = pd.DataFrame(models_info)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # 找出最佳模型（按R²）
        best_model = comparison_df.loc[comparison_df['r_squared'].idxmax(), 'model']
        best_r2 = comparison_df.loc[comparison_df['r_squared'].idxmax(), 'r_squared']
        print(f"\n🏆 最佳模型 (按R²): {best_model} (R² = {best_r2:.4f})")
    
    # Hausman检验结果
    if 'hausman_test' in comparison_results:
        hausman = comparison_results['hausman_test']
        if 'error' not in hausman:
            print(f"\n⚖️  Hausman检验推荐: {hausman.get('recommendation', 'unknown')}")
    
    # 可视化比较结果
    print(f"\n📈 生成模型比较图表...")
    try:
        comparator.visualize_model_comparison()
        print(f"   ✅ 图表已保存")
    except Exception as e:
        print(f"   ❌ 图表生成失败: {e}")
    
    return comparison_results


def main():
    """主函数"""
    print("🚀 固定效应模型和工具变量分析演示")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # 1. 面板数据构建
        panel_data = demonstrate_panel_data_construction()
        if panel_data is None:
            print("❌ 演示失败：无法构建面板数据")
            return
        
        # 2. 固定效应分析
        fe_analyzer = demonstrate_fixed_effects_analysis(panel_data)
        
        # 3. 工具变量分析
        iv_analyzer = demonstrate_instrumental_variables_analysis(panel_data)
        
        # 4. 模型比较
        comparison_results = demonstrate_model_comparison(panel_data)
        
        # 5. 总结
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print("✅ 演示完成!")
        print(f"   - 总耗时: {duration}")
        print(f"   - 面板数据: {len(panel_data)} 个观测值")
        print(f"   - 模型数量: {len(comparison_results)} 个")
        print(f"   - 结果文件保存在: {ANALYSIS_OUTPUT_DIR}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        logger.error(f"演示失败: {e}")


if __name__ == "__main__":
    main()
