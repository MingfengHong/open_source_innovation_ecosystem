import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录
ANALYSIS_OUTPUT_DIR = "analysis_output"
# 输入文件名
PANEL_DATA_FILENAME = "monthly_panel_data.csv"


# ==============================================================================
# --- 2. 辅助函数 ---
# ==============================================================================

def run_regression(df, y_var, x_vars, model_name=""):
    """
    一个辅助函数，用于运行OLS回归并打印格式化的结果。
    """
    # 准备数据
    Y = df[y_var]
    X = df[x_vars]
    X = sm.add_constant(X)  # 添加截距项

    # 移除因滞后产生的NaN行
    combined = pd.concat([Y, X], axis=1).dropna()
    Y = combined[y_var]
    X = combined[x_vars]

    # 拟合模型
    model = sm.OLS(Y, X).fit()

    # 打印结果
    print("\n" + "=" * 80)
    print(f"--- {model_name} ---")
    print(f"因变量 (Y): {y_var}")
    print(f"自变量 (X): {', '.join(x_vars)}")
    print("=" * 80)
    print(model.summary())
    print("=" * 80 + "\n")
    return model


# ==============================================================================
# --- 3. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责运行最终的统计分析。
    """
    print("--- RQ3 分析: 步骤 2 - 统计建模与假设检验 ---")

    # --- 1. 加载和准备数据 ---
    panel_path = os.path.join(ANALYSIS_OUTPUT_DIR, PANEL_DATA_FILENAME)
    panel_df = pd.read_csv(panel_path, index_col=0, parse_dates=True)

    print("[*] 成功加载月度面板数据。")

    # --- 2. 创建滞后变量 (Lagged Variables) ---
    # 我们将所有机制变量(X)都创建一个滞后1个月的版本
    mechanism_vars = [col for col in panel_df.columns if col.startswith('mech_')]
    for var in mechanism_vars:
        panel_df[f'{var}_lag1'] = panel_df[var].shift(1)

    print("[*] 已为所有机制变量创建滞后1个月的变量。")

    # --- 3. 运行回归模型以检验假设 ---

    # 3a. “应用驱动”路径是否与“吸引力”强相关？
    model_attract = run_regression(
        df=panel_df,
        y_var='attract_stars_growth',
        x_vars=[
            'mech_app_creation_lag1',
            'mech_knowledge_sharing_lag1',  # 加入其他机制作为控制变量
            'mech_problem_solving_lag1'
        ],
        model_name="模型 3a: 检验生态吸引力 (Attractiveness)"
    )

    # 3b. “代码共创”与“问题解决”路径是否与“健壮性”强相关？
    model_robust = run_regression(
        df=panel_df,
        y_var='robust_issue_closure_rate',
        x_vars=[
            'mech_code_contrib_lag1',
            'mech_problem_solving_lag1',
            'mech_knowledge_sharing_lag1'
        ],
        model_name="模型 3b: 检验生态健壮性 (Robustness)"
    )

    # 3c. “知识共享”路径是否是“创新性”的关键驱动力？
    model_innovate = run_regression(
        df=panel_df,
        y_var='innovate_topic_diversity',
        x_vars=[
            'mech_knowledge_sharing_lag1',
            'mech_app_creation_lag1',
            'mech_code_contrib_lag1'
        ],
        model_name="模型 3c: 检验生态创新性 (Innovativeness)"
    )

    # --- 4. 可视化关键关系 ---
    print("\n[*] 正在可视化关键关系...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Key Relationships between Mechanisms and Ecosystem Health', fontsize=18)

    # 吸引力图
    sns.regplot(x='mech_app_creation_lag1', y='attract_stars_growth', data=panel_df, ax=axes[0],
                line_kws={"color": "red"})
    axes[0].set_title('App Creation vs. Star Growth')

    # 健壮性图
    sns.regplot(x='mech_problem_solving_lag1', y='robust_issue_closure_rate', data=panel_df, ax=axes[1],
                line_kws={"color": "red"})
    axes[1].set_title('Problem Solving vs. Issue Closure Rate')

    # 创新性图
    sns.regplot(x='mech_knowledge_sharing_lag1', y='innovate_topic_diversity', data=panel_df, ax=axes[2],
                line_kws={"color": "red"})
    axes[2].set_title('Knowledge Sharing vs. Topic Diversity')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "rq3_key_relationships.png"), dpi=300)
    print(f"[✓] 关键关系图已保存。")
    plt.show()


if __name__ == "__main__":
    main()