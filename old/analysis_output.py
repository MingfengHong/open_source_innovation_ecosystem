import pandas as pd
import os
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================
ANALYSIS_OUTPUT_DIR = "analysis_output"
PANEL_DATA_FILENAME = "monthly_panel_data.csv"


# ==============================================================================
# --- 2. 辅助函数 (保持不变) ---
# ==============================================================================
def run_regression(df, y_var, x_vars, model_name=""):
    Y = df[y_var]
    X = df[x_vars]
    # --- 修改：此处不再需要 add_constant，因为我们的数据没有明显的截距需求 ---
    # X = sm.add_constant(X)

    combined = pd.concat([Y, X], axis=1).dropna()
    Y = combined[y_var]
    X = combined[x_vars]

    model = sm.OLS(Y, X).fit()

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
    print("--- RQ3 分析: 统计建模与假设检验 (V2 - 活跃度版) ---")
    panel_path = os.path.join(ANALYSIS_OUTPUT_DIR, PANEL_DATA_FILENAME)
    panel_df = pd.read_csv(panel_path, index_col=0, parse_dates=True)

    print("[*] 成功加载月度面板数据。")

    mechanism_vars = [col for col in panel_df.columns if col.startswith('mech_')]
    for var in mechanism_vars:
        panel_df[f'{var}_lag1'] = panel_df[var].shift(1)

    print("[*] 已为所有机制变量创建滞后1个月的变量。")

    # --- 3a. 吸引力模型 (保持不变) ---
    model_attract = run_regression(
        df=panel_df,
        y_var='attract_stars_growth',
        x_vars=['mech_app_creation_lag1', 'mech_knowledge_sharing_lag1', 'mech_problem_solving_lag1'],
        model_name="模型 3a: 检验生态吸引力 (Attractiveness)"
    )

    # --- 3b. 修改：活跃度模型 ---
    model_activeness = run_regression(
        df=panel_df,
        y_var='activeness_total_events',  # 新的因变量
        x_vars=[
            'mech_problem_solving_lag1',  # 主要检验的自变量
            'mech_app_creation_lag1',  # 控制变量
            'mech_knowledge_sharing_lag1'
        ],
        model_name="模型 3b (新): 检验生态活跃度 (Activeness)"
    )

    # --- 3c. 创新性模型 (保持不变) ---
    model_innovate = run_regression(
        df=panel_df,
        y_var='innovate_topic_diversity',
        x_vars=['mech_knowledge_sharing_lag1', 'mech_app_creation_lag1', 'mech_code_contrib_lag1'],
        model_name="模型 3c: 检验生态创新性 (Innovativeness)"
    )

    # --- 4. 可视化关键关系 (更新图表) ---
    print("\n[*] 正在可视化关键关系...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Key Relationships between Mechanisms and Ecosystem Health', fontsize=18)

    # 吸引力图 (不变)
    sns.regplot(x='mech_app_creation_lag1', y='attract_stars_growth', data=panel_df, ax=axes[0],
                line_kws={"color": "red"})
    axes[0].set_title('App Creation vs. Star Growth')

    # 修改：活跃度图
    sns.regplot(x='mech_problem_solving_lag1', y='activeness_total_events', data=panel_df, ax=axes[1],
                line_kws={"color": "red"})
    axes[1].set_title('Problem Solving vs. Total Activeness')

    # 创新性图 (不变)
    sns.regplot(x='mech_knowledge_sharing_lag1', y='innovate_topic_diversity', data=panel_df, ax=axes[2],
                line_kws={"color": "red"})
    axes[2].set_title('Knowledge Sharing vs. Topic Diversity')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, "rq3_key_relationships_v2.png"), dpi=300)
    print(f"[✓] 关键关系图已保存。")
    plt.show()


if __name__ == "__main__":
    main()
