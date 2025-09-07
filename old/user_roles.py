import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# --- 1. 配置区域 ---
# ==============================================================================

# 输入目录
ANALYSIS_OUTPUT_DIR = "analysis_output"
# 输入文件名
FEATURES_FILENAME = "user_behavior_features.csv"

# 输出文件名
ROLES_FILENAME = "user_roles.csv"
# 图表文件名
ELBOW_PLOT_FILENAME = "kmeans_elbow_plot.png"
PROFILE_HEATMAP_FILENAME = "user_role_profiles_heatmap.png"


# ==============================================================================
# --- 2. 主执行逻辑 ---
# ==============================================================================

def main():
    """
    主函数，负责用户角色的聚类分析和画像。
    """
    print("--- RQ2 分析: 步骤 2 - 用户角色聚类与画像 ---")

    # --- 1. 加载和准备数据 ---
    features_path = os.path.join(ANALYSIS_OUTPUT_DIR, FEATURES_FILENAME)
    print(f"[*] 正在从 '{features_path}' 加载用户行为特征...")
    try:
        features_df = pd.read_csv(features_path)
    except FileNotFoundError as e:
        print(f"[!] 错误: 未找到特征文件: {e}")
        return

    # 将 user_id 和 login 作为索引，方便后续合并
    user_info_df = features_df[['user_id', 'login']].copy()
    # 用于聚类的特征列是除了 user_id 和 login 之外的所有列
    feature_cols = [col for col in features_df.columns if col not in ['user_id', 'login']]
    X = features_df[feature_cols]

    # --- 2. 数据标准化 ---
    # 这是聚类前非常关键的一步
    print("[*] 正在对行为特征进行标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("[✓] 数据标准化完成。")

    # --- 3. 使用“肘部法则”确定最佳K值 ---
    print("[*] 正在使用“肘部法则”来估算最佳聚类数量 (K)...")
    inertia = []
    k_range = range(2, 16)  # 测试K从2到15
    for k in tqdm(k_range, desc="测试不同的K值"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Determining Optimal K')
    plt.xticks(k_range)
    plt.grid(True)

    # 保存图表
    elbow_plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, ELBOW_PLOT_FILENAME)
    plt.savefig(elbow_plot_path, dpi=300)
    print(f"[✓] “肘部法则”图表已保存至: '{elbow_plot_path}'")
    print("[!] 请查看生成的肘部图，找到曲线斜率变化最明显的“肘点”，并将其作为下一步的K值。")
    plt.show()

    # --- 4. 执行K-Means聚类 ---
    # !!! 行动要求: 请根据上一步生成的肘部图，在此处填入您选择的最佳K值 !!!
    # 例如，如果肘点在 K=5 或 K=6 的位置，就填入 5 或 6。
    # 我在这里先预设一个可能的值，您可以根据图表进行修改。
    CHOSEN_K = 6
    print(f"\n[*] 已选择 K={CHOSEN_K}。开始执行最终的K-Means聚类...")

    kmeans = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # 将聚类结果添加回原始数据
    features_df['cluster'] = clusters
    print("[✓] 聚类完成，已为每个用户分配角色标签。")

    # --- 5. 角色画像 (Profiling) ---
    print("\n--- 各用户角色的行为特征画像 ---")
    # 我们按聚类标签进行分组，并计算每个特征的均值
    # 这张表是理解每个角色本质的关键
    profile_df = features_df.groupby('cluster')[feature_cols].mean()

    # 同时计算每个角色的人数
    profile_df['population'] = features_df['cluster'].value_counts()

    print(profile_df)

    # --- 6. 可视化角色画像 ---
    print("\n[*] 正在生成角色画像热力图...")
    # 为了在热力图上更好地比较，我们对画像的均值再次进行标准化
    profile_scaled = scaler.fit_transform(profile_df[feature_cols])
    profile_scaled_df = pd.DataFrame(profile_scaled, index=profile_df.index, columns=feature_cols)

    plt.figure(figsize=(16, 8))
    sns.heatmap(profile_scaled_df, cmap='viridis', annot=True, fmt='.2f')
    plt.title('Heatmap of User Role Profiles (Standardized Mean Values)')
    plt.xlabel('Behavioral Features')
    plt.ylabel('User Role (Cluster ID)')

    heatmap_path = os.path.join(ANALYSIS_OUTPUT_DIR, PROFILE_HEATMAP_FILENAME)
    plt.savefig(heatmap_path, dpi=300)
    print(f"[✓] 角色画像热力图已保存至: '{heatmap_path}'")
    plt.show()

    # --- 7. 保存最终结果 ---
    output_path = os.path.join(ANALYSIS_OUTPUT_DIR, ROLES_FILENAME)
    features_df.to_csv(output_path, index=False)
    print(f"\n--- 用户角色识别完成 ---")
    print(f"[*] 最终的用户角色数据已保存至: '{output_path}'")


if __name__ == "__main__":
    main()